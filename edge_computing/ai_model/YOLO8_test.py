import os
import signal


def _handle_terminal_hangup(_signum, _frame):
    """Survive loss of the remote terminal and its stdout/stderr device."""
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
        finally:
            os.close(null_fd)
    except OSError:
        pass


if __name__ == "__main__" and hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _handle_terminal_hangup)

import cv2
import numpy as np
import socket
import threading
import time
import math
from contextlib import contextmanager
from flask import Flask, Response, request, jsonify
from rknnlite.api import RKNNLite

try:
    import fcntl
except ImportError:
    fcntl = None

os.environ["QT_X11_NO_MITSHM"] = "1"
import logging

app = Flask(__name__)

# 所有随程序部署的资源均以本文件所在目录为基准。systemd 的当前目录、
# SSH 登录目录或手工启动位置变化时，不再影响模型加载。
APP_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
import base64
import serial

# ==========================================
# Doubao & L610 配置参数
# ==========================================
DOUBAO_API_KEY = os.environ.get("DOUBAO_VLM_API_KEY", "")
DOUBAO_ENDPOINT = os.environ.get("DOUBAO_VLM_ENDPOINT", "ep-20260510010330-xdpwb")
AT_PORT = os.environ.get("RV1126_TTS_PORT", "/dev/ttyUSB5")
BAUD_RATE = 115200
TTS_VOLUME = 5
TTS_LOCK_PATH = "/tmp/rv1126_tts.lock"
TTS_MAX_GBK_BYTES_PER_CHUNK = 100
TTS_COMMAND_TIMEOUT_SECONDS = 3.0
TTS_PLAYBACK_TIMEOUT_SECONDS = 30.0
TTS_SERIAL_READ_SLICE_SECONDS = 0.2

global VLM_TARGET_FRAME
VLM_TARGET_FRAME = None  # 用于跨线程传递待识别画面

global VLM_TARGET_FRAME_TS
VLM_TARGET_FRAME_TS = 0.0  # VLM_TARGET_FRAME 最近一次成功更新的时间戳(time.time())；
                            # 用于拒绝复用过期旧帧(例如 8081 干净帧通道断流时)
VLM_TARGET_FRAME_MAX_AGE = 2.0  # 秒；超过此新鲜度阈值的帧视为"无有效帧"，不参与拍照识别

# 新增：用于存储豆包的最新回复文本
global VLM_LATEST_REPLY
VLM_LATEST_REPLY = ""

# 新增：云端 AI 运行状态锁，防止短按/空格连续触发导致重复启动
global VLM_IS_RUNNING
VLM_IS_RUNNING = False
VLM_STATE_LOCK = threading.Lock()

# 工作线程状态：每个长期运行任务都由 _resilient_worker 托管。socket 创建、
# bind 或运行期异常不再让线程永久消失，而是在短暂退避后重新初始化。
WORKER_STATE_LOCK = threading.Lock()
WORKER_STATE = {}
WORKER_RESTART_DELAY_SECONDS = 1.0


def _worker_mark(name, state=None, packet=False, error=None):
    now = time.monotonic()
    with WORKER_STATE_LOCK:
        info = WORKER_STATE.setdefault(name, {
            "state": "starting",
            "restarts": 0,
            "last_packet": None,
            "last_error": "",
        })
        if state is not None:
            info["state"] = state
        if packet:
            info["last_packet"] = now
        if error is not None:
            info["last_error"] = str(error)


def _resilient_worker(name, target):
    """长期任务监督循环：任务异常/意外返回后自动重建，避免静默失效。"""
    while True:
        _worker_mark(name, state="running")
        try:
            target()
            raise RuntimeError("worker returned unexpectedly")
        except Exception as exc:
            with WORKER_STATE_LOCK:
                info = WORKER_STATE.setdefault(name, {})
                info["state"] = "restarting"
                info["restarts"] = info.get("restarts", 0) + 1
                info["last_error"] = str(exc)
            print(f"[WORKER:{name}] 异常退出，{WORKER_RESTART_DELAY_SECONDS:.1f}s 后重启: {exc}")
            time.sleep(WORKER_RESTART_DELAY_SECONDS)


def _start_resilient_worker(name, target):
    thread = threading.Thread(
        target=_resilient_worker,
        args=(name, target),
        name=f"worker-{name}",
        daemon=True,
    )
    thread.start()
    return thread


@contextmanager
def tts_process_lock():
    """与 batch_doubao.py 共用的跨进程串口锁。"""
    lock_file = open(TTS_LOCK_PATH, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
# ==========================================
# 关闭 Flask 的默认 HTTP 请求刷屏打印
# ==========================================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==========================================
# 基础配置参数
# ==========================================
OBJ_THRESH = 0.45
NMS_THRESH = 0.45
IMG_SIZE = 640
MODEL_PATH = os.environ.get("RV1126_YOLO_MODEL", os.path.join(APP_DIR, "yolov8.rknn"))

# ==========================================
# VLM 拍照 ROI 配置：以实时注视点(眼动坐标)为中心跟随移动的固定大小正方形。
# ------------------------------------------
# 裁剪中心只由 (VIRTUAL_X, VIRTUAL_Y) 这两个数值决定——它们是
# rv1126_vision_system.cpp 通过 UDP(5005) 发来的纯坐标数值，本身不含任何
# 图像内容；因此 ROI 位置天然不受下列因素影响：
#   - Python 自己叠加绘制的 YOLO 检测框/"LOCKED: xxx"标签/十字准星
#     (裁剪在 video_processor_thread 里的绘制调用之前完成, 见下方)
#   - cpp 侧烧录进画面里的 "Z: xxxmm" 深度文本、ArUco 标定框、进度条等
#     (这些已经是像素内容, 但裁剪逻辑本身并不"识别"或"依赖"它们,
#      只是裁剪窗口恰好可能覆盖到这些像素——如需彻底避免连像素都不接触，
#      需要 cpp 侧额外发送一份未叠加的干净帧, 属于另一处改动)
# 越界时整体平移裁剪窗口(而非截断), 保证输出恒定为 VLM_ROI_WIDTH x VLM_ROI_HEIGHT。
# 调参: 修改下面几个常量即可, 无需改动截取逻辑代码。
# ==========================================
VLM_ROI_WIDTH = 300          # ROI 宽度(像素, 世界相机原生分辨率空间, 如 1280x720)
VLM_ROI_HEIGHT = 300         # ROI 高度(像素)
VLM_ROI_CENTER = None        # 手动覆盖裁剪中心 (cx, cy)，用于调试；
                              # None(默认) = 跟随实时注视点(VIRTUAL_X, VIRTUAL_Y)移动
VLM_ROI_OFFSET = (0, 0)      # 叠加在注视点上的微调偏移 (dx, dy)，仅在 VLM_ROI_CENTER
                              # 为 None 时生效；用于修正相机光轴与注视坐标系统的系统性偏差


def compute_vlm_roi_rect(frame_w, frame_h, gaze_x, gaze_y, scale=1.0):
    """计算 ROI 裁剪矩形 (x1, y1, x2, y2)。
    裁剪中心只由传入的 (gaze_x, gaze_y) 数值决定(眼动坐标, 非图像内容)，
    VLM_ROI_CENTER 非 None 时可手动覆盖用于调试。不读取、不依赖 frame
    上任何已绘制的叠加内容(深度文本/检测框/标签等)。
    越界时整体平移窗口以保证输出尺寸恒定, 不做截断裁切。

    scale: 当前 frame 相对于"VLM_ROI_WIDTH/HEIGHT 定义所在的原生分辨率
    (1280x720)"的缩放系数。VLM_ROI_* 常量按原生分辨率定义；真正送去裁剪的
    干净帧就是原生分辨率(scale=1.0)。但网页预览用的是降采样到 640x360 的
    叠加流，若仍按绝对像素 300 画黄框，300/640≈47% 宽度，会比真实裁剪区域
    300/1280≈23% 宽度大一倍(面积大4倍)——预览框会"撒谎"。
    调用方按自己 frame 相对原生分辨率的比例传入 scale(如 640x360 预览时传
    frame_w/1280.0)，才能让预览框与实际裁剪区域在真实世界坐标上完全一致。"""
    if VLM_ROI_CENTER is not None:
        cx, cy = VLM_ROI_CENTER
    else:
        cx = int(gaze_x) + int(round(VLM_ROI_OFFSET[0] * scale))
        cy = int(gaze_y) + int(round(VLM_ROI_OFFSET[1] * scale))

    roi_w = min(int(round(VLM_ROI_WIDTH * scale)), frame_w)
    roi_h = min(int(round(VLM_ROI_HEIGHT * scale)), frame_h)
    x1 = max(0, min(frame_w - roi_w, cx - roi_w // 2))
    y1 = max(0, min(frame_h - roi_h, cy - roi_h // 2))
    return x1, y1, x1 + roi_w, y1 + roi_h


def crop_vlm_roi(frame, gaze_x, gaze_y):
    """以注视点 (gaze_x, gaze_y) 为中心截取固定大小的正方形 ROI。
    frame 必须是原生分辨率(1280x720)的干净帧，故 scale 固定为 1.0，
    与 VLM_ROI_WIDTH/HEIGHT 的定义基准一致。"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = compute_vlm_roi_rect(w, h, gaze_x, gaze_y, scale=1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# ==========================================
# 功能开关 & 语音引导 (标定期 C++ 发短码, 这里用 H610 串口播报)
# ==========================================
SNAP_ENABLE = True          # 视线-YOLO 自动吸附总开关: False = 只画裸准星, 不锁物体

SERIAL_LOCK = threading.Lock()   # /dev/ttyUSB5 独占锁: 语音引导 与 VLM 播报 互斥, 防抢串口
VOICE_BUSY = False
VOICE_MAP = {
    "FAR":        "请后退",
    "NEAR":       "请靠近",
    "CONFLICT":   "请调整角度",
    "NOISY":      "请保持不动",
    "DONE":       "标定完成",
    "REUSE_OK":   "已复用上次标定",
}

# YOLO 检测播报与标定语音/VLM 播报使用同一个串口，但触发条件和状态完全独立。
# 使用 monotonic 避免系统时间校正影响 5 秒冷却计时。
YOLO_TTS_COOLDOWN_SECONDS = 5.0
YOLO_TTS_STATE_LOCK = threading.Lock()
YOLO_TTS_NEXT_ALLOWED_AT = 0.0
YOLO_TTS_IN_PROGRESS = False

# 物理按键双击判定窗口（秒）。这是两个 SNAP UDP 包到达 5010 的最大间隔，
# 不是按键触点的理论间隔；窗口越大，双击越不容易被误判为两次单击，
# 但单击拍照的确认延迟也会等比例增加。建议先用 1.0，按实际手速调整。
BUTTON_DOUBLE_PRESS_WINDOW_SECONDS = 1.0

# YOLOv8 识别结果语音播报总开关：由短按按键"双击"切换(见 udp_button_receiver)。
# 只是一个布尔读写，写者(按键线程)与读者(NPU 推理线程)之间沿用本文件中
# SNAP_ENABLE 等同类主开关的写法，不额外加锁；关闭时只跳过语音播报，
# 不影响 YOLOv8 识别、画面推流、检测框绘制等其他逻辑。
YOLO_TTS_ANNOUNCE_ENABLED = True

# ==========================================
# 全局变量定义
# ==========================================
global IS_CALIBRATED, VIRTUAL_X, VIRTUAL_Y, RENDERED_JPEG
global EYE_FRAME
global GLOBAL_BOXES, GLOBAL_CLASSES
global SNAPPED_MODE, SNAPPED_CENTER, LOCKED_GAZE
SNAPPED_MODE = False
SNAPPED_CENTER = (0, 0)
LOCKED_GAZE = (0, 0)
SHIFT_THRESHOLD = 120

global DEBUG_PHASE, DEBUG_MSG, DEBUG_PTS, DEBUG_Z
DEBUG_PHASE = "UNKNOWN"
DEBUG_MSG = "WAITING"
DEBUG_PTS = 0
DEBUG_Z = 0.0

global DEBUG_GY, DEBUG_EYEDY   # 垂直通道诊断: 注视向量 gy / 眼图(瞳孔-模型)垂直位移
DEBUG_GY = 0.0
DEBUG_EYEDY = 0.0

global DEBUG_TOF_VALID, DEBUG_TOF_STD   # ToF 质量: 视线邻域有效格数(0-9) / 深度标准差(mm)
DEBUG_TOF_VALID = 0
DEBUG_TOF_STD = 0.0
DEBUG_PHASE_UPDATED_AT = 0.0
DEBUG_PHASE_CHANGED_AT = 0.0

# 物理按键拍照必须建立在 C++ 已稳定进入正常追踪阶段的基础上。网页手动触发
# 不受该门禁影响；它仍由 IS_CALIBRATED 和帧新鲜度控制。
BUTTON_DEBUG_MAX_AGE_SECONDS = 1.0
BUTTON_TRACKING_STABLE_SECONDS = 1.0

IS_CALIBRATED = False
VIRTUAL_X, VIRTUAL_Y = -100, -100
RENDERED_JPEG = None
# /video_feed must be able to distinguish a newly rendered frame from a
# repeated copy of the last frame.  The monotonic timestamp is also used to
# close an idle MJPEG response so a browser can establish a fresh connection.
RENDERED_FRAME_ID = 0
RENDERED_FRAME_UPDATED_AT = 0.0
VIDEO_STREAM_IDLE_TIMEOUT_SECONDS = 3.0
VIDEO_STREAM_POLL_SECONDS = 0.03
EYE_FRAME = None
GLOBAL_BOXES = []
GLOBAL_CLASSES = []

frame_lock = threading.Lock()

# 渲染线程与 NPU 推理线程解耦所需的共享状态
LATEST_WORLD_FRAME = None      # 最近一帧世界画面(原生分辨率, 不放大)
INFER_LOCK = threading.Lock()
LOCKED_BOX_SHARED = None        # NPU 线程发布给渲染线程的锁定框(API 1280x720 空间)
LOCKED_INFO_SHARED = None
LOCK_STATE_LOCK = threading.Lock()

CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
           "zebra", "giraffe",
           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat",
           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl",
           "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
           "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush")

# 仅供语音播报使用；画面标签仍直接使用上面的 CLASSES 英文名称。
YOLO_CLASS_CN = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机",
    "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船", "traffic light": "红绿灯",
    "fire hydrant": "消防栓", "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛",
    "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包",
    "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "滑雪板", "snowboard": "单板滑雪板", "sports ball": "运动球", "kite": "风筝",
    "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉", "apple": "苹果",
    "sandwich": "三明治", "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕", "chair": "椅子",
    "couch": "沙发", "potted plant": "盆栽", "bed": "床", "dining table": "餐桌", "toilet": "马桶",
    "tv": "电视", "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘",
    "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "sink": "水槽", "refrigerator": "冰箱", "book": "书", "clock": "时钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷",
}

# YOLO 语音播报白名单：仅以下类别在被稳定锁定时播报中文名称，其余类别
# 只在画面上正常显示检测框和原始英文标签，不触发语音。键必须与上面
# CLASSES/YOLO_CLASS_CN 使用的英文类别名完全一致。
YOLO_TTS_ANNOUNCE_WHITELIST = {
    "person", "laptop", "tv", "stop sign", "keyboard", "mouse",
    "cake", "fire hydrant", "clock", "car", "dog", "chair", "backpack",
}


def point_in_box(x, y, box, margin=0):
    """判断 API 坐标的注视点是否仍在有效目标框内。
    margin>0 时对框做等比扩边(仅用于判定, 不影响实际绘制的框)，
    用于在框边缘附近提供一点缓冲，避免注视点在边界上抖动造成锁定框反复闪烁。"""
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def box_iou(box_a, box_b):
    """用于跨 NPU 帧关联同一目标，防止相邻目标之间跳框。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ==========================================
# YOLOv8 解码器
# ==========================================
def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)


def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e_y / e_y.sum(axis=2, keepdims=True)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(axis=2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_w, IMG_SIZE // grid_h]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

# ==========================================
# 新增：用于接收语音与大模型交互的全局变量
# ==========================================
global ASR_LATEST_TEXT, LLM_LATEST_REPLY
ASR_LATEST_TEXT = ""
LLM_LATEST_REPLY = ""

# ==========================================
# 新增：UDP 语音大模型文本接收模块 (Port 5008)
# ==========================================
def udp_nlp_receiver():
    global ASR_LATEST_TEXT, LLM_LATEST_REPLY
    global VOICE_BUSY
    import json
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5008))
    print("[*] Listening for ASR/LLM JSON on port 5008...")

    while True:
        try:
            # 接收缓冲区设大一点，防止大模型长回复被截断
            data, _ = sock.recvfrom(65535)
            _worker_mark("nlp", packet=True)
            info = json.loads(data.decode('utf-8'))

            msg_type = info.get("type")
            if msg_type == "asr":
                text = info.get("text") or ""
                ASR_LATEST_TEXT = text
                if not text.strip():
                    # 语音识别结果为空(常见于说话声音过小导致识别不到有效内容)：
                    # 复用标定语音引导的忙碌标志(VOICE_BUSY)与播放函数(_play_voice/
                    # serial_tts)，保持与项目里其他语音播报完全一致的 busy-skip
                    # 语义——播报忙或 VLM 正在说话时直接跳过(不排队/不阻塞本接收
                    # 循环)，避免重复播报、抢占串口或引入竞态。
                    if not VOICE_BUSY and not VLM_IS_RUNNING:
                        VOICE_BUSY = True
                        threading.Thread(target=_play_voice, args=("请大声一点！",), daemon=True).start()
            elif msg_type == "llm":
                LLM_LATEST_REPLY = info.get("text", "")
        except OSError:
            raise
        except Exception as e:
            pass

# ==========================================
# 眼动坐标(UDP 5005)有效性与在线状态看门狗
# ------------------------------------------
# 背景: 摘下眼镜重新戴上、或长时间运行后, 曾出现锚定点/锁定框卡死在画面
# 某个角落不再更新的问题。根因是 udp_coord_receiver 对收到的坐标不做任何
# 越界校验, 也完全没有"链路/内容是否仍然存活"的判断——recvfrom 是纯阻塞
# 调用且无超时, 一旦 C++ 端因追踪丢失而停止发送/持续发送同一个异常值,
# Python 侧会把最后一次收到的(可能是越界或错误的)坐标永远当作当前注视点,
# IS_CALIBRATED 也不会自动复位, 画面因此"冻结"在那个位置。
# 下面的常量与 _reset_gaze_tracking_state() 为此提供统一的越界拒绝 +
# 存活看门狗机制。
# ==========================================
COORD_SPACE_W = 1280            # VIRTUAL_X/Y 所在的 API 坐标空间宽 (与渲染 sx/sy 换算基准一致)
COORD_SPACE_H = 720
COORD_BOUND_MARGIN = 60         # 允许坐标略微越界的容差(标定误差/边缘余量), 超出视为异常值直接丢弃
COORD_SOCKET_TIMEOUT = 0.3      # recvfrom 超时(秒): 保证即使完全收不到包, 看门狗也能定期被检查
COORD_LOST_RESET_SECONDS = 2.0  # 连续这么久没有"有效且发生变化"的坐标 -> 判定追踪丢失/卡死并回退


def _reset_gaze_tracking_state(reason):
    """坐标看门狗判定追踪丢失(断流/长期无变化/重新进入标定阶段)时的统一回退处理。
    回到等待标定态, 并清空可能已经过期的 VLM 帧与 NPU 锁定框/共享渲染状态,
    避免锚定点或锁定框残留、冻结在失效前的最后位置; 一旦坐标恢复正常,
    既有的 warmup_count 机制会在数帧内自动重新标定, 无需人工干预即可平滑恢复。
    只影响眼动坐标/锁定框相关状态, 不触碰语音引导/VLM/YOLO 播报等独立子系统。"""
    global IS_CALIBRATED, VIRTUAL_X, VIRTUAL_Y, VLM_TARGET_FRAME
    global SNAPPED_MODE, LOCKED_BOX_SHARED, LOCKED_INFO_SHARED, LOCKED_GAZE
    IS_CALIBRATED = False
    VIRTUAL_X, VIRTUAL_Y = -100, -100
    VLM_TARGET_FRAME = None
    with LOCK_STATE_LOCK:
        SNAPPED_MODE = False
        LOCKED_BOX_SHARED = None
        LOCKED_INFO_SHARED = None
    LOCKED_GAZE = (0, 0)
    print(f"⚠️ [STATUS] {reason}，AI 已回到等待标定状态(锚定点/锁定框/VLM_TARGET_FRAME 已清空)")


# ==========================================
# UDP 接收模块
# ==========================================
def udp_coord_receiver():
    global VIRTUAL_X, VIRTUAL_Y, IS_CALIBRATED, VLM_TARGET_FRAME
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5005))
    # 超时而非纯阻塞: 保证即使 C++ 完全停止发送坐标(摘下眼镜/进程卡死/链路断开)，
    # 循环仍能定期醒来检查下面的存活看门狗，而不是永远阻塞在 recvfrom 上。
    sock.settimeout(COORD_SOCKET_TIMEOUT)

    warmup_count = 0
    recalib_count = 0   # 连续收到占位坐标(640,360)的帧数：用于探测 C++ 重新回到标定阶段
    THRESHOLD = 5

    # warmup 期间是否已经观察到坐标发生过变化——"复用上次标定"跳过标定手势后，
    # 追踪器/相机可能还没真正稳定，短暂输出一个静止不变的占位/默认坐标(例如
    # 落在画面右下角)；若只数包数不看是否变化，这个静止值会在 THRESHOLD 帧内
    # 被误判为"已标定"并当作锚定点固定显示——这正是"一启动就卡死在角落"而非
    # "运行一段时间后才失效"的根因。要求 warmup 窗口内至少观察到一次真实变化
    # 才允许转为已标定，杜绝静止占位值被当成有效锚定点。
    warmup_last_xy = None
    warmup_saw_change = False

    last_alive_time = time.time()   # 最近一次"有效且发生变化"的坐标时间，用于看门狗判定
    last_seen_xy = None              # 上一次被接受的(vx,vy)，用于识别"数值长期原地不动"

    print("--> [UDP Coord] Receiver Started. Waiting for VALID movement...")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            _worker_mark("coord", packet=True)
        except socket.timeout:
            data = None
        except OSError:
            raise
        except Exception as e:
            print(f"⚠️ [UDP Error] {e}")
            data = None

        # 存活看门狗: 不管本轮是完全没收到包(data is None)还是收到了包但下面
        # 解析/校验后被判定无效或未变化，只要已经连续 COORD_LOST_RESET_SECONDS
        # 没有刷新过 last_alive_time，就说明追踪已经丢失/卡死——主动回退到等待
        # 标定态，而不是让锚定点/锁定框永远冻结在失效前的最后一个位置。
        if IS_CALIBRATED and (time.time() - last_alive_time) > COORD_LOST_RESET_SECONDS:
            _reset_gaze_tracking_state(
                f"眼动坐标连续 {COORD_LOST_RESET_SECONDS:.1f}s 未见到有效变化"
                f"(可能摘下眼镜/追踪丢失/链路断开)")
            warmup_count = 0
            recalib_count = 0
            warmup_last_xy = None
            warmup_saw_change = False
            last_seen_xy = None
            last_alive_time = time.time()   # 避免在恢复前的每一圈都重复触发重置/刷屏

        if data is None:
            continue

        try:
            raw_data = data.decode('utf-8').strip()
            coords = raw_data.split(',')
            if len(coords) < 2:
                continue

            vx = int(float(coords[0]))
            vy = int(float(coords[1]))

            if vx == 640 and vy == 360:
                VIRTUAL_X, VIRTUAL_Y = vx, vy
                last_alive_time = time.time()   # 占位包代表链路仍存活，避免与断流看门狗误叠加
                last_seen_xy = None              # 占位坐标不参与"数值卡死"判定
                if not IS_CALIBRATED:
                    warmup_count = 0
                    recalib_count = 0
                    warmup_last_xy = None
                    warmup_saw_change = False
                else:
                    # IS_CALIBRATED 一旦置真不会自动复位；若 C++ 重启后重新回到
                    # 标定阶段(持续发占位坐标)，这里必须主动解锁，否则旧的
                    # IS_CALIBRATED/VLM_TARGET_FRAME 会在标定期间被误判为"已标定"，
                    # 导致短按按键误触发拍照识别。
                    recalib_count += 1
                    if recalib_count >= THRESHOLD:
                        _reset_gaze_tracking_state("检测到 C++ 重新进入标定阶段")
                        warmup_count = 0
                        recalib_count = 0
                        warmup_last_xy = None
                        warmup_saw_change = False
                continue

            # 越界/异常值防护: 拒绝明显超出画面坐标空间的数值(解析异常、姿态解算
            # 发散等)，不覆盖 VIRTUAL_X/Y、也不算作"存活"——防止把这类噪声坐标
            # 当真显示，导致锚定点瞬移或被钉死在错误位置。
            if not (-COORD_BOUND_MARGIN <= vx <= COORD_SPACE_W + COORD_BOUND_MARGIN and
                    -COORD_BOUND_MARGIN <= vy <= COORD_SPACE_H + COORD_BOUND_MARGIN):
                continue

            if last_seen_xy is None or (vx, vy) != last_seen_xy:
                # 坐标确实发生了变化(或刚从占位/异常态恢复)：真实眼动几乎不可能
                # 长时间原地不动(存在生理性微跳视/传感器噪声)，视为链路与追踪
                # 均正常存活，刷新看门狗计时。
                last_alive_time = time.time()
                last_seen_xy = (vx, vy)

            VIRTUAL_X, VIRTUAL_Y = vx, vy
            recalib_count = 0

            if not IS_CALIBRATED:
                if warmup_last_xy is not None and (vx, vy) != warmup_last_xy:
                    warmup_saw_change = True
                warmup_last_xy = (vx, vy)
                warmup_count += 1
                # 必须同时满足"样本数够"和"期间确实变化过"才转为已标定：
                # 只数样本数不看变化，会把追踪器初始化瞬间输出的静止占位坐标
                # (常见于跳过标定手势的"复用上次标定"场景)误当成真实注视锁定，
                # 一启动就把锚定点钉死在那个占位坐标上。warmup_count 达标后若
                # 仍未见变化则继续观察(不清零计数)，一旦后续任意一帧出现变化，
                # 立即在该帧完成标定，不会额外拖延。
                if warmup_count >= THRESHOLD and warmup_saw_change:
                    IS_CALIBRATED = True
                    print("✅ [STATUS] Valid Gaze Stream Detected. AI UNLOCKED!")

        except Exception as e:
            print(f"⚠️ [UDP Error] {e}")


def udp_debug_receiver():
    global DEBUG_PHASE, DEBUG_MSG, DEBUG_PTS, DEBUG_Z
    global DEBUG_GY, DEBUG_EYEDY
    global DEBUG_TOF_VALID, DEBUG_TOF_STD
    global DEBUG_PHASE_UPDATED_AT, DEBUG_PHASE_CHANGED_AT
    import json
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5006))
    print("[*] Listening for DEBUG JSON on port 5006...")

    while True:
        try:
            data, _ = sock.recvfrom(1024)
            _worker_mark("debug", packet=True)
            info = json.loads(data.decode('utf-8'))

            new_phase = info.get('phase', 'UNKNOWN')
            now = time.monotonic()
            if new_phase != DEBUG_PHASE:
                DEBUG_PHASE_CHANGED_AT = now
            DEBUG_PHASE = new_phase
            DEBUG_PHASE_UPDATED_AT = now
            DEBUG_MSG = info.get('msg', 'NORMAL')
            DEBUG_PTS = info.get('pts', 0)
            DEBUG_Z = info.get('z', 0.0)
            DEBUG_GY = info.get('gy', 0.0)
            DEBUG_EYEDY = info.get('eye_dy', 0.0)
            DEBUG_TOF_VALID = info.get('tof_valid', 0)
            DEBUG_TOF_STD = info.get('tof_std', 0.0)

        except OSError:
            raise
        except Exception as e:
            pass


# ==========================================
# 新增：短按按键 UDP 拍照触发接收 (Port 5010)
# GPIO 由 batch_doubao.py 独占 (gpiochip5 line 1, 高电平有效)；短按时它通过本地 UDP
# 发 "SNAP" 到此端口, 触发 process_vlm_task 拍照上传豆包 (与空格 /ask_doubao 等价)。
# 整个流程仅在内存中处理 VLM_TARGET_FRAME, 绝不落盘。
#
# 单击 / 双击消歧(互斥, 不会同时触发):
# 收到第一次 SNAP 后并不立即拍照, 而是启动一个短暂的异步确认计时器；
# 若窗口内又收到第二次 SNAP, 判定为双击，单击拍照逻辑因此完全不会执行，
# 只切换 YOLO 播报开关；若窗口内没有第二次 SNAP，计时器到期后执行单击拍照。
# DEBOUNCE_MIN_INTERVAL_SECONDS 用于过滤按键抖动/重复包, 防止其被误判为
# "第二次按键"。计时器带有递增事件代号，旧回调即使已经排队也不能执行拍照。
# ==========================================
def submit_vlm_task(source):
    """统一提交 VLM 任务，保证网页和物理按键共用同一并发/新鲜度规则。"""
    global VLM_IS_RUNNING

    with VLM_STATE_LOCK:
        frame_age = (time.time() - VLM_TARGET_FRAME_TS) if VLM_TARGET_FRAME_TS else None
        if VLM_IS_RUNNING:
            return False, "AI正在运行中", frame_age
        if source == "button":
            now = time.monotonic()
            debug_age = now - DEBUG_PHASE_UPDATED_AT if DEBUG_PHASE_UPDATED_AT else None
            phase_stable_age = now - DEBUG_PHASE_CHANGED_AT if DEBUG_PHASE_CHANGED_AT else 0.0
            if debug_age is None or debug_age > BUTTON_DEBUG_MAX_AGE_SECONDS:
                return False, "C++阶段状态不可用或已过期", frame_age
            if DEBUG_PHASE != "Anchoring":
                return False, f"当前处于{DEBUG_PHASE}阶段，禁止按键拍照", frame_age
            if phase_stable_age < BUTTON_TRACKING_STABLE_SECONDS:
                return False, "刚进入正常追踪阶段，暂缓按键拍照", frame_age
        if not IS_CALIBRATED:
            return False, "尚未标定完成", frame_age
        if VLM_TARGET_FRAME is None:
            return False, "暂无待识别帧", frame_age
        if frame_age is None or frame_age > VLM_TARGET_FRAME_MAX_AGE:
            age_text = "未知" if frame_age is None else f"{frame_age:.1f}s"
            return False, f"待识别帧已过期({age_text})", frame_age

        frame_copy = VLM_TARGET_FRAME.copy()
        VLM_IS_RUNNING = True

    try:
        threading.Thread(
            target=process_vlm_task,
            args=(frame_copy,),
            name=f"vlm-{source}",
            daemon=True,
        ).start()
        return True, "任务已启动", frame_age
    except Exception:
        with VLM_STATE_LOCK:
            VLM_IS_RUNNING = False
        raise


def udp_button_receiver():
    global YOLO_TTS_ANNOUNCE_ENABLED
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5010))
    print("[*] Listening for short-press PHOTO trigger on port 5010...")

    # 单击只增加这个确认延迟，原有拍照、录音和复用流程仍在各自线程/进程中运行。
    DOUBLE_PRESS_WINDOW_SECONDS = BUTTON_DOUBLE_PRESS_WINDOW_SECONDS
    DEBOUNCE_MIN_INTERVAL_SECONDS = 0.05  # 过滤同一次按压产生的重复 SNAP

    btn_state_lock = threading.Lock()
    btn_state = {
        "pending_timer": None,
        "pending_deadline": 0.0,
        "generation": 0,
        "last_event_time": 0.0,
    }

    def run_single_press_action(generation):
        """只允许当前事件代号的计时器执行一次单击拍照。"""
        with btn_state_lock:
            if generation != btn_state["generation"]:
                return
            if btn_state["pending_timer"] is None:
                return
            remaining = btn_state["pending_deadline"] - time.monotonic()
            if remaining > 0:
                retry = threading.Timer(
                    remaining,
                    run_single_press_action,
                    args=(generation,),
                )
                retry.daemon = True
                btn_state["pending_timer"] = retry
                retry.start()
                return
            btn_state["pending_timer"] = None
            btn_state["pending_deadline"] = 0.0
        try:
            started, reason, frame_age = submit_vlm_task("button")
            if started:
                print(f"🚀 [BTN] 确认单击SNAP(非双击) -> 已标定, 待识别帧新鲜度={frame_age:.2f}s, 启动豆包云端拍照识别...")
            else:
                print(f"⏸️ [BTN] 确认单击SNAP但忽略拍照识别 (原因: {reason})")
        except Exception as e:
            print(f"⚠️ [BTN] 单击拍照延迟执行异常: {e}")

    while True:
        try:
            data, _ = sock.recvfrom(64)
            _worker_mark("button", packet=True)
            cmd = data.decode('utf-8', errors='ignore').strip()
            if cmd != "SNAP":
                continue

            now_press = time.monotonic()
            is_double = False
            timer = None
            with btn_state_lock:
                if (now_press - btn_state["last_event_time"]) < DEBOUNCE_MIN_INTERVAL_SECONDS:
                    # 与上一次接受的事件过近，判定为同一次物理按压抖动。
                    continue
                btn_state["last_event_time"] = now_press

                pending_timer = btn_state["pending_timer"]
                pending_deadline = btn_state["pending_deadline"]
                if pending_timer is not None and now_press <= pending_deadline:
                    # 第二次 SNAP 在窗口内到达：取消当前代号的单击动作。
                    btn_state["generation"] += 1
                    pending_timer.cancel()
                    btn_state["pending_timer"] = None
                    btn_state["pending_deadline"] = 0.0
                    is_double = True
                else:
                    # 第一次 SNAP：异步挂起单击动作，接收线程继续处理其余数据。
                    btn_state["generation"] += 1
                    generation = btn_state["generation"]
                    deadline = now_press + DOUBLE_PRESS_WINDOW_SECONDS
                    timer = threading.Timer(
                        DOUBLE_PRESS_WINDOW_SECONDS,
                        run_single_press_action,
                        args=(generation,),
                    )
                    timer.daemon = True
                    btn_state["pending_timer"] = timer
                    btn_state["pending_deadline"] = deadline

            if is_double:
                YOLO_TTS_ANNOUNCE_ENABLED = not YOLO_TTS_ANNOUNCE_ENABLED
                state_text = "开启" if YOLO_TTS_ANNOUNCE_ENABLED else "关闭"
                print(f"🔁 [BTN] 确认双击短按 -> YOLOv8 识别语音播报已{state_text}(本次不触发拍照)")
            elif timer is not None:
                timer.start()
        except OSError:
            raise
        except Exception:
            pass


def udp_eye_receiver():
    global EYE_FRAME
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 8082))
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            _worker_mark("eye", packet=True)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                EYE_FRAME = cv2.resize(frame, (240, 180))
        except OSError:
            raise
        except Exception:
            pass


# ==========================================
# 新增：接收 C++ 发来的"干净世界帧"(端口 8081, 原生分辨率, 未叠加任何
# ArUco标定框/举牌进度条/"Z: xxxmm"深度文本等HUD内容) —— 专供 VLM 拍照
# ROI 截取使用，与网页预览用的 8080 叠加流彻底分开，避免叠加内容被裁进
# 送给豆包的识别图里。裁剪中心用 (VIRTUAL_X, VIRTUAL_Y) 原值(该干净帧就是
# 原生 1280x720, 与坐标空间 1:1，无需 sx/sy 缩放)。
# ==========================================
def udp_world_clean_receiver():
    global VLM_TARGET_FRAME, VLM_TARGET_FRAME_TS
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("127.0.0.1", 8081))
    sock.setblocking(False)
    print("[*] Listening for CLEAN world frame (VLM ROI source) on port 8081...")

    got_first_frame = False
    last_frame_time = time.time()
    last_warn_time = 0.0

    while True:
        img_bytes = None
        try:
            while True:
                data, _ = sock.recvfrom(65535)
                _worker_mark("world_clean", packet=True)
                img_bytes = data
        except BlockingIOError:
            pass

        if img_bytes is None:
            time.sleep(0.01)
            # 看门狗：长时间收不到任何数据包(不只是解码失败)时报警一次(节流)。
            # 覆盖两种情况: 启动后一直没收到过(首次未连通) / 曾经正常但后来断流
            # (例如 cpp 端异常退出、或编码后超限被丢弃)——两者都不应静默无提示。
            now = time.time()
            if (now - last_frame_time) > 5.0 and (now - last_warn_time) > 10.0:
                last_warn_time = now
                if not got_first_frame:
                    print("⚠️ [WORLD-CLEAN] 已 5 秒以上未收到端口 8081 的任何数据包！"
                          "VLM_TARGET_FRAME 无法更新，短按拍照会一直报'暂无待识别帧'。"
                          "请检查 cpp 端是否已用最新代码重新编译/运行，"
                          "以及 cpp 侧是否打印过 '⚠️ [UDP Frame Drop] ... dest_port=8081'。")
                else:
                    print(f"⚠️ [WORLD-CLEAN] 距上一次收到有效帧已 {now - last_frame_time:.1f}s，"
                          "8081 通道疑似断流(cpp 进程可能已退出/崩溃)，"
                          "VLM_TARGET_FRAME 将逐渐过期，短按拍照会被陈旧帧检测拦截。")
            continue

        try:
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ [WORLD-CLEAN] 收到数据包但 JPEG 解码失败(空帧/损坏数据), 已丢弃本包")
                continue
            last_frame_time = time.time()
            if not got_first_frame:
                got_first_frame = True
                print(f"✅ [WORLD-CLEAN] 已收到首帧干净世界帧, 尺寸={frame.shape[1]}x{frame.shape[0]}, VLM ROI 可正常工作")
            roi = crop_vlm_roi(frame, VIRTUAL_X, VIRTUAL_Y)
            if roi is not None:
                VLM_TARGET_FRAME = roi
                VLM_TARGET_FRAME_TS = last_frame_time
        except Exception as e:
            print(f"⚠️ [WORLD-CLEAN] 解码/裁剪异常: {e}")


# ==========================================
# 渲染线程（轻量·稳定 30fps）：抽最新 UDP 帧 -> 画准星/锁定框 -> 编码。
# 不做任何 NPU 推理，画面永不被推理拖卡。坐标统一在 API(1280x720) 空间。
# ==========================================
def video_processor_thread():
    global RENDERED_JPEG, RENDERED_FRAME_ID, RENDERED_FRAME_UPDATED_AT, LATEST_WORLD_FRAME

    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    video_sock.bind(('127.0.0.1', 8080))
    video_sock.setblocking(False)  # 非阻塞，抽干缓冲只取最新帧

    while True:
        # 1. 极致低延迟：抽空 UDP 缓冲区，只拿最新一帧
        img_bytes = None
        try:
            while True:
                data, addr = video_sock.recvfrom(65535)
                _worker_mark("world", packet=True)
                img_bytes = data
        except BlockingIOError:
            pass  # 缓冲区已空

        if img_bytes is None:
            time.sleep(0.003)
            continue

        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 关键修正：不再 cv2.resize 放大到 1280x720。保持原生分辨率，
        # 既省两次大缩放的算力，又让 NPU 线程拿到未被放大糊化的原图。
        with INFER_LOCK:
            LATEST_WORLD_FRAME = img
        H, W = img.shape[:2]
        sx, sy = W / 1280.0, H / 720.0   # API(1280x720) -> 显示(原生) 缩放

        # 注: VLM 拍照 ROI 的实际截取在 udp_world_clean_receiver 里进行(基于 C++
        # 发来的原生分辨率"干净帧", 端口 8081), 与这里的 640x360 预览渲染完全
        # 独立——去掉黄框调试可视化不影响真实的 VLM 截取/上传流程。

        # 2. 轻量渲染（不含任何推理）
        if not IS_CALIBRATED:
            cv2.putText(img, "EYE...", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        else:
            # 成对读取，避免渲染线程读到新框却配上旧标签。
            with LOCK_STATE_LOCK:
                snapped = SNAPPED_MODE
                box = LOCKED_BOX_SHARED
                locked_info = LOCKED_INFO_SHARED
            if snapped and box is not None:
                # YOLOv8 识别图层：与图片识别(VLM)功能相同的无锚定点图层，
                # 只画框和标签，不叠加锚定点标记。
                bx1, by1, bx2, by2 = box
                cv2.rectangle(img, (int(bx1 * sx), int(by1 * sy)), (int(bx2 * sx), int(by2 * sy)), (0, 0, 255), 3)
                if locked_info is not None:
                    label, score = locked_info
                    cv2.putText(img, f"LOCKED: {label} {score:.2f}",
                                (int(bx1 * sx), max(18, int(by1 * sy) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.circle(img, (int(VIRTUAL_X * sx), int(VIRTUAL_Y * sy)), 7, (0, 255, 0), -1)

        # 3. 编码为 JPEG 供 Flask 前端渲染
        if (W, H) != (640, 360):
            img = cv2.resize(img, (640, 360))
        ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret:
            with frame_lock:
                RENDERED_JPEG = buffer.tobytes()
                RENDERED_FRAME_ID += 1
                RENDERED_FRAME_UPDATED_AT = time.monotonic()


# ==========================================
# 独立 NPU 线程：对世界画面做整幅 letterbox 检测（不放大、不裁小块），
# 再挑离视线最近的目标做磁滞吸附。与渲染彻底解耦，NPU 慢也不卡画面。
# ==========================================
def npu_inference_thread():
    global GLOBAL_BOXES, GLOBAL_CLASSES
    global SNAPPED_MODE, LOCKED_GAZE
    global LOCKED_BOX_SHARED, LOCKED_INFO_SHARED

    IS_SNAPPED = False
    LOCK_C = None
    LOCK_BOX = None
    LOCK_INFO = None

    TRIGGER_RADIUS = 100        # API(1280x720) 像素（≈原 50 的 2 倍）
    ANCHOR_BOX_MARGIN = 15      # 锚定点"仍在框内"判定的等比扩边缓冲, 防止边缘抖动闪烁
    CONTINUITY_IOU_THRESH = 0.3 # 与上一帧锁定框的最小 IOU, 用于在本帧漏检时判断"仍是同一目标"
    NPU = 640
    last_vx, last_vy = 0, 0

    while True:
        if not IS_CALIBRATED or LATEST_WORLD_FRAME is None:
            # 未标定期间(含标定中、以及坐标看门狗判定追踪丢失后主动回退到此状态)：
            # 持续清空本地与共享的锁定状态、并重置速度基准。否则一旦标定恢复，
            # 本线程会在第一时间把断线前残留的旧锁定框重新发布出去，
            # 或用断线前的旧坐标当基准算出一次巨大的虚假速度——这正是
            # "重新戴上眼镜后锚定点/锁定框仍卡在最后位置"的另一个成因。
            if IS_SNAPPED or LOCKED_BOX_SHARED is not None:
                IS_SNAPPED = False
                LOCK_C = None
                LOCK_BOX = None
                LOCK_INFO = None
                with LOCK_STATE_LOCK:
                    SNAPPED_MODE = False
                    LOCKED_BOX_SHARED = None
                    LOCKED_INFO_SHARED = None
            last_vx, last_vy = 0, 0
            time.sleep(0.01)
            continue

        with INFER_LOCK:
            frame = LATEST_WORLD_FRAME.copy()
        H, W = frame.shape[:2]
        vx, vy = int(VIRTUAL_X), int(VIRTUAL_Y)          # API 空间

        # 眼动速度熔断：扫视时不喂 NPU（省算力，扫视中的画面本来就模糊，喂了也白喂）。
        # 但扫视意味着用户已经不再注视原来锁定的目标，必须立即释放锁定并清空共享状态，
        # 而不是简单 continue 冻结在原地——否则渲染线程会在整个扫视期间持续显示
        # 一个"过期"的红框，也就是拖影/滞留框问题的直接来源。
        velocity = math.sqrt((vx - last_vx) ** 2 + (vy - last_vy) ** 2)
        last_vx, last_vy = vx, vy
        if velocity >= 80:      # API 空间阈值（≈原 40 的 2 倍）
            if IS_SNAPPED or LOCKED_BOX_SHARED is not None:
                IS_SNAPPED = False
                LOCK_C = None
                LOCK_BOX = None
                LOCK_INFO = None
                with LOCK_STATE_LOCK:
                    SNAPPED_MODE = False
                    LOCKED_BOX_SHARED = None
                    LOCKED_INFO_SHARED = None
            time.sleep(0.005)
            continue

        # --- letterbox 原生帧 -> 640x640（只补边，绝不放大）---
        r = min(NPU / W, NPU / H)
        nw, nh = int(round(W * r)), int(round(H * r))
        pad_x, pad_y = (NPU - nw) // 2, (NPU - nh) // 2
        canvas = np.zeros((NPU, NPU, 3), dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(frame, (nw, nh))

        try:
            img_in = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            outputs = rknn.inference(inputs=[np.expand_dims(img_in, 0)])
            _worker_mark("npu", packet=True)
            boxes, classes, scores = post_process(outputs)
            if boxes is None:
                boxes, classes, scores = [], [], []
        except Exception as ai_e:
            print(f"❌ [AI 推理崩溃] {ai_e}")
            boxes, classes, scores = [], [], []

        ax, ay = 1280.0 / W, 720.0 / H      # 原生 -> API
        temp_boxes, temp_classes, temp_scores = [], [], []
        best_c = None
        best_box = None
        best_info = None
        closest = float('inf')

        for box, score, cls_id in zip(boxes, scores, classes):
            if score < 0.5:
                continue
            # NPU 坐标 -> 原生 -> API
            bx1 = int((box[0] - pad_x) / r * ax)
            by1 = int((box[1] - pad_y) / r * ay)
            bx2 = int((box[2] - pad_x) / r * ax)
            by2 = int((box[3] - pad_y) / r * ay)
            ocx, ocy = (bx1 + bx2) // 2, (by1 + by2) // 2
            temp_boxes.append((bx1, by1, bx2, by2))
            temp_classes.append(cls_id)
            temp_scores.append(score)
            in_box = (bx1 <= vx <= bx2) and (by1 <= vy <= by2)
            dist = math.sqrt((vx - ocx) ** 2 + (vy - ocy) ** 2)
            if in_box or dist < TRIGGER_RADIUS:
                if dist < closest:
                    closest = dist
                    best_c = (ocx, ocy)
                    best_box = (bx1, by1, bx2, by2)
                    best_info = (CLASSES[cls_id], score)

        GLOBAL_BOXES = temp_boxes
        GLOBAL_CLASSES = temp_classes

        if SNAP_ENABLE:
            # 磁滞吸附状态机（不覆盖 VIRTUAL_X/Y，避免与 UDP 抢写）
            if not IS_SNAPPED:
                if best_c is not None:
                    IS_SNAPPED = True
                    LOCK_C = best_c
                    LOCK_BOX = best_box
                    LOCK_INFO = best_info
                    LOCKED_GAZE = (vx, vy)
            else:
                # 折中方案: 锚定点(视线)仍在锁定框内 或 目标本帧仍被稳定跟踪时保留框,
                # 尽量减少"单帧漏检"造成的闪烁；一旦锚定点离开框且目标也不再有效
                # (本帧未命中、也无法通过 IOU 关联到同一目标)，立即释放, 不残留。
                anchor_in_box = (LOCK_BOX is not None) and point_in_box(vx, vy, LOCK_BOX, margin=ANCHOR_BOX_MARGIN)
                continuation_box, continuation_info = None, None
                if best_c is not None:
                    # 本帧正常命中(视线在框内或在触发半径内): 直接延续
                    continuation_box, continuation_info = best_box, best_info
                elif LOCK_BOX is not None:
                    # 本帧未命中(视线可能已略微偏出触发半径), 尝试用 IOU 判断
                    # 检测到的框里是否仍有一个与上一帧锁定框高度重叠的——
                    # 如果有, 说明目标仍被稳定跟踪(只是本帧漏检/评分不足),
                    # 而非目标真的消失, 用它延续锁定, 避免闪烁。
                    best_iou = CONTINUITY_IOU_THRESH
                    for tb, ts, tc in zip(temp_boxes, temp_scores, temp_classes):
                        iou = box_iou(tb, LOCK_BOX)
                        if iou > best_iou:
                            best_iou = iou
                            continuation_box = tb
                            continuation_info = (CLASSES[tc], ts)

                if continuation_box is not None:
                    bx1, by1, bx2, by2 = continuation_box
                    LOCK_C = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                    LOCK_BOX = continuation_box
                    LOCK_INFO = continuation_info
                elif not anchor_in_box:
                    # 锚定点已经离开目标框, 且本帧也没有可关联的目标: 立即释放
                    IS_SNAPPED = False
                    LOCK_C = None
                    LOCK_BOX = None
                    LOCK_INFO = None
                # else: 锚定点仍在框内但本帧漏检——保留旧框不变, 等待下一帧恢复检测

            with LOCK_STATE_LOCK:
                SNAPPED_MODE = IS_SNAPPED
                LOCKED_BOX_SHARED = LOCK_BOX
                LOCKED_INFO_SHARED = LOCK_INFO

            # YOLOv8 识别结果语音播报: 仅在目标被稳定锁定时尝试触发, 内部按
            # 独立冷却/忙碌状态非阻塞判断, 不影响本线程后续帧的实时性。
            if IS_SNAPPED and LOCK_INFO is not None:
                maybe_announce_yolo_detection(LOCK_INFO[0])
        else:
            # 吸附功能关闭: 不锁物体, 渲染线程只画裸准星(检测框/VLM 仍照常工作)
            IS_SNAPPED = False
            with LOCK_STATE_LOCK:
                SNAPPED_MODE = False
                LOCKED_BOX_SHARED = None
                LOCKED_INFO_SHARED = None


# ==========================================
# 共享串口 TTS (H610/L610): 语音引导 与 VLM 播报都走这里, SERIAL_LOCK 互斥防抢串口
# ==========================================
def _split_tts_text(text):
    """Split on GBK bytes, because L610 limits the AT+GTTS text buffer by bytes."""
    clean = str(text).replace('\n', '，').replace('\r', '')
    clean = clean.replace('"', '').replace('*', '').replace('#', '')
    clean = clean.encode('gbk', errors='ignore').decode('gbk').strip()
    chunks = []
    break_chars = "，。！？；：、,.!?;:"

    while clean:
        used_bytes = 0
        hard_end = 0
        for index, char in enumerate(clean):
            char_bytes = len(char.encode('gbk'))
            if used_bytes + char_bytes > TTS_MAX_GBK_BYTES_PER_CHUNK:
                break
            used_bytes += char_bytes
            hard_end = index + 1

        if hard_end == 0:
            break
        if hard_end == len(clean):
            chunks.append(clean)
            break

        natural_end = 0
        for index, char in enumerate(clean[:hard_end]):
            if char in break_chars:
                natural_end = index + 1
        if natural_end < hard_end // 2:
            natural_end = hard_end

        chunk = clean[:natural_end]
        if chunk:
            chunks.append(chunk)
        clean = clean[natural_end:]

    return chunks


def _serial_read_until(ser, marker, timeout_seconds, stage):
    deadline = time.monotonic() + timeout_seconds
    response = bytearray()
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        data = ser.read(waiting if waiting > 0 else 1)
        if not data:
            continue
        response.extend(data)
        if marker in response:
            return bytes(response)
        if (b"\r\nERROR\r\n" in response
                or b"\r\n+CME ERROR:" in response
                or b"\r\n+CMS ERROR:" in response):
            detail = response.decode('gbk', errors='replace').strip()
            raise RuntimeError(f"{stage} returned an error: {detail}")

    detail = response.decode('gbk', errors='replace').strip()
    raise TimeoutError(f"{stage} timed out after {timeout_seconds:.0f}s: {detail or 'no response'}")


def _serial_command_ok(ser, payload, stage):
    ser.write(payload)
    ser.flush()
    return _serial_read_until(
        ser,
        b"\r\nOK\r\n",
        TTS_COMMAND_TIMEOUT_SECONDS,
        stage,
    )


def serial_tts(text):
    chunks = _split_tts_text(text)
    if not chunks:
        return False

    with SERIAL_LOCK, tts_process_lock():
        try:
            with serial.Serial(
                AT_PORT,
                BAUD_RATE,
                timeout=TTS_SERIAL_READ_SLICE_SECONDS,
                write_timeout=2,
                exclusive=True,
            ) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                _serial_command_ok(
                    ser,
                    f"AT+CLVL={TTS_VOLUME}\r\n".encode('ascii'),
                    "AT+CLVL",
                )
                time.sleep(0.1)

                # This L610 firmware rejects CSCS="GBK"; GTTS accepts raw GBK bytes directly.
                for index, chunk in enumerate(chunks, start=1):
                    ser.reset_input_buffer()
                    payload = b'AT+GTTS=1,"' + chunk.encode('gbk') + b'",1\r\n'
                    ack = _serial_command_ok(ser, payload, f"AT+GTTS chunk {index}/{len(chunks)}")
                    if b"+TTS: END" not in ack:
                        _serial_read_until(
                            ser,
                            b"+TTS: END",
                            TTS_PLAYBACK_TIMEOUT_SECONDS,
                            f"TTS playback chunk {index}/{len(chunks)}",
                        )
            return True
        except serial.SerialException as e:
            print(f"❌ [SERIAL TTS] 串口失败，L610 可能正在重启: {e}")
        except (TimeoutError, RuntimeError) as e:
            print(f"❌ [SERIAL TTS] 协议失败: {e}")
        except Exception as e:
            print(f"❌ [SERIAL TTS] {e}")
        return False


# ==========================================
# 标定语音引导接收 (Port 5009): C++ 发短码 -> 播报中文; busy-skip 防堆积/过载
# ==========================================
def udp_voice_receiver():
    global VOICE_BUSY
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5009))
    print("[*] Listening for calibration VOICE codes on port 5009...")
    warned_unknown_codes = set()   # 兜底: 每个未知语音码只警告一次, 避免刷屏
    while True:
        try:
            data, _ = sock.recvfrom(64)
            _worker_mark("voice", packet=True)
            code = data.decode('utf-8', errors='ignore').strip()
            phrase = VOICE_MAP.get(code)

            if phrase is None:
                # 兜底处理: 收到的语音码不在当前可用资源(VOICE_MAP)里——可能是
                # cpp 端还没更新到最新代码、协议不一致等——只忽略播报并提示一次，
                # 绝不因缺失语音资源而抛异常或影响其他功能。
                if code and code not in warned_unknown_codes:
                    warned_unknown_codes.add(code)
                    print(f"⚠️ [VOICE] 收到未知语音码 '{code}'，不在当前可用键 "
                          f"{sorted(VOICE_MAP.keys())} 内，已忽略播报")
                continue

            # 播报忙 / VLM 正在说话 时直接丢弃, 保证不过载、不抢串口
            if not VOICE_BUSY and not VLM_IS_RUNNING:
                VOICE_BUSY = True
                threading.Thread(target=_play_voice, args=(phrase,), daemon=True).start()
        except OSError:
            raise
        except Exception as e:
            print(f"⚠️ [VOICE] 接收/处理异常: {e}")


def _play_voice(phrase):
    global VOICE_BUSY
    try:
        serial_tts(phrase)
    finally:
        VOICE_BUSY = False


# ==========================================
# YOLOv8 识别结果语音播报：触发条件与状态(YOLO_TTS_*)完全独立于
# 标定语音引导(VOICE_BUSY)与 VLM 拍照播报(VLM_IS_RUNNING)，
# 不复用、不影响它们各自的状态机；三者仅通过 serial_tts 内部的
# SERIAL_LOCK 互斥共享同一条物理串口。
# ==========================================
def _yolo_tts_worker(label_cn):
    """独立线程执行实际串口播报，播报期间绝不占用 NPU/渲染线程；
    播报结束(含异常)后才开始计算下一次 5 秒冷却，满足"每次播报后冷却 5 秒"。"""
    global YOLO_TTS_IN_PROGRESS, YOLO_TTS_NEXT_ALLOWED_AT
    try:
        serial_tts(label_cn)
    finally:
        with YOLO_TTS_STATE_LOCK:
            YOLO_TTS_IN_PROGRESS = False
            YOLO_TTS_NEXT_ALLOWED_AT = time.monotonic() + YOLO_TTS_COOLDOWN_SECONDS


def maybe_announce_yolo_detection(label_en):
    """由 NPU 推理线程在每帧目标被稳定锁定时调用。
    只做一次极短的加锁状态检查(非阻塞)，真正播报交给独立线程执行，
    因此不会拖慢 NPU 推理/渲染，也不会与短按拍照、长按录音发生资源抢占；
    冷却期间/播报进行中直接跳过，不排队、不累积。"""
    global YOLO_TTS_IN_PROGRESS
    if not YOLO_TTS_ANNOUNCE_ENABLED:
        # 双击短按切换的播报总开关：关闭时只跳过语音播报，不影响本次调用之前
        # 已经完成的 YOLOv8 识别/锁定框发布/画面渲染。
        return
    if label_en not in YOLO_TTS_ANNOUNCE_WHITELIST:
        # 不在播报白名单内的类别：画面检测框/标签照常显示，仅跳过语音播报。
        return
    label_cn = YOLO_CLASS_CN.get(label_en)
    if not label_cn:
        return

    now = time.monotonic()
    with YOLO_TTS_STATE_LOCK:
        if YOLO_TTS_IN_PROGRESS or now < YOLO_TTS_NEXT_ALLOWED_AT:
            return
        # 让路给优先级更高的标定语音引导 / VLM 拍照播报：本轮跳过且不推进冷却，
        # 只要目标仍被稳定锁定，下一帧会再次尝试，不影响它们的触发时机。
        if VOICE_BUSY or VLM_IS_RUNNING:
            return
        YOLO_TTS_IN_PROGRESS = True

    threading.Thread(target=_yolo_tts_worker, args=(label_cn,), daemon=True).start()


# 豆包API
def process_vlm_task(frame_to_analyze):
    """后台线程：编码图像 -> 请求豆包 -> 串口播报"""
    global VLM_LATEST_REPLY
    global VLM_IS_RUNNING
    try:
        print("[VLM] 正在分析用户凝视区域...")
        # 1. 直接在内存中将 OpenCV 图像转为 Base64
        _, buffer = cv2.imencode('.jpg', frame_to_analyze)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # 2. 请求豆包 API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOUBAO_API_KEY}"
        }
        payload = {
            "model": DOUBAO_ENDPOINT,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用一段话简短描述这个画面中最主要的物体是什么？有什么内容？必须用中文回答"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        }

        response = requests.post(
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            headers=headers,
            json=payload,
            timeout=(3.05, 20),
        )
        response.raise_for_status()
        reply_text = response.json()['choices'][0]['message']['content']
        print(f"[VLM] 豆包回复: {reply_text}")

        # --->将拿到的结果存入全局变量 <---
        VLM_LATEST_REPLY = reply_text

        # ---------------------------------------------
        # 3. 串口播报: 统一走 serial_tts (SERIAL_LOCK 与标定语音引导互斥, 不抢串口)
        # ---------------------------------------------
        print("🔊 [SYS] Playing VLM reply via H610 TTS...")
        if serial_tts(reply_text):
            print("✅ [SYS] TTS Playback finished.")
        else:
            print("❌ [SYS] TTS Playback failed; L610 network recovery will continue in the background.")

    except Exception as e:
        print(f"❌ [VLM 任务全局失败]: {e}")
    finally:
        # 网络失败、JSON 异常或串口异常都必须释放任务状态。
        with VLM_STATE_LOCK:
            VLM_IS_RUNNING = False


# ==========================================
# Flask 路由配置
# ==========================================
def generate_frames():
    """Yield each rendered frame once and close stale streams promptly.

    Repeating the same JPEG forever lets a broken browser/MJPEG connection
    look healthy while it is actually showing the last frame from before a
    WiFi interruption.  A frame id makes the stream edge-triggered and the
    idle timeout gives the browser's reconnect logic a reliable signal.
    """
    last_frame_id = 0
    idle_since = time.monotonic()

    while True:
        with frame_lock:
            frame_id = RENDERED_FRAME_ID
            rendered_at = RENDERED_FRAME_UPDATED_AT
            latest_jpeg = RENDERED_JPEG

        now = time.monotonic()
        is_new_frame = latest_jpeg and frame_id != last_frame_id
        is_fresh_frame = rendered_at > 0.0 and (now - rendered_at) <= VIDEO_STREAM_IDLE_TIMEOUT_SECONDS
        if is_new_frame and is_fresh_frame:
            last_frame_id = frame_id
            idle_since = now
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Cache-Control: no-cache\r\n\r\n' + latest_jpeg + b'\r\n')
            continue

        if (now - idle_since) >= VIDEO_STREAM_IDLE_TIMEOUT_SECONDS:
            return
        time.sleep(VIDEO_STREAM_POLL_SECONDS)


# 新增：独立分发眼部摄像头的生成器
def generate_eye_frames():
    while True:
        if EYE_FRAME is not None:
            ret, buffer = cv2.imencode('.jpg', EYE_FRAME, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


# 新增：独立服务眼球画面的路由
@app.route('/eye_feed')
def eye_feed():
    return Response(generate_eye_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/click', methods=['POST'])
def handle_click():
    data = request.get_json()
    click_x = data.get('x', 0)
    click_y = data.get('y', 0)

    hit_label = "None"
    pseudo_depth = "Unknown (Requires Stereo/ToF or C++ Depth Map)"

    for i, box in enumerate(GLOBAL_BOXES):
        x1, y1, x2, y2 = box
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            hit_label = CLASSES[GLOBAL_CLASSES[i]]
            box_width = x2 - x1
            if box_width > 0:
                pseudo_depth = f"Estimated proxy depth: {10000 / box_width:.2f} (Based on BBox width)"
            break

    print("\n" + "=" * 50)
    print("🎯 [MOUSE CALIBRATION & DEBUG INFO] 🎯")
    print(f"📍 Actual Mouse Click  : X = {click_x}, Y = {click_y}")
    print(f"👁️ Implemented Gaze    : X = {VIRTUAL_X}, Y = {VIRTUAL_Y}")
    dx = click_x - VIRTUAL_X
    dy = click_y - VIRTUAL_Y
    print(f"📐 Calibration Offset  : dX = {dx}, dY = {dy}")
    print(f"📦 Target Object Hit   : {hit_label}")
    print(f"📏 Depth Information   : {pseudo_depth}")
    print(f"🟢 Eye Tracking Status : {'Calibrated (Active)' if IS_CALIBRATED else 'Uncalibrated (Locked)'}")
    if EYE_FRAME is not None:
        print(f"👀 Eye Frame Stream    : Active, Frame Shape={EYE_FRAME.shape}")
    else:
        print(f"👀 Eye Frame Stream    : Offline")
    print("=" * 50 + "\n")

    return jsonify({"status": "success", "dx": dx, "dy": dy})


@app.route('/status')
def status():
    global IS_CALIBRATED, VIRTUAL_X, VIRTUAL_Y, SNAPPED_MODE, GLOBAL_BOXES
    global DEBUG_PHASE, DEBUG_MSG, DEBUG_PTS, DEBUG_Z
    global DEBUG_GY, DEBUG_EYEDY
    global DEBUG_TOF_VALID, DEBUG_TOF_STD
    global VLM_LATEST_REPLY  # 引入文本变量
    global VLM_IS_RUNNING  # <--- 新增：引入 AI 运行状态锁
    global ASR_LATEST_TEXT, LLM_LATEST_REPLY # <-- 引入新增的全局变量
    now = time.monotonic()
    with WORKER_STATE_LOCK:
        worker_health = {
            name: {
                "state": info.get("state", "unknown"),
                "restarts": info.get("restarts", 0),
                "last_packet_age": (
                    round(now - info["last_packet"], 2)
                    if info.get("last_packet") is not None else None
                ),
                "last_error": info.get("last_error", ""),
            }
            for name, info in WORKER_STATE.items()
        }
    return jsonify({
        "calibrated": IS_CALIBRATED,
        "gaze_x": int(VIRTUAL_X),
        "gaze_y": int(VIRTUAL_Y),
        "snapped": SNAPPED_MODE,
        "objects_count": len(GLOBAL_BOXES),
        "debug_phase": DEBUG_PHASE,
        "debug_msg": DEBUG_MSG,
        "debug_pts": DEBUG_PTS,
        "debug_z": DEBUG_Z,
        "debug_gy": DEBUG_GY,           # 垂直诊断: 注视向量 gy
        "debug_eye_dy": DEBUG_EYEDY,    # 垂直诊断: 眼图瞳孔-模型垂直位移
        "debug_tof_valid": DEBUG_TOF_VALID,  # ToF 质量: 有效格数(0-9)
        "debug_tof_std": DEBUG_TOF_STD,      # ToF 质量: 深度标准差(mm)
        "vlm_reply": VLM_LATEST_REPLY,  # 将文本塞入 JSON 发给网页
        "vlm_running": VLM_IS_RUNNING,  # <--- 新增：将状态传给前端网页
        "asr_text": ASR_LATEST_TEXT,     # <-- 传给前端
        "llm_reply": LLM_LATEST_REPLY,   # <-- 传给前端
        "workers": worker_health
    })


@app.route('/ask_doubao', methods=['POST'])
def trigger_doubao():
    started, reason, _ = submit_vlm_task("web")
    if started:
        return jsonify({"status": "VLM task started"})
    return jsonify({"status": "error", "message": reason}), 409


@app.route('/trigger_audio', methods=['POST'])
def trigger_audio():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"status": "error", "message": "No filename provided"}), 400

    try:
        # 向本地的 5007 端口发送文件名
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(filename.encode('utf-8'), ("127.0.0.1", 5007))
        sock.close()
        print(f"\n🚀 [IPC] 已向 ASR 服务发送音频触发指令: {filename}")
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"❌ [IPC Error] UDP 发送失败: {e}")
        return jsonify({"status": "error"}), 500


# ==========================================
# 前端主页面
# ==========================================
@app.route('/')
def index():
    html = r"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>RV1126B 智能眼镜调试上位机</title>
        <style>
            body {
                background-color: #0b101a;
                color: #00ffcc;
                font-family: 'Courier New', Courier, monospace;
                margin: 0;
                padding: 10px 20px;
                box-sizing: border-box;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow: hidden;
            }

            .top-hud {
                display: flex;
                justify-content: space-around;
                align-items: center;
                background: rgba(0, 30, 20, 0.6);
                border: 1px solid rgba(0, 255, 204, 0.4);
                border-top: 3px solid #00ffcc;
                padding: 10px 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
                z-index: 10;
            }

            .top-hud .hud-item {
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 2px;
            }

            .hud-item span:first-child { color: rgba(0, 255, 204, 0.7); margin-right: 10px; }

            /* 独立出来的眼球图像，固定在左上角下方区域 */
            .eye-pip-inline {
                width: 100%;
                height: auto;
                background-color: #000;
                margin-bottom: 20px;
                box-sizing: border-box;

                /* Unified Border & Shadow Style */
                border: 1px solid rgba(0, 255, 204, 0.4);
                border-left: 5px solid #00ffcc;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            }

            .main-layout {
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: stretch; /* 让子元素填满高度 */
                gap: 20px;
                flex: 1;
                margin: 15px 0;
                min-height: 0;
            }

            /* 调整 Wrapper 使视频内容尽可能大 */
            .video-wrapper {
                position: relative;
                flex: 1; /* 占据尽可能多的空间 */
                display: flex;
                justify-content: center;
                align-items: center;
                border: 2px solid rgba(0, 255, 204, 0.5);
                box-shadow: 0 0 20px rgba(0, 255, 204, 0.15);
                background-color: #000;
                overflow: hidden;
            }

            .video-feed {
                width: 100%;
                height: 100%;
                object-fit: contain; /* 保证原比例前提下尽可能缩放撑满容器 */
                cursor: crosshair;
            }

            .data-column {
                display: flex;
                flex-direction: column;
                justify-content: center;
                width: 320px;
            }

           <img id="eye_stream" class="eye-pip" src="/eye_feed" alt="Eye Camera Feed">

            /* Ensure your data-panel matches exactly */
            .data-panel {
                background: rgba(0, 30, 20, 0.6);
                padding: 20px;

                /* Unified Border & Shadow Style */
                border: 1px solid rgba(0, 255, 204, 0.4);
                border-left: 5px solid #00ffcc;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            }

            .data-panel h3 {
                margin-top: 0;
                border-bottom: 1px dashed rgba(0, 255, 204, 0.3);
                padding-bottom: 10px;
                font-size: 17px;
                letter-spacing: 1px;
            }

            .data-panel p {
                margin: 10px 0;
                font-size: 14px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
            }

            /* 减小底部终端的尺寸 */
            .bottom-terminal {
                background: rgba(0, 30, 20, 0.6);
                border: 1px solid rgba(0, 255, 204, 0.4);
                border-left: 5px solid #00ffcc;
                padding: 10px 20px;
                height: 85px; /* 进一步压低固定高度 */
                display: flex;
                flex-direction: column;
                box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.5);
            }

            .terminal-header {
                font-size: 14px;
                font-weight: bold;
                border-bottom: 1px dashed rgba(0, 255, 204, 0.3);
                padding-bottom: 5px;
                margin-bottom: 5px;
            }

            .terminal-content {
                flex: 1;
                overflow-y: auto;
                font-size: 13px;
                line-height: 1.4;
            }

            .value-highlight, .value { color: #fff; text-shadow: 0 0 5px #00ffcc; }
            .status-red { color: #ff4444; text-shadow: 0 0 5px #ff4444; }

            .click-ripple {
                position: absolute;
                border: 2px solid #00ffcc;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                animation: ripple 0.5s ease-out;
                pointer-events: none;
            }
            @keyframes ripple {
                from { width: 0; height: 0; opacity: 1; }
                to { width: 100px; height: 100px; opacity: 0; }
            }
        </style>
    </head>
    <body>

        <div class="top-hud">
            <div class="hud-item"><span>PHASE:</span> <span id="debug-phase" class="value" style="color:#00ffcc;">UNKNOWN</span></div>
            <div class="hud-item"><span>ACTION:</span> <span id="sys-msg" class="value" style="color:#ffcc00; text-shadow: 0 0 5px #ffcc00;">WAITING...</span></div>
            <div class="hud-item"><span>CALIB PTS:</span> <span id="debug-pts" class="value">0/32</span></div>
            <div class="hud-item"><span>DEPTH Z:</span> <span id="debug-z" class="value">0.0 mm</span></div>
            <div class="hud-item"><span>GY / eyeDY:</span> <span id="debug-vert" class="value" style="color:#ff9944; text-shadow:0 0 5px #ff9944;">0.00 / 0</span></div>
            <div class="hud-item"><span>ToF:</span> <span id="debug-tof" class="value" style="color:#44ddff; text-shadow:0 0 5px #44ddff;">-/9 &sigma;0</span></div>
        </div>

        <div class="main-layout">
            <div class="video-wrapper" id="video_wrapper">
                <img id="video_stream" class="video-feed" src="/video_feed">
            </div>

            <div class="data-column">
                <img id="eye_stream" class="eye-pip-inline" src="/eye_feed" alt="Eye Camera Feed">

                <div class="data-panel">
                    <h3>[ 综合感知系统 ]</h3>
                    <p><span>拍照识别触发</span> <span id="vlm-trigger-status" class="value-highlight" style="color:#00ffcc;">🟢 在线</span></p>
                    <p><span>绝对坐标 X</span> <span id="gaze-x" class="value-highlight">0</span></p>
                    <p><span>绝对坐标 Y</span> <span id="gaze-y" class="value-highlight">0</span></p>
                    <p><span>云端 AI 模型</span> <span id="cloud-ai-status" class="value-highlight">🟢 在线</span></p>
                    <hr style="border: 0; border-top: 1px dashed rgba(0, 255, 204, 0.3); margin: 15px 0;">
                    <p><span>本地推理框架</span> <span class="value-highlight">YOLOv8 .rknn</span></p>
                    <p><span>实时检测数</span> <span id="obj-count" class="value-highlight">0</span></p>
                    <p><span>UI 更新延迟</span> <span id="latency" class="value-highlight">-- ms</span></p>
                    <p><span>AI对话</span> <span id="ai-dialog-status" class="value-highlight" style="color:#00ffcc;">🟢 在线</span></p>
                </div>
            </div>
        </div>

        <div class="bottom-terminal">
            <div class="terminal-header">SYSTEM TERMINAL</div>
            <div class="terminal-content" id="terminal-out">
                > SYSTEM BOOT SEQ INITIATED...<br>
                > AWAITING SENSOR DATA...
            </div>
        </div>

        <script>
            const stream = document.getElementById('video_stream');
            const wrapper = document.getElementById('video_wrapper');

            stream.addEventListener('click', function(e) {
                var rect = this.getBoundingClientRect();

                // 获取当前图片实际渲染出的宽高
                var renderedWidth = this.clientWidth;
                var renderedHeight = this.clientHeight;

                // 由于使用 object-fit: contain, 需要计算实际图片的黑边偏移量
                var videoRatio = 1280 / 720;
                var containerRatio = renderedWidth / renderedHeight;
                var offsetX = 0;
                var offsetY = 0;
                var drawWidth = renderedWidth;
                var drawHeight = renderedHeight;

                if (containerRatio > videoRatio) {
                    drawWidth = renderedHeight * videoRatio;
                    offsetX = (renderedWidth - drawWidth) / 2;
                } else {
                    drawHeight = renderedWidth / videoRatio;
                    offsetY = (renderedHeight - drawHeight) / 2;
                }

                var scaleX = 1280 / drawWidth;
                var scaleY = 720 / drawHeight;

                var clickX = e.clientX - rect.left - offsetX;
                var clickY = e.clientY - rect.top - offsetY;

                if (clickX < 0 || clickY < 0 || clickX > drawWidth || clickY > drawHeight) {
                    return; // 点击在黑边外
                }

                var x = Math.round(clickX * scaleX);
                var y = Math.round(clickY * scaleY);

                var ripple = document.createElement('div');
                ripple.className = 'click-ripple';
                ripple.style.left = (e.clientX - rect.left) + 'px';
                ripple.style.top = (e.clientY - rect.top) + 'px';
                wrapper.appendChild(ripple);
                setTimeout(() => ripple.remove(), 500);

                fetch('/click', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x: x, y: y})
                }).catch(err => console.error('发送失败:', err));
            });

            window.lastMsgBase = "";
            window.lastVlmReply = ""; // 记录上一条豆包回复
            window.lastAsrText = "";
            window.lastLlmReply = "";
            window.backendWasOffline = false;

            const streamRetryTimers = {};
            const streamRecoveryTimers = [];
            const STREAM_RECOVERY_DELAYS_MS = [0, 600, 1800, 4000];
            const streamPaths = {
                video_stream: '/video_feed',
                eye_stream: '/eye_feed'
            };

            function reloadStream(id) {
                const img = document.getElementById(id);
                if (!img) return;
                if (streamRetryTimers[id]) {
                    clearTimeout(streamRetryTimers[id]);
                    streamRetryTimers[id] = null;
                }
                img.src = streamPaths[id] + '?reconnect=' + Date.now();
            }

            function reloadAllStreams() {
                Object.keys(streamPaths).forEach(reloadStream);
            }

            // A WiFi reconnect can complete before the browser fires a useful
            // <img> error/load event.  Retry after the link comes back at a
            // few increasing delays so the request is not made only once
            // while the route is still being restored.
            function scheduleStreamRecovery() {
                while (streamRecoveryTimers.length) {
                    clearTimeout(streamRecoveryTimers.pop());
                }
                STREAM_RECOVERY_DELAYS_MS.forEach(delay => {
                    const timer = setTimeout(() => reloadAllStreams(), delay);
                    streamRecoveryTimers.push(timer);
                });
            }

            Object.keys(streamPaths).forEach(id => {
                const img = document.getElementById(id);
                if (!img) return;
                img.addEventListener('error', () => {
                    if (!streamRetryTimers[id]) {
                        streamRetryTimers[id] = setTimeout(() => reloadStream(id), 1000);
                    }
                });
                img.addEventListener('load', () => {
                    if (streamRetryTimers[id]) {
                        clearTimeout(streamRetryTimers[id]);
                        streamRetryTimers[id] = null;
                    }
                });
            });
            window.addEventListener('offline', () => {
                window.backendWasOffline = true;
            });
            window.addEventListener('online', scheduleStreamRecovery);

            function fetchStatus() {
                let reqStart = Date.now();
                const controller = new AbortController();
                const requestTimeout = setTimeout(() => controller.abort(), 2000);
                fetch('/status', {signal: controller.signal, cache: 'no-store'})
                    .then(response => {
                        if (!response.ok) throw new Error('HTTP ' + response.status);
                        if (window.backendWasOffline) scheduleStreamRecovery();
                        window.backendWasOffline = false;
                        return response.json();
                    })
                    .then(data => {
                        document.getElementById('gaze-x').innerText = data.gaze_x;
                        document.getElementById('gaze-y').innerText = data.gaze_y;
                        // 根据 AI 是否在运行，动态切换文字和颜色
                        let cloudAiEl = document.getElementById('cloud-ai-status');
                        cloudAiEl.innerText = data.vlm_running ? "⏳ 识别中..." : "🟢 在线";
                        cloudAiEl.style.color = data.vlm_running ? "#ffcc00" : "#00ffcc"; // 识别中变黄色，在线为青色
                        document.getElementById('obj-count').innerText = data.objects_count;
                        document.getElementById('latency').innerText = (Date.now() - reqStart) + " ms";

                        document.getElementById('debug-phase').innerText = data.debug_phase;
                        document.getElementById('debug-pts').innerText = data.debug_pts + "/32";
                        document.getElementById('debug-z').innerText = data.debug_z.toFixed(1) + " mm";

                        // 垂直通道诊断: gy 与 眼图垂直位移 eyeDY 同步显示
                        let vertEl = document.getElementById('debug-vert');
                        if (vertEl) {
                            let gyv = (data.debug_gy !== undefined) ? data.debug_gy.toFixed(2) : "0.00";
                            let edy = Math.round(data.debug_eye_dy || 0);
                            vertEl.innerText = gyv + " / " + edy;
                        }

                        // ToF 质量诊断: 有效格数<3 或 std>200mm 变红, 提示该处深度不可信
                        let tofEl = document.getElementById('debug-tof');
                        if (tofEl) {
                            let tv = (data.debug_tof_valid !== undefined) ? data.debug_tof_valid : 0;
                            let ts = Math.round(data.debug_tof_std || 0);
                            tofEl.innerText = tv + "/9 σ" + ts;
                            tofEl.style.color = (tv < 3 || ts > 200) ? "#ff3333" : "#44ddff";
                        }

                        let msgText = data.debug_msg || "NORMAL";
                        let sysMsgEl = document.getElementById('sys-msg');
                        sysMsgEl.innerText = msgText;

                        let msgUpper = msgText.toUpperCase();
                        let colorHex = "#ffcc00";

                        if (msgUpper.includes("UNSTABLE") || msgUpper.includes("OUT OF") || msgUpper.includes("TOO CLOSE") || msgUpper.includes("WARN")) {
                            colorHex = "#ff3333";
                        } else if (msgUpper.includes("SUCCESS") || msgUpper.includes("TRACKING") || msgUpper.includes("TRAINING")) {
                            colorHex = "#00ffcc";
                        }
                        sysMsgEl.style.color = colorHex;
                        sysMsgEl.style.textShadow = `0 0 5px ${colorHex}`;

                        if (msgText !== "NORMAL") {
                            let baseMsg = msgText.replace(/[0-9.\- ]/g, '');
                            if (window.lastMsgBase !== baseMsg) {
                                let term = document.getElementById('terminal-out');
                                let timeStr = new Date().toLocaleTimeString();
                                term.innerHTML += `<br><span style='color:#555;'>[${timeStr}]</span> > <span style='color:#00ffcc;'>[${data.debug_phase}]</span> <span style='color:${colorHex};'>${msgText}</span>`;
                                term.scrollTop = term.scrollHeight;
                                window.lastMsgBase = baseMsg;
                            }
                        }

                        // --->处理并显示豆包 VLM 回复 <---
                        if (data.vlm_reply && data.vlm_reply !== window.lastVlmReply) {
                            let term = document.getElementById('terminal-out');
                            let timeStr = new Date().toLocaleTimeString();
                            // 用醒目的粉紫色 (#ff66ff) 显示 AI 视觉回复
                            term.innerHTML += `<br><span style='color:#555;'>[${timeStr}]</span> > <span style='color:#ff66ff; font-weight:bold;'>[DOUBAO VLM] ${data.vlm_reply}</span>`;
                            term.scrollTop = term.scrollHeight;
                            window.lastVlmReply = data.vlm_reply; // 更新记录
                        }

                        // ---> 处理并显示 Zipformer STT 语音识别结果 <---
                        if (data.asr_text && data.asr_text !== window.lastAsrText) {
                            let term = document.getElementById('terminal-out');
                            let timeStr = new Date().toLocaleTimeString();
                            // 使用亮橙色显示使用者的语音内容
                            term.innerHTML += `<br><span style='color:#555;'>[${timeStr}]</span> > <span style='color:#ffaa00; font-weight:bold;'>[ZIPFORMER ASR] 🗣️: "${data.asr_text}"</span>`;
                            term.scrollTop = term.scrollHeight;
                            window.lastAsrText = data.asr_text;
                        }

                        // ---> 处理并显示 Doubao LLM 语音对话回复 <---
                        if (data.llm_reply && data.llm_reply !== window.lastLlmReply) {
                            let term = document.getElementById('terminal-out');
                            let timeStr = new Date().toLocaleTimeString();
                            // 使用天蓝色显示豆包的回复内容
                            term.innerHTML += `<br><span style='color:#555;'>[${timeStr}]</span> > <span style='color:#00aaff; font-weight:bold;'>[DOUBAO LLM] 🤖: ${data.llm_reply}</span>`;
                            term.scrollTop = term.scrollHeight;
                            window.lastLlmReply = data.llm_reply;

                            // 收到答复后，恢复 AI 对话的状态为在线
                            let aiDialogSpan = document.getElementById('ai-dialog-status');
                            if (aiDialogSpan) {
                                aiDialogSpan.innerText = "🟢 在线";
                                aiDialogSpan.style.color = "#00ffcc";
                            }
                        }
                    })
                    .catch(err => {
                        window.backendWasOffline = true;
                    })
                    .finally(() => {
                        clearTimeout(requestTimeout);
                        setTimeout(fetchStatus, window.backendWasOffline ? 1000 : 200);
                    });
            }

            function askDoubao() {
                fetch('/ask_doubao', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    let term = document.getElementById('terminal-out');
                    term.innerHTML += `<br><span style='color:#00ffcc;'>[SYSTEM] Requesting Cloud VLM for targeted region...</span>`;
                    term.scrollTop = term.scrollHeight;
                });
            }

            // ==========================================
            // 监听全局按键：空格触发视觉，数字键触发语音
            // ==========================================
            document.addEventListener('keydown', function(event) {
                // 空格键按下触发视觉识别 (等价于短按按键)
                if (event.code === 'Space' || event.key === ' ') {
                    if (event.repeat) return; // 防止长按导致频繁请求
                    event.preventDefault();

                    let vlmSpan = document.getElementById('vlm-trigger-status');
                    if (vlmSpan) {
                        vlmSpan.innerText = "🔴 触发";
                        vlmSpan.style.color = "#ff4444";
                    }
                    askDoubao();
                }
                // 监听数字键 1, 2, 3
                else if (['1', '2', '3'].includes(event.key)) {
                    event.preventDefault();

                    // 按下数字键，修改 AI 对话状态
                    let aiDialogSpan = document.getElementById('ai-dialog-status');
                    if (aiDialogSpan) {
                        aiDialogSpan.innerText = "⏳ 正在思考";
                        aiDialogSpan.style.color = "#ffcc00";
                    }

                    let targetFile = event.key + ".wav";

                    let term = document.getElementById('terminal-out');
                    term.innerHTML += `<br><span style='color:#00ffcc;'>[SYSTEM] 正在识别音频输入......</span>`;
                    term.scrollTop = term.scrollHeight;

                    // 发送给后端的 trigger_audio 路由
                    fetch('/trigger_audio', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ filename: targetFile })
                    }).catch(err => console.error('Audio trigger failed:', err));
                }
            });

            // 监听按键松开，重置状态
            document.addEventListener('keyup', function(event) {
                if (event.code === 'Space' || event.key === ' ') {
                    let vlmSpan = document.getElementById('vlm-trigger-status');
                    if (vlmSpan) {
                        vlmSpan.innerText = "🟢 在线";
                        vlmSpan.style.color = "#00ffcc";
                    }
                }
            });

            fetchStatus();
        </script>
    </body>
    </html>
    """
    return html


# ==========================================
# main 函数
# ==========================================
if __name__ == '__main__':
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0 or rknn.init_runtime() != 0:
        print("RKNN Init Failed");
        exit()

    _start_resilient_worker("coord", udp_coord_receiver)
    _start_resilient_worker("debug", udp_debug_receiver)
    _start_resilient_worker("world", video_processor_thread)
    _start_resilient_worker("npu", npu_inference_thread)
    _start_resilient_worker("eye", udp_eye_receiver)
    _start_resilient_worker("world_clean", udp_world_clean_receiver)
    _start_resilient_worker("button", udp_button_receiver)
    _start_resilient_worker("nlp", udp_nlp_receiver)
    _start_resilient_worker("voice", udp_voice_receiver)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
