# -*- coding: utf-8 -*-
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

import sys
import logging
import requests
import serial
import time
import socket
import json
import subprocess
import gpiod
import argparse
import torch
import soundfile as sf
import numpy as np
import scipy
import threading
import queue
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

# 资源路径固定相对于脚本自身，而不是相对于启动命令的当前目录。
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 全局配置区
# ==========================================
# 1. GPIO 按键配置（按键接3.3V，高电平有效）
GPIO_CHIP = "gpiochip5"
GPIO_LINE = 1
ACTIVE_LOW = False
DEBOUNCE_MS = 20
MIN_RECORD_MS = 300

# 2. 录音配置
AUDIO_DEVICE = "plughw:0,0"  # You can also try commenting this out if it still fails
SAMPLE_RATE = 44100  # Changed from 16000 to match hardware
CHANNELS = 2  # Changed from 1 to Stereo to match hardware

# 3. 豆包 & TTS 配置
API_KEY = os.environ.get("DOUBAO_CHAT_API_KEY", "")
ENDPOINT_ID = os.environ.get("DOUBAO_CHAT_ENDPOINT", "bot-20260513165540-xqfvh")
AT_PORT = os.environ.get("RV1126_TTS_PORT", "/dev/ttyUSB5")
BAUD_RATE = 115200
TTS_VOLUME = 5
TTS_LOCK_PATH = "/tmp/rv1126_tts.lock"
TTS_MAX_GBK_BYTES_PER_CHUNK = 100
TTS_COMMAND_TIMEOUT_SECONDS = 3.0
TTS_PLAYBACK_TIMEOUT_SECONDS = 30.0
TTS_SERIAL_READ_SLICE_SECONDS = 0.2
CLOUD_CONNECT_TIMEOUT_SECONDS = 3.05
CLOUD_READ_TIMEOUT_SECONDS = 15

# 4. UI 推送配置
DASHBOARD_UDP = ("127.0.0.1", 5008)

# 5. 短按拍照触发：本进程独占 GPIO(gpiochip5 line 1)。短按时先请求 C++
#    复用标定；只有 C++ 没有接受复用时，才通知 YOLO8_test.py 拍照。
PHOTO_TRIGGER_UDP = ("127.0.0.1", 5010)

# 6. 复用标定握手：C++ 仅在阶段A真正接受复用时返回 REUSE_ACCEPTED。
#    这让一次短按只能二选一：复用标定，或正常拍照，不再同时广播两个动作。
CPP_REUSE_CALIB_UDP = ("127.0.0.1", 5011)
REUSE_ACK_TIMEOUT_SECONDS = 0.2
REUSE_PHOTO_SUPPRESS_SECONDS = 1.0

# ==========================================
# 日志兼容补丁
# ==========================================
_original_checkLevel = logging._checkLevel


def _safe_checkLevel(level):
    try:
        return _original_checkLevel(level)
    except ValueError:
        return 30


logging._checkLevel = _safe_checkLevel
os.environ['TORCH_FX_MATCH_LEVEL'] = '30'
os.environ['PYTHONWARNINGS'] = 'ignore'

# 可选导入
try:
    import onnxruntime
except ImportError:
    pass
try:
    from rknn.api import RKNN
except ImportError:
    from rknnlite.api import RKNNLite as RKNN
try:
    import kaldifeat
except ImportError:
    print("错误：缺少 kaldifeat 依赖，请先安装")
    sys.exit(1)


# ==========================================
# 通用工具函数
# ==========================================
def send_to_dashboard(msg_type, text):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = json.dumps({"type": msg_type, "text": text}).encode('utf-8')
        sock.sendto(payload, DASHBOARD_UDP)
        sock.close()
    except Exception:
        pass


def send_photo_trigger():
    """短按按键 -> 通知 YOLO8_test.py 拍照上传豆包 (本进程独占 GPIO, 仅转发信号)"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(b"SNAP", PHOTO_TRIGGER_UDP)
        sock.close()
        print(f"   ↳ [IPC] 已发送 SNAP -> {PHOTO_TRIGGER_UDP}")
    except Exception as e:
        print(f"   ↳ [IPC] SNAP 发送失败: {e}")


def request_reuse_calibration():
    """请求复用标定；只有 C++ 在 Phase A 接受请求时才返回 True。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.settimeout(REUSE_ACK_TIMEOUT_SECONDS)
            sock.sendto(b"REUSE_CALIB", CPP_REUSE_CALIB_UDP)
            response, _ = sock.recvfrom(64)
        accepted = response.decode("utf-8", errors="ignore").strip() == "REUSE_ACCEPTED"
        if accepted:
            print("   ↳ [IPC] C++ 已确认复用标定，本次不发送 SNAP")
        return accepted
    except socket.timeout:
        # 正常识别阶段没有进程监听 5011，超时代表本次短按应按拍照处理。
        return False
    except Exception as e:
        print(f"   ↳ [IPC] 复用标定握手失败，按正常拍照处理: {e}")
        return False


@contextmanager
def tts_process_lock():
    """跨进程串口锁，避免本进程与 YOLO 同时打开 /dev/ttyUSB5。"""
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
# GPIO 按键 + 录音函数
# ==========================================
def init_gpio():
    chip = gpiod.Chip(GPIO_CHIP)
    line = chip.get_line(GPIO_LINE)
    flags = gpiod.LINE_REQ_EV_BOTH_EDGES
    if ACTIVE_LOW:
        flags |= gpiod.LINE_REQ_FLAG_ACTIVE_LOW
    try:
        bias = gpiod.LINE_REQ_BIAS_PULL_UP if ACTIVE_LOW else gpiod.LINE_REQ_BIAS_PULL_DOWN
        line.request(consumer="voice_assistant", type=flags, bias=bias)
    except Exception:
        line.request(consumer="voice_assistant", type=flags)
    return line


def start_record(file_path):
    cmd = [
        "arecord",
        "-c", str(CHANNELS),
        "-r", str(SAMPLE_RATE),
        "-f", "S16_LE",
        "-t", "wav",
        "-q",
        file_path
    ]
    # Removed stderr=subprocess.DEVNULL to help catch future hardware errors
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL)


def stop_record(process):
    if process and process.poll() is None:
        # Send Ctrl+C (SIGINT) so arecord safely finalizes the WAV header
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


# ==========================================
# Zipformer ASR 核心逻辑（原封不动保留）
# ==========================================
class BaseModel():
    def __init__(self):
        self.model_config = {'x': [1, 103, 80], 'cached_len_0': [2, 1], 'cached_len_1': [2, 1], 'cached_len_2': [2, 1],
                             'cached_len_3': [2, 1],
                             'cached_len_4': [2, 1], 'cached_avg_0': [2, 1, 256], 'cached_avg_1': [2, 1, 256],
                             'cached_avg_2': [2, 1, 256],
                             'cached_avg_3': [2, 1, 256], 'cached_avg_4': [2, 1, 256], 'cached_key_0': [2, 192, 1, 192],
                             'cached_key_1': [2, 96, 1, 192],
                             'cached_key_2': [2, 48, 1, 192], 'cached_key_3': [2, 24, 1, 192],
                             'cached_key_4': [2, 96, 1, 192], 'cached_val_0': [2, 192, 1, 96],
                             'cached_val_1': [2, 96, 1, 96], 'cached_val_2': [2, 48, 1, 96],
                             'cached_val_3': [2, 24, 1, 96], 'cached_val_4': [2, 96, 1, 96],
                             'cached_val2_0': [2, 192, 1, 96], 'cached_val2_1': [2, 96, 1, 96],
                             'cached_val2_2': [2, 48, 1, 96], 'cached_val2_3': [2, 24, 1, 96],
                             'cached_val2_4': [2, 96, 1, 96], 'cached_conv1_0': [2, 1, 256, 30],
                             'cached_conv1_1': [2, 1, 256, 30], 'cached_conv1_2': [2, 1, 256, 30],
                             'cached_conv1_3': [2, 1, 256, 30], 'cached_conv1_4': [2, 1, 256, 30],
                             'cached_conv2_0': [2, 1, 256, 30], 'cached_conv2_1': [2, 1, 256, 30],
                             'cached_conv2_2': [2, 1, 256, 30], 'cached_conv2_3': [2, 1, 256, 30],
                             'cached_conv2_4': [2, 1, 256, 30]}

    def init_encoder_input(self):
        self.encoder_input = []
        self.encoder_input_dict = {}
        for input_name in self.model_config:
            if 'cached_len' in input_name:
                data = np.zeros((self.model_config[input_name]), dtype=np.int64)
                self.encoder_input.append(data)
                self.encoder_input_dict[input_name] = data
            else:
                data = np.zeros((self.model_config[input_name]), dtype=np.float32)
                self.encoder_input.append(data)
                self.encoder_input_dict[input_name] = data

    def update_encoder_input(self, out, model_type):
        for idx, input_name in enumerate(self.encoder_input_dict):
            if idx == 0:
                continue
            if idx > 10 and model_type == 'rknn':
                data = self.convert_nchw_to_nhwc(out[idx])
            else:
                data = out[idx]
            self.encoder_input[idx] = data
            self.encoder_input_dict[input_name] = data

    def convert_nchw_to_nhwc(self, src):
        dst = np.transpose(src, (0, 2, 3, 1))
        return dst

    def init_model(self, model_path, target, device_id):
        pass

    def release_model(self):
        pass

    def run_encoder(self, x):
        pass

    def run_decoder(self, decoder_input):
        pass

    def run_joiner(self, encoder_out, decoder_out):
        pass

    def run_greedy_search(self, frames, context_size, decoder_out, hyp, num_processed_frames, timestamp, frame_offset):
        encoder_out = self.run_encoder(frames)
        encoder_out = encoder_out.squeeze(0)

        blank_id = 0
        unk_id = 2
        if decoder_out is None and hyp is None:
            hyp = [blank_id] * context_size
            decoder_input = np.array([hyp], dtype=np.int64)
            decoder_out = self.run_decoder(decoder_input)

        T = encoder_out.shape[0]
        for t in range(T):
            cur_encoder_out = encoder_out[t: t + 1]
            joiner_out = self.run_joiner(cur_encoder_out, decoder_out).squeeze(0)
            y = np.argmax(joiner_out, axis=0)
            if y != blank_id and y != unk_id:
                timestamp.append(frame_offset + t)
                hyp.append(y)
                decoder_input = hyp[-context_size:]
                decoder_input = np.array([decoder_input], dtype=np.int64)
                decoder_out = self.run_decoder(decoder_input)
        frame_offset += T

        return hyp, decoder_out, timestamp, frame_offset


class OnnxModel(BaseModel):
    def __init__(self, encoder_model_path, decoder_model_path, joiner_model_path, target, device_id):
        super().__init__()
        self.encoder = self.init_model(encoder_model_path, target, device_id)
        self.decoder = self.init_model(decoder_model_path, target, device_id)
        self.joiner = self.init_model(joiner_model_path, target, device_id)

    def init_model(self, model_path, target, device_id):
        model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return model

    def release_model(self):
        del self.encoder, self.decoder, self.joiner
        self.encoder = self.decoder = self.joiner = None

    def run_encoder(self, x):
        self.encoder_input[0] = x.numpy()
        self.encoder_input_dict['x'] = x.numpy()
        out = self.encoder.run(None, self.encoder_input_dict)
        self.update_encoder_input(out, 'onnx')
        return out[0]

    def run_decoder(self, decoder_input):
        out = self.decoder.run(None, {self.decoder.get_inputs()[0].name: decoder_input})[0]
        return out

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.run(None, {self.joiner.get_inputs()[0].name: encoder_out,
                                     self.joiner.get_inputs()[1].name: decoder_out})[0]
        return out


class RKNNModel(BaseModel):
    def __init__(self, encoder_model_path, decoder_model_path, joiner_model_path, target, device_id):
        super().__init__()
        self.encoder = self.init_model(encoder_model_path, target, device_id)
        self.decoder = self.init_model(decoder_model_path, target, device_id)
        self.joiner = self.init_model(joiner_model_path, target, device_id)

    def init_model(self, model_path, target, device_id):
        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f'Load RKNN model {model_path} failed!')
            exit(ret)
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        return rknn

    def release_model(self):
        self.encoder.release()
        self.decoder.release()
        self.joiner.release()
        self.encoder = self.decoder = self.joiner = None

    def run_encoder(self, x):
        self.encoder_input[0] = x.numpy()
        self.encoder_input_dict['x'] = x.numpy()
        out = self.encoder.inference(inputs=self.encoder_input)
        self.update_encoder_input(out, 'rknn')
        return out[0]

    def run_decoder(self, decoder_input):
        out = self.decoder.inference(inputs=decoder_input)[0]
        return out

    def run_joiner(self, encoder_out, decoder_out):
        out = self.joiner.inference(inputs=[encoder_out, decoder_out])[0]
        return out


def read_vocab(tokens_file):
    with open(tokens_file, 'r', encoding='utf-8') as f:
        vocab = {}
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                key = parts[0]
                value = ""
            else:
                value, key = parts[0], parts[1]
            vocab[key] = value
    return vocab


def set_model(args):
    if args.encoder_model_path.endswith(".rknn") and args.decoder_model_path.endswith(
            ".rknn") and args.joiner_model_path.endswith(".rknn"):
        model = RKNNModel(args.encoder_model_path, args.decoder_model_path, args.joiner_model_path,
                          target=args.target, device_id=args.device_id)
    elif args.encoder_model_path.endswith(".onnx") and args.decoder_model_path.endswith(
            ".onnx") and args.joiner_model_path.endswith(".onnx"):
        model = OnnxModel(args.encoder_model_path, args.decoder_model_path, args.joiner_model_path,
                          target=args.target, device_id=args.device_id)
    else:
        print("模型格式不统一，请统一使用rknn或onnx")
        sys.exit(1)
    return model


def run_model(model, audio_data, global_sample_rate):
    opts = kaldifeat.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    fbank = kaldifeat.OnlineFbank(opts)

    num_processed_frames = 0
    segment = 103
    offset = 96
    context_size = 2
    hyp = None
    decoder_out = None
    fbank.accept_waveform(sampling_rate=global_sample_rate, waveform=audio_data)
    num_frames = fbank.num_frames_ready
    timestamp = []
    frame_offset = 0

    while num_frames - num_processed_frames > 0:
        if (num_frames - num_processed_frames) < segment:
            tail_padding_len = (segment - (num_frames - num_processed_frames)) / 100.0
            tail_padding = torch.zeros(int(tail_padding_len * global_sample_rate), dtype=torch.float32)
            fbank.accept_waveform(sampling_rate=global_sample_rate, waveform=tail_padding)
        frames = []
        for i in range(segment):
            frames.append(fbank.get_frame(num_processed_frames + i))
        frames = torch.cat(frames, dim=0).unsqueeze(0)
        hyp, decoder_out, timestamp, frame_offset = model.run_greedy_search(
            frames, context_size, decoder_out, hyp,
            num_processed_frames, timestamp, frame_offset
        )
        num_processed_frames += offset
    return hyp[context_size:], timestamp


def post_process(hyp, vocab, timestamp):
    text = "".join([vocab[str(i)] for i in hyp]).replace(" ", " ").strip()
    frame_shift_ms = 10
    subsampling_factor = 4
    frame_shift_s = frame_shift_ms / 1000.0 * subsampling_factor
    real_timestamp = [round(frame_shift_s * t, 2) for t in timestamp]
    return text, real_timestamp


def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print(f"resample_audio: {original_sample_rate} HZ -> {desired_sample_rate} HZ")
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate


def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print(f"convert_channels: {original_channels} -> {desired_channels}")
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels


# ==========================================
# 豆包 API + TTS 播报
# ==========================================
def send_to_doubao(stt_text,
                   prompt="请根据以下语音识别的内容，使用中文回答，回答必须不允许包含任何非中文词汇，进行适当的回复：\n"):
    if not stt_text:
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": ENDPOINT_ID,
        "messages": [{"role": "user", "content": prompt + stt_text}],
        "enable_plugins": True,
        "stream": False
    }
    try:
        url = "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(CLOUD_CONNECT_TIMEOUT_SECONDS, CLOUD_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        result = response.json()
        reply_text = result['choices'][0]['message']['content']
        return reply_text
    except Exception as e:
        print(f"豆包请求失败: {e}")
        return None


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


def play_tts(text):
    chunks = _split_tts_text(text)
    if not chunks:
        return False

    try:
        with tts_process_lock():
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
        print(f"TTS 串口失败（L610 可能正在重启）: {e}")
    except (TimeoutError, RuntimeError) as e:
        print(f"TTS 协议失败: {e}")
    except Exception as e:
        print(f"TTS 失败: {e}")
    return False


def process_voice_file(audio_file, model, vocab):
    """在后台工作线程中执行 ASR、云请求和 TTS，绝不阻塞 GPIO 监听。"""
    try:
        send_to_dashboard("status", "正在语音转文字...")
        audio_data, sr = sf.read(audio_file)
        channels = audio_data.ndim
        audio_data, channels = ensure_channels(audio_data, channels)
        audio_data, sr = ensure_sample_rate(audio_data, sr)
        audio_data = torch.tensor(audio_data, dtype=torch.float32)

        model.init_encoder_input()
        hyp, ts = run_model(model, audio_data, sr)
        text, _ = post_process(hyp, vocab, ts)
        print(f"[识别结果] {text}")

        if not text:
            send_to_dashboard("status", "未识别到有效语音")
            return

        send_to_dashboard("asr", text)
        send_to_dashboard("status", "正在调用豆包...")
        reply = send_to_doubao(text)
        if not reply:
            send_to_dashboard("status", "豆包请求失败，按键监听仍在运行")
            return

        print(f"[豆包回复] {reply}")
        send_to_dashboard("llm", reply)
        send_to_dashboard("status", "正在语音播报...")
        if play_tts(reply):
            send_to_dashboard("status", "就绪")
        else:
            send_to_dashboard("status", "语音播报失败，请稍后重试")
    except Exception as e:
        print(f"处理失败: {e}")
        send_to_dashboard("status", "处理出错，按键监听仍在运行")
    finally:
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"[清理] 已删除临时文件: {audio_file}")
            except Exception as e:
                print(f"[清理] 删除临时文件失败: {e}")


def voice_task_worker(task_queue, busy_event, model, vocab):
    while True:
        audio_file = task_queue.get()
        try:
            if audio_file is None:
                return
            process_voice_file(audio_file, model, vocab)
        finally:
            if audio_file is not None:
                busy_event.clear()
            task_queue.task_done()


def reinitialize_gpio(old_line):
    try:
        old_line.release()
    except Exception:
        pass

    while True:
        try:
            line = init_gpio()
            print(f"[GPIO] 已重新初始化: {GPIO_CHIP} line {GPIO_LINE}")
            return line
        except Exception as e:
            print(f"[GPIO] 重建失败，1 秒后重试: {e}")
            time.sleep(1)


def resolve_model_path(explicit_path, env_name, component):
    """解析 ASR 模型：命令行 > 环境变量 > 程序目录中的标准名/唯一匹配。"""
    if explicit_path:
        return os.path.abspath(explicit_path)

    env_path = os.environ.get(env_name)
    if env_path:
        return os.path.abspath(env_path)

    preferred = [
        os.path.join(APP_DIR, f"{component}.rknn"),
        os.path.join(APP_DIR, f"{component}.onnx"),
    ]
    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    matches = []
    component_lower = component.lower()
    for root, _, files in os.walk(APP_DIR):
        for filename in files:
            lower_name = filename.lower()
            if component_lower in lower_name and lower_name.endswith((".rknn", ".onnx")):
                matches.append(os.path.join(root, filename))

    if len(matches) == 1:
        return os.path.abspath(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"未在程序目录 {APP_DIR} 中找到 {component} 的 RKNN/ONNX 模型；"
            f"可通过 --{component}_model_path 或 {env_name} 指定"
        )
    raise RuntimeError(
        f"程序目录中找到多个 {component} 模型，无法自动选择: {matches}；"
        f"请通过 --{component}_model_path 或 {env_name} 明确指定"
    )


# ==========================================
# 主程序：按键触发全链路
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按键语音助手：按住录音、松手识别+豆包回复")
    parser.add_argument("--encoder_model_path", type=str)
    parser.add_argument("--decoder_model_path", type=str)
    parser.add_argument("--joiner_model_path", type=str)
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=os.environ.get("RV1126_VOCAB_PATH", os.path.join(APP_DIR, "vocab.txt")),
    )
    parser.add_argument("--target", type=str, default="rv1126")
    parser.add_argument("--device_id", type=str, default=None)
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=os.environ.get("RV1126_AUDIO_DIR", os.path.join(APP_DIR, "audio_folder")),
    )
    args = parser.parse_args()

    try:
        args.encoder_model_path = resolve_model_path(
            args.encoder_model_path, "RV1126_ENCODER_MODEL", "encoder")
        args.decoder_model_path = resolve_model_path(
            args.decoder_model_path, "RV1126_DECODER_MODEL", "decoder")
        args.joiner_model_path = resolve_model_path(
            args.joiner_model_path, "RV1126_JOINER_MODEL", "joiner")
    except (FileNotFoundError, RuntimeError) as e:
        parser.error(str(e))

    # 1. 预加载模型和词表（启动只加载一次，保证触发后零延迟推理）
    print("[初始化] 正在加载ASR模型...")
    vocab = read_vocab(args.vocab_path)
    model = set_model(args)
    os.makedirs(args.audio_dir, exist_ok=True)
    print("[初始化] 模型加载完成")

    # 2. 初始化GPIO按键
    try:
        line = init_gpio()
    except Exception as e:
        print(f"GPIO初始化失败: {e}")
        sys.exit(1)
    print(f"[就绪] 按键监听中，GPIO: {GPIO_CHIP} line {GPIO_LINE}")
    print("[说明] 按住按键开始录音，松开自动识别并获取豆包回复")

    record_proc = None
    current_file = None
    press_start = 0
    reuse_photo_suppress_until = 0.0
    voice_tasks = queue.Queue(maxsize=1)
    voice_busy = threading.Event()
    worker_thread = threading.Thread(
        target=voice_task_worker,
        args=(voice_tasks, voice_busy, model, vocab),
        name="voice-task-worker",
        daemon=True,
    )
    worker_thread.start()

    try:
        while True:
            try:
                # 只在这里阻塞等待 GPIO；ASR/网络/TTS 均已移到后台线程。
                line.event_wait()
                event = line.event_read()
                time.sleep(DEBOUNCE_MS / 1000)
                real_level = line.get_value()
            except Exception as e:
                print(f"[GPIO] 事件读取失败，准备重建: {e}")
                if record_proc:
                    stop_record(record_proc)
                if current_file and os.path.exists(current_file):
                    try:
                        os.remove(current_file)
                    except OSError:
                        pass
                record_proc = None
                current_file = None
                line = reinitialize_gpio(line)
                continue

            # 按键按下：开始录音
            if event.type == gpiod.LineEvent.RISING_EDGE and real_level == 1:
                if record_proc is not None:
                    continue
                press_start = time.time()

                # [MODIFICATION 1] Add milliseconds to timestamp to prevent filename collision
                ms_str = f"{int((press_start % 1) * 1000):03d}"
                timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{ms_str}"

                file_name = f"record_{timestamp}.wav"
                current_file = os.path.join(args.audio_dir, file_name)
                try:
                    record_proc = start_record(current_file)
                except Exception as e:
                    print(f"[录音] 启动失败: {e}")
                    send_to_dashboard("status", "录音设备启动失败")
                    record_proc = None
                    current_file = None
                    continue
                print(f"[开始录音] {file_name}")
                send_to_dashboard("status", "正在录音...")

            # 按键松开：停止录音 → ASR → 豆包 → TTS
            elif event.type == gpiod.LineEvent.FALLING_EDGE and real_level == 0:
                if not record_proc:
                    continue

                stop_record(record_proc)

                # [MODIFICATION 2] Brief buffer to allow `arecord` to finish flushing the WAV header
                time.sleep(0.05)

                duration = time.time() - press_start
                print(f"[停止录音] 时长: {duration:.1f}s")
                completed_file = current_file
                record_proc = None
                current_file = None

                # 短按（低于录音时长阈值）：先与 C++ 做复用握手。Phase A 接受
                # 复用时绝不发送 SNAP；其他阶段收不到确认，才转为普通拍照。
                if duration < MIN_RECORD_MS / 1000:
                    now = time.monotonic()
                    if now < reuse_photo_suppress_until:
                        print("[短按] 刚完成标定复用，保护窗口内忽略拍照触发")
                        send_to_dashboard("status", "标定复用保护中，本次未拍照")
                    elif request_reuse_calibration():
                        reuse_photo_suppress_until = time.monotonic() + REUSE_PHOTO_SUPPRESS_SECONDS
                        send_to_dashboard("status", "已复用上次标定，本次未触发拍照")
                    else:
                        print("[短按] C++ 未接受标定复用，转为正常拍照触发")
                        send_photo_trigger()
                        send_to_dashboard("status", "已发送拍照识别触发")
                    # 短按录下的极短音频无用，直接清理（保持全程不占存储）
                    if completed_file and os.path.exists(completed_file):
                        try:
                            os.remove(completed_file)
                        except OSError:
                            pass  # File wasn't created yet or was already cleaned up
                    continue

                # 后台只允许一个语音任务。云端离线时任务可以超时失败，但 GPIO
                # 主循环仍会继续接收短按，不影响拍照和标定复用。
                if voice_busy.is_set():
                    print("[语音任务] 上一个任务仍在处理，本次长按录音已丢弃")
                    send_to_dashboard("status", "上一条语音仍在处理，已忽略本次长按")
                    if completed_file and os.path.exists(completed_file):
                        try:
                            os.remove(completed_file)
                        except OSError:
                            pass
                    continue

                voice_busy.set()
                try:
                    voice_tasks.put_nowait(completed_file)
                    send_to_dashboard("status", "语音任务已提交，按键监听继续运行")
                except queue.Full:
                    voice_busy.clear()
                    if completed_file and os.path.exists(completed_file):
                        try:
                            os.remove(completed_file)
                        except OSError:
                            pass

    except KeyboardInterrupt:
        print("\n[退出] 程序停止")
    finally:
        if record_proc:
            stop_record(record_proc)
        try:
            line.release()
        except Exception:
            pass
        if not voice_busy.is_set():
            try:
                voice_tasks.put_nowait(None)
                worker_thread.join(timeout=2)
            except queue.Full:
                pass
            model.release_model()
        else:
            print("[退出] 后台任务仍在运行，跳过模型主动释放并交由进程回收")
        print("[退出] 资源已释放")
