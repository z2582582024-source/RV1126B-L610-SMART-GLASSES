#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <deque>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>      // <-- ADD: needed for std::nth_element
#include <array>          // <-- ADD: needed for the median ring buffer
#include <fstream>        // <-- ADD: 标定数据持久化(读写 /userdata)
#include <cstring>        // <-- ADD: strncmp (复用标定 UDP 消息判断)
#include <iomanip>        // <-- ADD: std::setprecision (标定数据落盘精度)
#include <cerrno>         // errno / EAGAIN / ENOBUFS
#include <csignal>        // SIGPIPE
#include <atomic>         // main-loop watchdog heartbeat
#include <chrono>
#include <exception>
// I2C Headers
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
extern "C" {
    #include "vl53l5cx_api.h"
}
// Network Headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
//system
#include <cstdlib>
// ==========================================
// 1. GLOBAL STATE & CONFIGURATION
// ==========================================
const int FRAME_W = 1280;
const int FRAME_H = 720;

// --- 标定系统配置 ---
const int TARGET_CALIB_POINTS = 32;     // 6(45cm) + 12(90cm) + 6(135cm) + 8(PnP Corners)
const float ARUCO_SIZE_MM = 100.0f;
const float MIN_PIXEL_DIST = 10.0f;

// --- S型标定状态机序列 ---
int current_calib_phase = 1;            // 1: 45cm, 2: 90cm, 3: 135cm, 4: PnP Corners
size_t target_idx = 0;
double cooldown_end_time = 0.0;        // 冷却结束时间

// 定义四个阶段的最优眼动轨迹序列
const std::vector<int> seq_45cm  = {0, 1, 2, 3, 4, 5};
const std::vector<int> seq_90cm  = {6, 7, 8, 2, 1, 0, 5, 4, 3, 9, 10, 11};
const std::vector<int> seq_135cm = {6, 7, 8, 9, 10, 11};
// 前4个目标80cm，后4个目标140cm
const std::vector<int> seq_pnp   = {6, 8, 9, 11, 6, 8, 9, 11};

// ==========================================
// --- Network Coordinate & Stream Config ---
// ==========================================
const char* LOCAL_IP_ADDRESS = "127.0.0.1";  // 本机环回地址，发给板子上的 Pythn

const int LOCAL_UDP_PORT_WORLD = 8080;       // World 画面(已叠加HUD/标定/深度文本)发给本机 Python, 供网页预览

const int LOCAL_UDP_PORT_EYE = 8082;         // 新增：发送给本地 Python 的眼球端口

// 干净世界帧(未叠加任何 ArUco/HUD/深度文本): 在 frame_world 被绘制前拷贝一份,
// 原生分辨率(FRAME_W x FRAME_H)专线发送, 供 Python 侧 VLM 拍照 ROI 裁剪专用,
// 避免网页预览用的叠加内容(如 "Z: xxxmm" 文本)被裁进识别图里。
const int LOCAL_UDP_PORT_WORLD_CLEAN = 8081;

const int LOCAL_UDP_PORT_COORD = 5005; // 对应 Python 监听的端口

int udp_socket_world = -1;
int udp_socket_world_clean = -1;
int udp_socket_eye = -1;
int udp_socket_coord = -1; // 新增坐标 Socket
struct sockaddr_in world_addr, world_clean_addr, eye_addr, coord_addr;

// ==========================================
// 定义 Debug 调试信息的网络参数
// ==========================================
const int LOCAL_UDP_PORT_DEBUG = 5006;
int udp_socket_debug = -1;
struct sockaddr_in debug_addr;

// === 语音引导 UDP (C++ 只发短码 -> Python 用 H610 串口播报，串口由 Python 独占) ===
const int LOCAL_UDP_PORT_VOICE = 5009;
int udp_socket_voice = -1;
struct sockaddr_in voice_addr;
static std::vector<double> temporal_depth_buffer; //扬声器
static std::atomic<long long> vision_heartbeat_ms(0);

static long long monotonic_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void vision_watchdog_thread() {
    const long long timeout_ms = 15000;
    while (true) {
        sleep(1);
        long long heartbeat = vision_heartbeat_ms.load(std::memory_order_relaxed);
        if (heartbeat > 0 && monotonic_ms() - heartbeat > timeout_ms) {
            std::cerr << "❌ Vision main loop stalled for over " << timeout_ms
                      << "ms; exiting for service restart." << std::endl;
            std::_Exit(5);
        }
    }
}

// === 标定复用 UDP (batch_doubao.py 短按按键转发 "REUSE_CALIB" -> 本程序,
//     仅在 Phase A(等待举牌解锁标定)的整个生命周期内有效监听) ===
const int LOCAL_UDP_PORT_REUSE_CALIB = 5011;
const char* CALIB_SAVE_PATH = "/userdata/gaze_calib.dat";   // 断电不丢: 与现有 /userdata/alarm.MP3 同一持久化分区

// --- I2C ToF Config ---
const char* I2C_DEVICE = "/dev/i2c-4";       // Using your confirmed working I2C bus

// ---------------------------------------------------------
// [NEW] OV13855 Camera Intrinsics (Camera Matrix)
// ---------------------------------------------------------
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    921.3991788225999, 0.0, 648.2065385818529,
    0.0, 690.740332095819, 348.2987322575188,
    0.0, 0.0, 1.0
);
// ---------------------------------------------------------
// [NEW] OV13855 Distortion Coefficients (k1, k2, p1, p2, k3)
// ---------------------------------------------------------
cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) <<
    0.05898668049892976,
   -0.2778536852041644,
   -0.01163010852757357,
   -0.003931690035967262,
    0.2745600638309493
);

cv::Mat R_cam2tof = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
cv::Mat T_cam2tof = (cv::Mat_<double>(3, 1) << 0.0, -31.0, -22.9227);

// ToF State
uint16_t latest_tof_matrix[8][8];
std::mutex tof_mutex;
const double tof_z_offset = -22.9227;  // <-- UPDATED Z OFFSET
const int tof_roi_x1 = 350, tof_roi_x2 = 980;
const int tof_roi_y1 = 10, tof_roi_y2 = 600;

// Shared Gaze State
cv::Vec3f latest_gaze_vector(0, 0, 0);
std::mutex gaze_mutex;

// --- Calibration & Smoothing State ---
std::deque<cv::Vec3f> gaze_history;
bool calibration_started = false;
const int TRIGGER_ARUCO_ID = 0;           // Trigger marker ID
const double TRIGGER_DURATION_SEC = 3.0;  // Hold time to unlock
bool trigger_seen_continuous = false;
double trigger_start_time = 0.0;
double last_calib_time = 0.0;
double safe_min_depth = 300.0;
double safe_max_depth = 1000.0;
float smoothed_u = FRAME_W / 2.0f;
float smoothed_v = FRAME_H / 2.0f;

// ==========================================
// 调试开关: k=true 时锚定点(smoothed_u/v, 即最终发给 Python 的注视坐标)
// 强制固定在画面正中心, 便于单独测试 YOLOv8 检测框吸附与 VLM 拍照 ROI，
// 不受实际眼动/标定结果影响。k=false(默认)时完全不生效，现有逻辑不变。
// 仅在下方 Phase C(推理) 里生效一处，不影响标定 Phase A/B 及其他任何流程。
// ==========================================
bool k = false;

// === 垂直通道调试 (诊断注视点纵向漂移) ===
// 分解链路：瞳孔中心Y(眼图) - 眼球模型中心Y(眼图) = 原始垂直信号 -> gy -> 锚点 v
std::mutex eye_dbg_mutex;
float dbg_pupil_x = 0.0f, dbg_pupil_y = 0.0f;   // 瞳孔椭圆中心 (眼图像素)
float dbg_model_x = 0.0f, dbg_model_y = 0.0f;   // 眼球模型中心 (眼图像素)
bool  g_eye_found = false;                       // 本帧是否检测到有效瞳孔 (眨眼判断, 主线程内读写)

// === ToF 质量调试 (诊断 8x8 ToF 深度误判) ===
int   dbg_tof_valid = 0;     // 视线邻域内有效 ToF 网格数 (0-9)
float dbg_tof_std   = 0.0f;  // 视线邻域深度标准差(mm), 越大越不可信

// ==========================================
// 音频控制模块
// ==========================================
void init_audio_system() {
    std::cout << "🔈 Initializing RV1126 Audio Speaker..." << std::endl;
    // 1. 打开 Speaker (屏蔽输出避免刷屏)
    system("amixer -c rockchiprv1126b sset \"Speaker\" on > /dev/null 2>&1");

    // 2. 设置 DAC 音量为 250 (约 49%)
    system("amixer -c rockchiprv1126b sset \"DAC Digital\" 250 > /dev/null 2>&1");
    std::cout << "✅ Audio System Ready." << std::endl;
}

void play_beep(bool is_phase_complete = false) {
    // 末尾的 & 表示在后台运行，绝不能阻塞摄像头的 30fps 循环！
    if (is_phase_complete) {
        // 阶段完成：播放双响
        system("gst-play-1.0 /userdata/alarm_2.MP3 > /dev/null 2>&1 &");
    } else {
        // 单点完成：播放短促滴声
        system("gst-play-1.0 /userdata/alarm.MP3 > /dev/null 2>&1 &");

    }
}

// Helper function for gaze stability calculation
cv::Vec3f calculate_gaze_stddev(const std::deque<cv::Vec3f>& history) {
    if (history.empty()) return cv::Vec3f(0,0,0);
    cv::Vec3f mean(0,0,0), stddev(0,0,0);
    for (const auto& v : history) mean += v;
    mean /= (float)history.size();
    for (const auto& v : history) {
        stddev[0] += std::pow(v[0] - mean[0], 2);
        stddev[1] += std::pow(v[1] - mean[1], 2);
        stddev[2] += std::pow(v[2] - mean[2], 2);
    }
    stddev[0] = std::sqrt(stddev[0] / history.size());
    stddev[1] = std::sqrt(stddev[1] / history.size());
    stddev[2] = std::sqrt(stddev[2] / history.size());
    return stddev;
}
// =====================================================================
// One-Euro Filter: low-lag adaptive low-pass for noisy 1-D signals.
// Cuts jitter when fixating, opens up when saccading.
// Replaces the broken dynamic-alpha EMA.
// =====================================================================
class OneEuroFilter {
public:
    OneEuroFilter(double freq, double mincutoff = 1.0,
                  double beta = 0.0, double dcutoff = 1.0)
        : freq_(freq), mincutoff_(mincutoff), beta_(beta),
          dcutoff_(dcutoff), x_prev_(0), dx_prev_(0), initialized_(false) {}

    double filter(double x) {
        if (!initialized_) { x_prev_ = x; dx_prev_ = 0; initialized_ = true; return x; }
        double dx = (x - x_prev_) * freq_;
        double a_d = alpha(dcutoff_);
        double dx_hat = a_d * dx + (1 - a_d) * dx_prev_;
        double cutoff = mincutoff_ + beta_ * std::abs(dx_hat);
        double a = alpha(cutoff);
        double x_hat = a * x + (1 - a) * x_prev_;
        x_prev_ = x_hat; dx_prev_ = dx_hat;
        return x_hat;
    }
    void reset() { initialized_ = false; }
private:
    double alpha(double cutoff) {
        double tau = 1.0 / (2.0 * CV_PI * cutoff);
        double te  = 1.0 / freq_;
        return 1.0 / (1.0 + tau / te);
    }
    double freq_, mincutoff_, beta_, dcutoff_;
    double x_prev_, dx_prev_;
    bool   initialized_;
};

// Two filters for the OUTPUT pixel (replaces dynamic EMA).
// Tuned so fixation jitter < 2 px while saccades are followed within 1 frame.
static OneEuroFilter ef_u(30.0, /*mincutoff*/0.8, /*beta*/0.012);
static OneEuroFilter ef_v(30.0, /*mincutoff*/0.8, /*beta*/0.012);

// Three filters for the INPUT gaze vector (pre-mapping de-noise).
// Lower mincutoff => more smoothing of the upstream eye-tracker output,
// without touching compute_gaze_vector itself.
static OneEuroFilter gf_x(30.0, 0.6, 0.008);
static OneEuroFilter gf_y(30.0, 0.6, 0.008);
static OneEuroFilter gf_z(30.0, 0.6, 0.008);

// Median-of-N for the gaze 3-vector — kills isolated pupil-fit spikes
// (a single bad ellipse fit can punch the gaze by 0.1+ for one frame).
template <size_t N>
class VecMedian {
public:
    cv::Vec3f push(const cv::Vec3f& v) {
        buf_[idx_ % N] = v; idx_++;
        size_t m = std::min(idx_, N);
        cv::Vec3f out;
        for (int c = 0; c < 3; c++) {
            std::array<float, N> tmp{};
            for (size_t i = 0; i < m; i++) tmp[i] = buf_[i][c];
            std::nth_element(tmp.begin(), tmp.begin() + m / 2, tmp.begin() + m);
            out[c] = tmp[m / 2];
        }
        return out;
    }
private:
    std::array<cv::Vec3f, N> buf_{};
    size_t idx_ = 0;
};
static VecMedian<5> gaze_median;
// ==========================================
// 2. HARDWARE THREADS (I2C ToF Official API)
// ==========================================
void tof_i2c_thread() {
    // Initialize matrix to out-of-bounds default
    for(int r=0; r<8; r++) {
        for(int c=0; c<8; c++) latest_tof_matrix[r][c] = 2000;
    }

    while (true) {
        VL53L5CX_Configuration Dev;
        memset(&Dev, 0, sizeof(Dev));
        uint8_t status = VL53L5CX_STATUS_OK;
        uint8_t isAlive = 0;

        Dev.platform.address = 0x29; // 7-bit I2C address
        Dev.platform.fd = open(I2C_DEVICE, O_RDWR);
        if (Dev.platform.fd < 0) {
            std::cerr << "❌ Failed to open I2C bus: " << I2C_DEVICE
                      << ", retrying in 1s: " << strerror(errno) << std::endl;
            sleep(1);
            continue;
        }

        std::cout << "⏳ Resetting VL53L5CX Sensor..." << std::endl;
        VL53L5CX_Reset_Sensor(&Dev.platform);
        status = vl53l5cx_is_alive(&Dev, &isAlive);
        if (status != VL53L5CX_STATUS_OK || !isAlive) {
            std::cerr << "❌ Sensor not responding, retrying in 1s." << std::endl;
            close(Dev.platform.fd);
            sleep(1);
            continue;
        }

        std::cout << "⏳ Loading FW (Wait 3s)..." << std::endl;
        status = vl53l5cx_init(&Dev);
        if (status != VL53L5CX_STATUS_OK ||
            vl53l5cx_set_resolution(&Dev, VL53L5CX_RESOLUTION_8X8) != VL53L5CX_STATUS_OK ||
            vl53l5cx_set_ranging_frequency_hz(&Dev, 15) != VL53L5CX_STATUS_OK ||
            vl53l5cx_start_ranging(&Dev) != VL53L5CX_STATUS_OK) {
            std::cerr << "❌ TOF initialization failed, retrying in 1s." << std::endl;
            close(Dev.platform.fd);
            sleep(1);
            continue;
        }
        std::cout << "✅ TOF 8x8 @ 15Hz Ranging Started!" << std::endl;

        VL53L5CX_ResultsData Results;
        uint8_t isDataReady = 0;
        int consecutive_errors = 0;

        while (true) {
            status = vl53l5cx_check_data_ready(&Dev, &isDataReady);
            if (status != VL53L5CX_STATUS_OK) {
                if (++consecutive_errors >= 5) {
                    std::cerr << "❌ TOF runtime read failed repeatedly, reinitializing." << std::endl;
                    break;
                }
                usleep(20000);
                continue;
            }

            if (isDataReady) {
                status = vl53l5cx_get_ranging_data(&Dev, &Results);
                if (status != VL53L5CX_STATUS_OK) {
                    if (++consecutive_errors >= 5) {
                        std::cerr << "❌ TOF ranging data failed repeatedly, reinitializing." << std::endl;
                        break;
                    }
                    continue;
                }
                consecutive_errors = 0;
                {
                    std::lock_guard<std::mutex> lock(tof_mutex);
                    for (int i = 0; i < 64; i++) {
                        int r = i / 8;
                        int c = i % 8;
                        uint8_t target_status = Results.target_status[VL53L5CX_NB_TARGET_PER_ZONE * i];
                        if (target_status == 5 || target_status == 9 || target_status == 6 || target_status == 10) {
                            latest_tof_matrix[r][c] = Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE * i];
                        } else {
                            latest_tof_matrix[r][c] = 2000;
                        }
                    }
                }
                usleep(10000);
            } else {
                usleep(5000);
            }
        }

        close(Dev.platform.fd);
        sleep(1);
    }
}


// ==========================================
// 优化版：带去畸变、迭代求交与外参变换的 ToF 深度映射
// ==========================================
double get_depth_from_pixel(float u, float v, bool is_calibration = false, bool use_min_pool = false) {
    // 1. 坐标去畸变 (极其关键，保证针孔模型的几何准确性)
    std::vector<cv::Point2f> raw_pts = { cv::Point2f(u, v) };
    std::vector<cv::Point2f> undist_pts;
    cv::undistortPoints(raw_pts, undist_pts, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);
    float u_undist = undist_pts[0].x;
    float v_undist = undist_pts[0].y;

    // 相机内参提取
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);

    // 2. 迭代深度查找法 (解决视差导致的坐标偏移)
    uint16_t current_raw_depth;
    {
        std::lock_guard<std::mutex> lock(tof_mutex);
        current_raw_depth = latest_tof_matrix[4][4]; // 初始猜测：用中心区域深度
    }
    double Z_cam = (current_raw_depth <= 50 || current_raw_depth > 4000) ? 1000.0 : (double)current_raw_depth + tof_z_offset;

    int grid_x = 4, grid_y = 4;
    double fov_tan = 0.4142; // VL53L5CX FoV tan(45°/2) ≈ 0.4142

    // 迭代 2 次以精准锁定 ToF 真实落点
    for (int iter = 0; iter < 2; iter++) {
        // A. 像素 -> 相机 3D 坐标
        double X_cam = (u_undist - cx) * Z_cam / fx;
        double Y_cam = (v_undist - cy) * Z_cam / fy;
        cv::Mat P_cam = (cv::Mat_<double>(3, 1) << X_cam, Y_cam, Z_cam);

        // B. 相机 3D -> ToF 3D (应用外参)
        cv::Mat P_tof = R_cam2tof * P_cam + T_cam2tof;
        double X_tof = P_tof.at<double>(0, 0);
        double Y_tof = P_tof.at<double>(1, 0);
        double Z_tof = P_tof.at<double>(2, 0);

        if (Z_tof <= 0) return is_calibration ? -1.0 : 1000.0;

        // C. ToF 3D -> 归一化网格坐标
        double ratio_x = ((X_tof / Z_tof) + fov_tan) / (2.0 * fov_tan);
        double ratio_y = ((Y_tof / Z_tof) + fov_tan) / (2.0 * fov_tan);

        // C. ToF 3D -> 归一化网格坐标
        // 🚨 新增拦截逻辑：如果在标定模式下，ArUco超出了ToF的真实视场角，直接熔断！
        if (is_calibration) {
            if (ratio_x < 0.0 || ratio_x > 1.0 || ratio_y < 0.0 || ratio_y > 1.0) {
                return -1.0; // 越界，返回无效深度
            }
        }

        grid_x = std::max(0, std::min(7, (int)(ratio_x * 8)));
        grid_y = std::max(0, std::min(7, (int)(ratio_y * 8)));

        // D. 获取该精确网格的深度，为下一次迭代更新 Z_cam
        {
            std::lock_guard<std::mutex> lock(tof_mutex);
            current_raw_depth = latest_tof_matrix[grid_y][grid_x];
        }
        Z_cam = (current_raw_depth <= 50 || current_raw_depth > 4000) ? 1000.0 : (double)current_raw_depth + tof_z_offset;
    }

    // 3. 最终深度提取与最小池化 (边缘防抖)
    uint16_t final_raw_depth = current_raw_depth;
    if (use_min_pool) {
        uint16_t min_d = 4000;
        std::lock_guard<std::mutex> lock(tof_mutex);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = grid_y + dy, nx = grid_x + dx;
                if (ny >= 0 && ny < 8 && nx >= 0 && nx < 8) {
                    uint16_t d = latest_tof_matrix[ny][nx];
                    if (d > 50 && d <= 4000 && d < min_d) min_d = d;
                }
            }
        }
        // 如果周围有更近的有效点，取最小值防止穿透到远景
        final_raw_depth = (min_d != 4000) ? min_d : latest_tof_matrix[grid_y][grid_x];
    }

    if (final_raw_depth <= 50 || final_raw_depth > 4000) return is_calibration ? -1.0 : 1000.0;
    return (double)final_raw_depth + tof_z_offset;
}

// ==========================================
// 视线邻域稳健深度 + ToF 质量评估 (诊断 8x8 ToF 误判)
// 返回 3x3 邻域内有效深度的中位数；同时输出有效格数与深度标准差。
// valid 少 / std 大 => 该处 ToF 极不可信，推理端应保持上一帧深度而非跳变。
// ==========================================
double get_depth_robust(float u, float v, int& valid_out, float& std_out) {
    valid_out = 0; std_out = 0.0f;
    std::vector<cv::Point2f> raw_pts = { cv::Point2f(u, v) }, undist_pts;
    cv::undistortPoints(raw_pts, undist_pts, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);
    float u_u = undist_pts[0].x, v_u = undist_pts[0].y;
    double cx = camera_matrix.at<double>(0, 2), cy = camera_matrix.at<double>(1, 2);
    double fx = camera_matrix.at<double>(0, 0), fy = camera_matrix.at<double>(1, 1);

    uint16_t rawd;
    { std::lock_guard<std::mutex> lk(tof_mutex); rawd = latest_tof_matrix[4][4]; }
    double Z_cam = (rawd <= 50 || rawd > 4000) ? 1000.0 : (double)rawd + tof_z_offset;
    int gx = 4, gy = 4; double ft = 0.4142;
    for (int it = 0; it < 2; it++) {
        double X = (u_u - cx) * Z_cam / fx, Y = (v_u - cy) * Z_cam / fy;
        cv::Mat Pc = (cv::Mat_<double>(3, 1) << X, Y, Z_cam);
        cv::Mat Pt = R_cam2tof * Pc + T_cam2tof;
        double Xt = Pt.at<double>(0), Yt = Pt.at<double>(1), Zt = Pt.at<double>(2);
        if (Zt <= 0) return -1.0;
        double rx = ((Xt / Zt) + ft) / (2 * ft), ry = ((Yt / Zt) + ft) / (2 * ft);
        gx = std::max(0, std::min(7, (int)(rx * 8)));
        gy = std::max(0, std::min(7, (int)(ry * 8)));
        { std::lock_guard<std::mutex> lk(tof_mutex); rawd = latest_tof_matrix[gy][gx]; }
        Z_cam = (rawd <= 50 || rawd > 4000) ? 1000.0 : (double)rawd + tof_z_offset;
    }

    std::vector<double> vals;
    { std::lock_guard<std::mutex> lk(tof_mutex);
      for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
          int ny = gy + dy, nx = gx + dx;
          if (ny < 0 || ny > 7 || nx < 0 || nx > 7) continue;
          uint16_t d = latest_tof_matrix[ny][nx];
          if (d > 50 && d <= 4000) vals.push_back((double)d + tof_z_offset);
      } }
    valid_out = (int)vals.size();
    if (vals.empty()) return -1.0;
    std::sort(vals.begin(), vals.end());
    double med = vals[vals.size() / 2];
    double m = 0; for (double x : vals) m += x; m /= vals.size();
    double var = 0; for (double x : vals) var += (x - m) * (x - m); var /= vals.size();
    std_out = (float)std::sqrt(var);
    return med;
}


// ==========================================
// 3. NETWORK VIDEO TRANSMISSION (DUAL UDP)
// ==========================================
struct sockaddr_in pc_addr_world, pc_addr_eye;

static int create_udp_sender(const char* channel) {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cerr << "[UDP:" << channel << "] socket() failed: " << strerror(errno) << std::endl;
        return -1;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        std::cerr << "[UDP:" << channel << "] failed to enable nonblocking mode: "
                  << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }

    int send_buffer = 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    return fd;
}

static bool send_udp_datagram(int& fd, const struct sockaddr_in& addr,
                              const void* data, size_t size, const char* channel) {
    if (data == nullptr || size == 0) return false;
    if (fd < 0) fd = create_udp_sender(channel);
    if (fd < 0) return false;

    for (int attempt = 0; attempt < 2; ++attempt) {
        ssize_t sent = sendto(fd, data, size, 0,
                              reinterpret_cast<const struct sockaddr*>(&addr), sizeof(addr));
        if (sent == static_cast<ssize_t>(size)) return true;
        if (sent < 0 && errno == EINTR) continue;

        int saved_errno = (sent < 0) ? errno : EIO;
        static double last_error_log = 0.0;
        double now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        if (now - last_error_log >= 2.0) {
            std::cerr << "[UDP:" << channel << "] send failed: "
                      << strerror(saved_errno) << " (frame/event dropped)" << std::endl;
            last_error_log = now;
        }

        // 缓冲区暂满时直接丢当前帧，绝不能阻塞视觉主循环。
        if (saved_errno == EAGAIN || saved_errno == EWOULDBLOCK || saved_errno == ENOBUFS) {
            return false;
        }

        close(fd);
        fd = -1;
        if (attempt == 0) fd = create_udp_sender(channel);
        if (fd < 0) return false;
    }
    return false;
}

void setup_udp_video_stream() {
    // 1. 初始化 World 画面 Socket (指向本机 127.0.0.1:8080)
    udp_socket_world = create_udp_sender("world");
    memset(&world_addr, 0, sizeof(world_addr));
    world_addr.sin_family = AF_INET;
    world_addr.sin_port = htons(LOCAL_UDP_PORT_WORLD);
    world_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // 1.5 初始化"干净世界帧"Socket (指向本机 127.0.0.1:8081, 未叠加任何HUD/标定/深度文本)
    udp_socket_world_clean = create_udp_sender("world_clean");
    memset(&world_clean_addr, 0, sizeof(world_clean_addr));
    world_clean_addr.sin_family = AF_INET;
    world_clean_addr.sin_port = htons(LOCAL_UDP_PORT_WORLD_CLEAN);
    world_clean_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // 2. 初始化 Eye 画面 Socket (指向本机 127.0.0.1:8082)
    udp_socket_eye = create_udp_sender("eye");
    memset(&eye_addr, 0, sizeof(eye_addr));
    eye_addr.sin_family = AF_INET;
    eye_addr.sin_port = htons(LOCAL_UDP_PORT_EYE);
    eye_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // ==========================================
    // 3.初始化 Coord 坐标发送 Socket (指向本机 127.0.0.1:5005)
    // ==========================================
    udp_socket_coord = create_udp_sender("coord");
    memset(&coord_addr, 0, sizeof(coord_addr));
    coord_addr.sin_family = AF_INET;
    coord_addr.sin_port = htons(LOCAL_UDP_PORT_COORD); // 使用 5005 端口
    coord_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // ==========================================
    // 初始化 Debug Socket
    // ==========================================
    udp_socket_debug = create_udp_sender("debug");
    memset(&debug_addr, 0, sizeof(debug_addr));
    debug_addr.sin_family = AF_INET;
    debug_addr.sin_port = htons(LOCAL_UDP_PORT_DEBUG);
    debug_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // ==========================================
    // 初始化 Voice Socket (标定语音引导 -> Python 5009)
    // ==========================================
    udp_socket_voice = create_udp_sender("voice");
    memset(&voice_addr, 0, sizeof(voice_addr));
    voice_addr.sin_family = AF_INET;
    voice_addr.sin_port = htons(LOCAL_UDP_PORT_VOICE);
    voice_addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);

    // 日志打印修正
    std::cout << "[Network] World Stream -> " << LOCAL_IP_ADDRESS << ":" << LOCAL_UDP_PORT_WORLD << std::endl;
    std::cout << "[Network] World(Clean) -> " << LOCAL_IP_ADDRESS << ":" << LOCAL_UDP_PORT_WORLD_CLEAN << std::endl;
    std::cout << "[Network] Eye Stream   -> " << LOCAL_IP_ADDRESS << ":" << LOCAL_UDP_PORT_EYE << std::endl;
    std::cout << "[Network] Coord Stream -> " << LOCAL_IP_ADDRESS << ":" << LOCAL_UDP_PORT_COORD << std::endl;
    std::cout << "[Network] Debug Stream -> " << LOCAL_IP_ADDRESS << ":" << LOCAL_UDP_PORT_DEBUG << std::endl; // Debug打印

}

// ==========================================
// 发送 JSON 字符串的辅助函数
// ==========================================
void send_debug_info(const char* json_str) {
    if (json_str == nullptr) return;
    send_udp_datagram(udp_socket_debug, debug_addr, json_str, strlen(json_str), "debug");
}

// ==========================================
// 语音引导：发短码给 Python (自带节流，避免刷屏/串口过载)
// 同一提示 3s 内不重复；任意提示全局最小间隔 1.2s
// ==========================================
void send_voice(const char* code) {
    if (code == nullptr) return;
    static std::string last_code = "";
    static double last_time = 0.0;
    double now = (double)cv::getTickCount() / cv::getTickFrequency();
    bool same = (last_code == code);
    if (same && (now - last_time) < 3.0) return;   // 同码 3s 内不重播
    if (!same && (now - last_time) < 1.2) return;  // 换码也保持 1.2s 最小间隔
    last_code = code; last_time = now;
    send_udp_datagram(udp_socket_voice, voice_addr, code, strlen(code), "voice");
}

// UDP 单包安全上限：留在硬上限 65507 字节(65535 - 8字节UDP头 - 20字节IP头)之下
// 一截安全余量，避免任何路径上的额外开销把包顶爆。
static const size_t UDP_JPEG_SAFE_LIMIT = 60000;

// 固定 quality 猜不准：同一分辨率下, 画面越复杂(边缘/纹理越多)编码体积越大，
// 一个"看起来够低"的 quality 在特定场景下仍可能超限(现象: 日志里 1280x720@q40
// 依然频繁被丢帧)。改为自适应：从 start_quality 开始编码, 超限则逐步降质重试,
// 直到 min_quality 仍超限则视为分辨率本身过大, 交给调用方降分辨率兜底。
static bool encode_jpeg_within_limit(const cv::Mat& frame, std::vector<uchar>& out_buf,
                                      size_t max_bytes, int start_quality, int min_quality) {
    for (int q = start_quality; q >= min_quality; q -= 10) {
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, q };
        cv::imencode(".jpg", frame, out_buf, params);
        if (out_buf.size() <= max_bytes) return true;
    }
    return false;
}

void send_frame_over_ethernet(int& socket_fd, const struct sockaddr_in& addr, const cv::Mat& frame, int out_w, int out_h, int quality) {
    if (frame.empty()) return;
    try {

    cv::Mat stream_frame;
    cv::resize(frame, stream_frame, cv::Size(out_w, out_h));

    std::vector<uchar> jpeg_buffer;
    bool ok = encode_jpeg_within_limit(stream_frame, jpeg_buffer, UDP_JPEG_SAFE_LIMIT, quality, 10);

    if (!ok) {
        // 降质降到最低仍然超限(说明分辨率本身太大): 最后手段——整体降采样到
        // 60% 尺寸再试一次(约压缩到原体积的 36% 左右), 通常足以脱困。
        cv::Mat smaller;
        cv::resize(stream_frame, smaller, cv::Size(std::max(1, out_w * 6 / 10), std::max(1, out_h * 6 / 10)));
        ok = encode_jpeg_within_limit(smaller, jpeg_buffer, UDP_JPEG_SAFE_LIMIT, quality, 10);
        if (ok) {
            std::cerr << "⚠️ [UDP Frame] " << out_w << "x" << out_h << " 降质到最低仍超限，"
                      << "已临时降分辨率至 " << smaller.cols << "x" << smaller.rows
                      << " 发送 (dest_port=" << ntohs(addr.sin_port) << ")" << std::endl;
        }
    }

    if (!ok) {
        // 曾经的静默丢帧是真正元凶：8081 干净帧曾在固定质量下经常超过此限
        // 被整体丢弃，导致 Python 侧 udp_world_clean_receiver 永远收不到任何数据，
        // VLM_TARGET_FRAME 永远是 None（表现为短按一直报"暂无待识别帧"）。
        // 即使自适应降质+降分辨率都无法脱困(极端情况)，也要留下日志而不是静默丢弃。
        std::cerr << "⚠️ [UDP Frame Drop] " << out_w << "x" << out_h
                  << " 自适应降质/降分辨率后仍 > " << UDP_JPEG_SAFE_LIMIT
                  << " 字节上限, 本帧已丢弃 (dest_port=" << ntohs(addr.sin_port) << ")" << std::endl;
        return;
    }

        send_udp_datagram(socket_fd, addr, jpeg_buffer.data(), jpeg_buffer.size(), "video");
    } catch (const cv::Exception& e) {
        static double last_encode_error_log = 0.0;
        double now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        if (now - last_encode_error_log >= 2.0) {
            std::cerr << "[UDP Frame] OpenCV encode/resize failed, frame dropped: "
                      << e.what() << std::endl;
            last_encode_error_log = now;
        }
    }
}

// ==========================================
// 4. MACHINE LEARNING (Native OpenCV SVR)
// ==========================================
// =====================================================================
// Polynomial gaze→pixel mapper, v3 — global polynomial baseline + a local
// Gaussian-RBF residual head.
//
// Why both:
//   * The polynomial captures the smooth global mapping from (gx, gy, Z)
//     to (u, v). It is bounded and well-behaved everywhere, including at
//     extrapolation, and serves as the SAFE FALLBACK when the query is
//     far from any training point.
//   * The RBF head learns the LOCAL deviations the polynomial cannot
//     express, e.g. the asymmetric saturation that occurs at the screen
//     corners (left-corner gaze tops out near |gx|=0.31 while right-corner
//     gaze reaches |gx|=0.42 in our calibration data).
//   * RBF basis functions decay smoothly to zero outside the training
//     cube, so far-from-training queries collapse to the polynomial-only
//     prediction. No extrapolation surprises.
//   * Corner samples are weighted 10x in the RBF fit so the screen-edge
//     shape is locked in despite there being only 8 corner points out of 58.
// =====================================================================
class NativeGazeMapper {
public:
    void train(const std::vector<std::vector<float>>& features,
               const std::vector<std::vector<float>>& targets) {
        const int N = (int)features.size();
        if (N < 12) { is_trained_ = false; return; }

        // ---- 1) Per-axis centring statistics (same as v2) ----
        double mx = 0, my = 0, zmin = 1e9, zmax = -1e9;
        for (int i = 0; i < N; i++) {
            mx += features[i][0];
            my += features[i][1];
            if (features[i][2] < zmin) zmin = features[i][2];
            if (features[i][2] > zmax) zmax = features[i][2];
        }
        mx_ = (float)(mx / N);
        my_ = (float)(my / N);
        z_min_train_ = (float)zmin;
        z_max_train_ = (float)zmax;
        z_mid_   = 0.5f * (z_min_train_ + z_max_train_);
        z_scale_ = std::max(1.0f, 0.5f * (z_max_train_ - z_min_train_));

        // ---- 2) Polynomial baseline fit (9-term centred basis) ----
        cv::Mat A(N, NUM_POLY_TERMS, CV_64F);
        cv::Mat bU(N, 1, CV_64F), bV(N, 1, CV_64F);
        for (int i = 0; i < N; i++) {
            double gx = features[i][0] - mx_;
            double gy = features[i][1] - my_;
            double zn = (features[i][2] - z_mid_) / z_scale_;   // 归一化深度 ∈ ~[-1,1]
            fill_poly_row(A.ptr<double>(i), gx, gy, zn);
            bU.at<double>(i, 0) = targets[i][0];
            bV.at<double>(i, 0) = targets[i][1];
        }
        cv::Mat AtA = A.t() * A;
        cv::Mat regP = cv::Mat::eye(NUM_POLY_TERMS, NUM_POLY_TERMS, CV_64F) * 0.1;
        cv::solve(AtA + regP, A.t() * bU, w_u_, cv::DECOMP_CHOLESKY);
        cv::solve(AtA + regP, A.t() * bV, w_v_, cv::DECOMP_CHOLESKY);

        // Polynomial residuals (the RBF fits these)
        cv::Mat predU = A * w_u_, predV = A * w_v_;
        cv::Mat resU = bU - predU;
        cv::Mat resV = bV - predV;

        double rmsU_poly = cv::norm(resU) / std::sqrt((double)N);
        double rmsV_poly = cv::norm(resV) / std::sqrt((double)N);

        // ---- 3) Cache the training inputs for the RBF kernel evaluation ----
        n_train_ = N;
        train_gx_.assign(N, 0.f);
        train_gy_.assign(N, 0.f);
        train_z_ .assign(N, 0.f);
        for (int i = 0; i < N; i++) {
            train_gx_[i] = features[i][0];     // raw, not centred — kernel
            train_gy_[i] = features[i][1];     // distance is invariant to centring
            train_z_ [i] = features[i][2];
        }

        // ---- 4) Build the RBF kernel matrix K (N x N) on training points ----
        cv::Mat K(N, N, CV_64F);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                K.at<double>(i, j) = kernel(train_gx_[i], train_gy_[i], train_z_[i],
                                            train_gx_[j], train_gy_[j], train_z_[j]);
            }
        }

        // ---- 5) Diagonal weights — corners weighted 10x (indices 50..57
        //         under the existing calibration protocol).
        //         Also ridge-regularise to control how peaked the RBF is.
        cv::Mat W = cv::Mat::zeros(N, N, CV_64F);
        for (int i = 0; i < N; i++) {
            // The first 50 samples are the ToF in-FoV phase (NEAR/MID/FAR),
            // the last 8 are the explicit corner samples. If the training
            // set ever exceeds 58 (e.g. extra rounds), every sample beyond
            // index 50 is by construction a corner — keep the 10x weight.
            W.at<double>(i, i) = (i >= 50) ? 10.0 : 1.0;
        }
        cv::Mat regR = cv::Mat::eye(N, N, CV_64F) * 0.1;
        cv::Mat lhs  = K.t() * W * K + regR;
        cv::solve(lhs, K.t() * W * resU, alpha_u_, cv::DECOMP_CHOLESKY);
        cv::solve(lhs, K.t() * W * resV, alpha_v_, cv::DECOMP_CHOLESKY);

        // ---- 6) Combined-fit residuals so the user can see the improvement ----
        cv::Mat correctedU = predU + K * alpha_u_;
        cv::Mat correctedV = predV + K * alpha_v_;
        double rmsU_full = cv::norm(correctedU - bU) / std::sqrt((double)N);
        double rmsV_full = cv::norm(correctedV - bV) / std::sqrt((double)N);

        std::cout << "[Mapper] Centring  mx=" << mx_ << "  my=" << my_
                  << "  Z=[" << z_min_train_ << "," << z_max_train_ << "]\n";
        std::cout << "[Mapper] Poly RMS   U=" << rmsU_poly << "px  V=" << rmsV_poly << "px\n";
        std::cout << "[Mapper] Hybrid RMS U=" << rmsU_full << "px  V=" << rmsV_full << "px\n";

        // === NEW DEBUG CODE: PRINT CALIBRATION MATRICES ===
        std::cout << "==================================================\n";
        std::cout << "🎯 [CALIBRATION MATRIX / POLYNOMIAL WEIGHTS] 🎯\n";
        std::cout << "Basis: [1, gx, gy, gx^2, gx*gy, gy^2, zn, gx*zn, gy*zn]  (zn=normalized depth)\n";
        std::cout << "w_u (X-axis weights): [";
        for(int k = 0; k < NUM_POLY_TERMS; k++) {
            std::cout << w_u_.at<double>(k, 0) << (k < NUM_POLY_TERMS - 1 ? ", " : "]\n");
        }
        std::cout << "w_v (Y-axis weights): [";
        for(int k = 0; k < NUM_POLY_TERMS; k++) {
            std::cout << w_v_.at<double>(k, 0) << (k < NUM_POLY_TERMS - 1 ? ", " : "]\n");
        }
        std::cout << "RBF Alphas U (First 5 of " << n_train_ << "): [";
        for(int k = 0; k < std::min(5, n_train_); k++) std::cout << alpha_u_.at<double>(k, 0) << ", ";
        std::cout << "...]\n";
        std::cout << "==================================================\n";
        // ==================================================

        is_trained_ = true;
    }

    cv::Point2f predict(float gx, float gy, float z) const {
        if (!is_trained_) return cv::Point2f(0, 0);

        // Clamp Z to trained range — same safety as v2.
        float z_eff = z;
        if (z_eff < z_min_train_) z_eff = z_min_train_;
        if (z_eff > z_max_train_) z_eff = z_max_train_;

        // Polynomial prediction.
        double row[NUM_POLY_TERMS];
        double zn = ((double)z_eff - z_mid_) / z_scale_;
        fill_poly_row(row, (double)gx - mx_, (double)gy - my_, zn);
        double u = 0, v = 0;
        for (int k = 0; k < NUM_POLY_TERMS; k++) {
            u += row[k] * w_u_.at<double>(k, 0);
            v += row[k] * w_v_.at<double>(k, 0);
        }

        // RBF correction. Each training point contributes
        // alpha_i * exp(-||(query - train_i)||² / (2 σ²)) to the residual,
        // and decays to zero when the query is far from training data.
        for (int i = 0; i < n_train_; i++) {
            double k_i = kernel((double)gx, (double)gy, (double)z_eff,
                                train_gx_[i], train_gy_[i], train_z_[i]);
            u += k_i * alpha_u_.at<double>(i, 0);
            v += k_i * alpha_v_.at<double>(i, 0);
        }

        return cv::Point2f((float)u, (float)v);
    }
    bool trained() const { return is_trained_; }

private:
    // ---- Polynomial part ----
    static constexpr int NUM_POLY_TERMS = 9;
    // 第 3 个参数现在是【归一化深度 zn】，不是毫米原值。
    // 旧版 1/Z(mm) 使深度列数值仅 ~0.001，被岭回归压成 0 → 模型对深度完全失明。
    static void fill_poly_row(double* r, double gx, double gy, double zn) {
        r[0]=1;     r[1]=gx;       r[2]=gy;
        r[3]=gx*gx; r[4]=gx*gy;    r[5]=gy*gy;
        r[6]=zn;    r[7]=gx*zn;    r[8]=gy*zn;
    }

    // ---- RBF part. σ chosen empirically:
    //      σ_g = 0.12 (about half the median pairwise gaze distance in our
    //                  calibration data ~ 0.26 — small enough to localise
    //                  the corner correction, large enough that nearby
    //                  ToF points reinforce each other);
    //      σ_z = 0.30  (in scaled-Z units; trained Z spans ~0.3–1.2 so
    //                   this puts ~3-σ across the whole Z range).
    double kernel(double gx1, double gy1, double z1,
                  double gx2, double gy2, double z2) const {
        const double sg = 0.12;
        // [MODIFICATION 1] Mute the Z-axis in the RBF distance calculation.
        // By increasing sz to 10000.0, 'dz' becomes astronomically small.
        // The RBF will now provide stable corner corrections based solely
        // on your eye vector (gx, gy), ignoring ToF depth fluctuations completely.
        const double sz = 10000.0;

        double dgx = (gx1 - gx2) / sg;
        double dgy = (gy1 - gy2) / sg;
        double dz  = (z1  - z2 ) / sz;

        double d2  = dgx*dgx + dgy*dgy + dz*dz;
        return std::exp(-0.5 * d2);
    }

    cv::Mat w_u_, w_v_;          // polynomial weights
    cv::Mat alpha_u_, alpha_v_;  // RBF residual weights, one per training point
    int     n_train_ = 0;
    std::vector<float> train_gx_, train_gy_, train_z_;
    float   mx_ = 0.f, my_ = 0.f;
    float   z_min_train_ = 300.f, z_max_train_ = 1200.f;
    float   z_mid_ = 750.f, z_scale_ = 450.f;
    bool    is_trained_ = false;
};

// ==========================================
// 标定数据持久化 (断电不丢): 只保存/加载训练前的原始采样点
// (gx, gy, z) -> (u, v)，复用时重新灌回 gaze_mapper.train()，
// 不改动 NativeGazeMapper 本身、也不序列化其内部矩阵。
// ==========================================
// 文件格式(与实际磁盘文件一致)：
//   第 1 行  : 数据条数 N (例如 32)
//   第 2..N+1 行 : 每行 5 个浮点数 "gx gy z u v"
void save_calib_data(const std::vector<std::vector<float>>& features,
                      const std::vector<std::vector<float>>& targets) {
    std::ofstream ofs(CALIB_SAVE_PATH, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "⚠️ [CALIB][SAVE] 打开失败, 无法写入 " << CALIB_SAVE_PATH
                  << " (检查目录是否存在/是否可写)" << std::endl;
        return;
    }
    ofs << std::fixed << std::setprecision(6);
    ofs << features.size() << "\n";                         // 首行: 数据条数
    for (size_t i = 0; i < features.size(); i++) {
        ofs << features[i][0] << " " << features[i][1] << " " << features[i][2] << " "
            << targets[i][0] << " " << targets[i][1] << "\n";
    }
    ofs.flush();
    ofs.close();
    std::cout << "💾 [CALIB][SAVE] 已保存: 首行条数=" << features.size()
              << ", 随后 " << features.size() << " 行(每行 gx gy z u v) -> " << CALIB_SAVE_PATH << std::endl;
}

bool load_calib_data(std::vector<std::vector<float>>& features,
                      std::vector<std::vector<float>>& targets) {
    std::cout << "📂 [CALIB][LOAD] 尝试打开标定文件: " << CALIB_SAVE_PATH << std::endl;
    std::ifstream ifs(CALIB_SAVE_PATH);
    if (!ifs.is_open()) {
        std::cout << "ℹ️ [CALIB][LOAD] 打开失败(文件不存在或无读权限) -> 将走全新标定流程" << std::endl;
        return false;
    }
    std::cout << "✅ [CALIB][LOAD] 文件已成功打开" << std::endl;

    // 1) 读取首行声明的数据条数
    size_t declared = 0;
    if (!(ifs >> declared)) {
        std::cerr << "⚠️ [CALIB][LOAD] 读取首行条数失败(文件为空/损坏) -> 判定无效, 全新标定" << std::endl;
        return false;
    }
    std::cout << "🔢 [CALIB][LOAD] 首行声明条数 N = " << declared << std::endl;
    if (declared < 12 || declared > 1000) {
        std::cerr << "⚠️ [CALIB][LOAD] 声明条数 " << declared
                  << " 不合理(至少需 12, 见 NativeGazeMapper::train) -> 判定无效, 全新标定" << std::endl;
        return false;
    }

    // 2) 按声明条数逐行解析, 每行 5 个浮点数
    std::vector<std::vector<float>> f, t;
    f.reserve(declared); t.reserve(declared);
    size_t ok = 0;
    for (size_t i = 0; i < declared; i++) {
        float gx, gy, z, u, v;
        if (ifs >> gx >> gy >> z >> u >> v) {
            f.push_back({gx, gy, z});
            t.push_back({u, v});
            ok++;
        } else {
            std::cerr << "⚠️ [CALIB][LOAD] 第 " << (i + 1) << " 行解析失败(字段缺失/非数字), 已中止" << std::endl;
            break;
        }
    }
    std::cout << "📊 [CALIB][LOAD] 实际成功解析 " << ok << " / " << declared << " 条" << std::endl;

    // 3) 严格校验: 实际解析条数必须与声明一致, 否则判定损坏
    if (ok != declared) {
        std::cerr << "⚠️ [CALIB][LOAD] 解析条数与声明不符 -> 判定文件损坏/不完整, 安全回退全新标定" << std::endl;
        return false;
    }

    features = std::move(f);
    targets = std::move(t);
    std::cout << "🎯 [CALIB][LOAD] 历史标定数据加载成功, 共 " << ok << " 点, 可供短按复用" << std::endl;
    return true;
}

// ==========================================
// 5. EYE TRACKING LOGIC
// ==========================================
// ==========================================
// PnP 深度推算 (用于弥补 ToF 视场角外的边缘盲区)
// ==========================================
double get_pnp_depth(const std::vector<cv::Point2f>& corners, const cv::Mat& cam_mat, const cv::Mat& dist_coeff) {
    if (corners.size() != 4) return -1.0;

    float half_s = ARUCO_SIZE_MM / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        cv::Point3f(-half_s,  half_s, 0),
        cv::Point3f( half_s,  half_s, 0),
        cv::Point3f( half_s, -half_s, 0),
        cv::Point3f(-half_s, -half_s, 0)
    };

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(obj_pts, corners, cam_mat, dist_coeff, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);

    if (!success) return -1.0;

    // === NEW: convert PnP perpendicular-Z to ToF-equivalent depth at the
    //          marker centre, so corner samples live in the same feature
    //          space as the 50 ToF samples. ===
    cv::Point2f c = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
    std::vector<cv::Point2f> ud;
    cv::undistortPoints(std::vector<cv::Point2f>{c}, ud, cam_mat, dist_coeff, cv::noArray(), cam_mat);

    double fx = cam_mat.at<double>(0,0), fy = cam_mat.at<double>(1,1);
    double cx = cam_mat.at<double>(0,2), cy = cam_mat.at<double>(1,2);
    double Zc = tvec.at<double>(2,0);
    double Xc = (ud[0].x - cx) * Zc / fx;
    double Yc = (ud[0].y - cy) * Zc / fy;

    // Apply the same cam->tof transform used by get_depth_from_pixel,
    // and return the resulting Z-along-tof-axis.
    cv::Mat P_cam = (cv::Mat_<double>(3,1) << Xc, Yc, Zc);
    cv::Mat P_tof = R_cam2tof * P_cam + T_cam2tof;
    return P_tof.at<double>(2, 0);   // <-- now in the SAME UNITS as ToF samples
}
// --- 5.1 Eye Tracking Globals ---
std::vector<cv::RotatedRect> ray_lines;
std::vector<cv::Point> model_centers;
int max_rays = 100;
cv::Point prev_model_center_avg(320, 240);
double max_observed_distance = 0.0;
std::vector<cv::Point> stored_intersections;

// --- 5.2 Pure Math & Image Helpers ---
cv::Mat crop_to_aspect_ratio(const cv::Mat& image, int width = 640, int height = 480) {
    int current_height = image.rows;
    int current_width = image.cols;
    double desired_ratio = (double)width / height;
    double current_ratio = (double)current_width / current_height;

    cv::Mat cropped_img;
    if (current_ratio > desired_ratio) {
        int new_width = (int)(desired_ratio * current_height);
        int offset = (current_width - new_width) / 2;
        cropped_img = image(cv::Rect(offset, 0, new_width, current_height));
    } else {
        int new_height = (int)(current_width / desired_ratio);
        int offset = (current_height - new_height) / 2;
        cropped_img = image(cv::Rect(0, offset, current_width, new_height));
    }
    cv::Mat resized;
    cv::resize(cropped_img, resized, cv::Size(width, height));
    return resized;
}

cv::Mat apply_binary_threshold(const cv::Mat& image, int darkestPixelValue, int addedThreshold) {
    int threshold_val = darkestPixelValue + addedThreshold;
    cv::Mat thresholded_image;
    cv::threshold(image, thresholded_image, threshold_val, 255, cv::THRESH_BINARY_INV);
    return thresholded_image;
}

cv::Point get_darkest_area(const cv::Mat& image) {
    int ignoreBounds = 20, imageSkipSize = 10, searchArea = 20, internalSkipSize = 5;
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;

    double min_sum = 1e9;
    cv::Point darkest_point(320, 240);

    for (int y = ignoreBounds; y < gray.rows - ignoreBounds; y += imageSkipSize) {
        for (int x = ignoreBounds; x < gray.cols - ignoreBounds; x += imageSkipSize) {
            double current_sum = 0;
            int num_pixels = 0;
            for (int dy = 0; dy < searchArea; dy += internalSkipSize) {
                if (y + dy >= gray.rows) break;
                for (int dx = 0; dx < searchArea; dx += internalSkipSize) {
                    if (x + dx >= gray.cols) break;
                    current_sum += gray.at<uchar>(y + dy, x + dx);
                    num_pixels++;
                }
            }
            if (current_sum < min_sum && num_pixels > 0) {
                min_sum = current_sum;
                darkest_point = cv::Point(x + searchArea / 2, y + searchArea / 2);
            }
        }
    }
    return darkest_point;
}

cv::Mat mask_outside_square(const cv::Mat& image, cv::Point center, int size) {
    int half_size = size / 2;
    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    int top_left_x = std::max(0, center.x - half_size);
    int top_left_y = std::max(0, center.y - half_size);
    int bottom_right_x = std::min(image.cols, center.x + half_size);
    int bottom_right_y = std::min(image.rows, center.y + half_size);

    cv::Rect roi(top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y);
    mask(roi).setTo(cv::Scalar::all(255));
    cv::Mat result;
    cv::bitwise_and(image, mask, result);
    return result;
}

std::vector<cv::Point> optimize_contours_by_angle(const std::vector<cv::Point>& contour, const cv::Mat& image) {
    if (contour.empty()) return contour;
    int spacing = std::max(1, (int)(contour.size() / 25));
    std::vector<cv::Point> filtered_points;

    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return contour;
    cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    double cos_threshold = std::cos(60.0 * CV_PI / 180.0);

    for (size_t i = 0; i < contour.size(); ++i) {
        cv::Point2f current_point = contour[i];
        cv::Point2f prev_point = contour[(i - spacing + contour.size()) % contour.size()];
        cv::Point2f next_point = contour[(i + spacing) % contour.size()];

        cv::Point2f vec1 = prev_point - current_point;
        cv::Point2f vec2 = next_point - current_point;
        cv::Point2f vec_to_centroid = centroid - current_point;
        cv::Point2f avg_vec = (vec1 + vec2) * 0.5f;

        double norm1 = cv::norm(vec_to_centroid);
        double norm2 = cv::norm(avg_vec);

        if (norm1 * norm2 > 0) {
            double dot = vec_to_centroid.dot(avg_vec);
            if (dot / (norm1 * norm2) >= cos_threshold) {
                filtered_points.push_back(cv::Point(current_point.x, current_point.y));
            }
        }
    }
    return filtered_points;
}

std::vector<std::vector<cv::Point>> filter_contours_by_area_and_return_largest(
    const std::vector<std::vector<cv::Point>>& contours, double pixel_thresh, double ratio_thresh) {
    double max_area = 0;
    std::vector<cv::Point> largest_contour;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= pixel_thresh) {
            cv::Rect rect = cv::boundingRect(contour);
            double ratio = std::max((double)rect.width / rect.height, (double)rect.height / rect.width);
            if (ratio <= ratio_thresh && area > max_area) {
                max_area = area; largest_contour = contour;
            }
        }
    }
    if (largest_contour.empty()) return {};
    return {largest_contour};
}

struct PixelCheckResult { double absolute_thick; double ratio_under_ellipse; };

PixelCheckResult check_contour_pixels(const std::vector<cv::Point>& contour, cv::Size image_shape) {
    if (contour.size() < 5) return {0, 0};
    cv::Mat contour_mask = cv::Mat::zeros(image_shape, CV_8UC1);
    std::vector<std::vector<cv::Point>> temp_contours = {contour};
    cv::drawContours(contour_mask, temp_contours, -1, cv::Scalar(255), 1);

    cv::Mat ellipse_mask_thick = cv::Mat::zeros(image_shape, CV_8UC1);
    cv::Mat ellipse_mask_thin = cv::Mat::zeros(image_shape, CV_8UC1);
    cv::RotatedRect ellipse = cv::fitEllipse(contour);

    // [🚨 SAFETY GUARD] Reject impossible math
    if (ellipse.size.width <= 0 || ellipse.size.height <= 0 || std::isnan(ellipse.size.width) || std::isnan(ellipse.size.height)) {
        return {0, 0};
    }

    cv::ellipse(ellipse_mask_thick, ellipse, cv::Scalar(255), 10);
    cv::ellipse(ellipse_mask_thin, ellipse, cv::Scalar(255), 4);

    cv::Mat overlap_thick, overlap_thin;
    cv::bitwise_and(contour_mask, ellipse_mask_thick, overlap_thick);
    cv::bitwise_and(contour_mask, ellipse_mask_thin, overlap_thin);

    double absolute_pixel_total_thick = cv::countNonZero(overlap_thick);
    double absolute_pixel_total_thin = cv::countNonZero(overlap_thin);
    double total_border_pixels = cv::countNonZero(contour_mask);

    double ratio = (total_border_pixels > 0) ? (absolute_pixel_total_thin / total_border_pixels) : 0;
    return {absolute_pixel_total_thick, ratio};
}

std::vector<double> check_ellipse_goodness(const cv::Mat& binary_image, const std::vector<cv::Point>& contour) {
    std::vector<double> goodness = {0, 0, 0};
    if (contour.size() < 5) return goodness;

    cv::RotatedRect ellipse = cv::fitEllipse(contour);

    // [🚨 SAFETY GUARD] Reject impossible math
    if (ellipse.size.width <= 0 || ellipse.size.height <= 0 || std::isnan(ellipse.size.width) || std::isnan(ellipse.size.height)) {
        return goodness;
    }

    cv::Mat mask = cv::Mat::zeros(binary_image.size(), CV_8UC1);
    cv::ellipse(mask, ellipse, cv::Scalar(255), -1);

    double ellipse_area = cv::countNonZero(mask);
    cv::Mat covered_mask;
    cv::bitwise_and(binary_image, mask, covered_mask);
    double covered_pixels = cv::countNonZero(covered_mask);

    if (ellipse_area == 0) return goodness;
    goodness[0] = covered_pixels / ellipse_area;
    goodness[2] = std::min(ellipse.size.width / ellipse.size.height, ellipse.size.height / ellipse.size.width);
    return goodness;
}

bool find_line_intersection(const cv::RotatedRect& ellipse1, const cv::RotatedRect& ellipse2, cv::Point& intersection) {
    double cx1 = ellipse1.center.x, cy1 = ellipse1.center.y;
    double minor_axis1 = ellipse1.size.height;
    double angle1_rad = ellipse1.angle * CV_PI / 180.0;

    double cx2 = ellipse2.center.x, cy2 = ellipse2.center.y;
    double minor_axis2 = ellipse2.size.height;
    double angle2_rad = ellipse2.angle * CV_PI / 180.0;

    double dx1 = (minor_axis1 / 2.0) * std::cos(angle1_rad);
    double dy1 = (minor_axis1 / 2.0) * std::sin(angle1_rad);
    double dx2 = (minor_axis2 / 2.0) * std::cos(angle2_rad);
    double dy2 = (minor_axis2 / 2.0) * std::sin(angle2_rad);

    double det = -dx1 * dy2 + dx2 * dy1;
    if (std::abs(det) < 1e-6) return false;

    double bx = cx2 - cx1, by = cy2 - cy1;
    double t1 = (-dy2 * bx + dx2 * by) / det;
    intersection.x = std::round(cx1 + t1 * dx1);
    intersection.y = std::round(cy1 + t1 * dy1);
    return true;
}

cv::Point compute_average_intersection(cv::Mat& frame, const std::vector<cv::RotatedRect>& current_rays, int N, int M, int spacing) {
    if (current_rays.size() < 2 || N < 2) return cv::Point(0, 0);

    std::vector<cv::RotatedRect> selected_lines = current_rays;
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(selected_lines.begin(), selected_lines.end(), g);
    if (selected_lines.size() > (size_t)N) selected_lines.resize(N);

    for (size_t i = 0; i < selected_lines.size() - 1; ++i) {
        if (std::abs(selected_lines[i].angle - selected_lines[i+1].angle) >= 2.0) {
            cv::Point inter;
            if (find_line_intersection(selected_lines[i], selected_lines[i+1], inter)) {
                if (inter.x >= 0 && inter.x < frame.cols && inter.y >= 0 && inter.y < frame.rows) {
                    stored_intersections.push_back(inter);
                }
            }
        }
    }

    if (stored_intersections.size() > (size_t)M) stored_intersections.erase(stored_intersections.begin(), stored_intersections.begin() + (stored_intersections.size() - M));
    if (stored_intersections.empty()) return cv::Point(0, 0);

    double sum_x = 0, sum_y = 0;
    for (const auto& pt : stored_intersections) { sum_x += pt.x; sum_y += pt.y; }
    return cv::Point(std::round(sum_x / stored_intersections.size()), std::round(sum_y / stored_intersections.size()));
}

cv::Point update_and_average_point(std::vector<cv::Point>& point_list, cv::Point new_point, int N) {
    point_list.push_back(new_point);
    if (point_list.size() > (size_t)N) point_list.erase(point_list.begin());
    double sum_x = 0, sum_y = 0;
    for (const auto& p : point_list) { sum_x += p.x; sum_y += p.y; }
    return cv::Point(std::round(sum_x / point_list.size()), std::round(sum_y / point_list.size()));
}

// --- 5.3 Modified Compute Gaze Vector ---
void compute_gaze_vector(double x, double y, double center_x, double center_y,
                         cv::Vec3f& out_sphere_center, cv::Vec3f& out_gaze_direction) {
    double viewport_width = 640.0, viewport_height = 480.0, far_clip = 100.0;
    double fov_y_rad = 45.0 * CV_PI / 180.0;
    double aspect_ratio = viewport_width / viewport_height;

    cv::Vec3f camera_position(0.0f, 0.0f, 3.0f);
    double half_height_far = std::tan(fov_y_rad / 2.0) * far_clip;
    double half_width_far = half_height_far * aspect_ratio;

    double ndc_x = (2.0 * x) / viewport_width - 1.0;
    double ndc_y = 1.0 - (2.0 * y) / viewport_height;

    cv::Vec3f far_point(ndc_x * half_width_far, ndc_y * half_height_far, camera_position[2] - far_clip);
    cv::Vec3f ray_direction = cv::normalize(far_point - camera_position);
    ray_direction = -ray_direction;

    double inner_radius = 1.0 / 1.05;
    double sphere_offset_x = (center_x / viewport_width) * 2.0 - 1.0;
    double sphere_offset_y = 1.0 - (center_y / viewport_height) * 2.0;
    cv::Vec3f sphere_center(sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0);

    cv::Vec3f origin = camera_position, direction = -ray_direction, L = origin - sphere_center;

    double a = direction.dot(direction), b = 2.0 * direction.dot(L), c = L.dot(L) - inner_radius * inner_radius;
    double discriminant = b * b - 4 * a * c;
    cv::Vec3f target_direction, intersection_point;

    if (discriminant < 0) {
        double t = -direction.dot(L) / direction.dot(direction);
        intersection_point = origin + t * direction;
        target_direction = cv::normalize(intersection_point - sphere_center);
    } else {
        double sqrt_disc = std::sqrt(discriminant);
        double t1 = (-b - sqrt_disc) / (2 * a), t2 = (-b + sqrt_disc) / (2 * a);
        double t = -1;
        if (t1 > 0 && t2 > 0) t = std::min(t1, t2);
        else if (t1 > 0) t = t1;
        else if (t2 > 0) t = t2;

        if (t < 0) return;
        intersection_point = origin + t * direction;
        target_direction = cv::normalize(intersection_point - sphere_center);
    }

    cv::Vec3f circle_local_center = cv::normalize(cv::Vec3f(0.0f, 0.0f, inner_radius));
    cv::Vec3f rotation_axis = circle_local_center.cross(target_direction);
    double rotation_axis_norm = cv::norm(rotation_axis);
    out_sphere_center = sphere_center;

    if (rotation_axis_norm < 1e-6) {
        out_gaze_direction = circle_local_center;
    } else {
        rotation_axis /= rotation_axis_norm;
        double dot = std::max(-1.0, std::min(1.0, (double)circle_local_center.dot(target_direction)));
        double angle_rad = std::acos(dot);
        double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad), t_ = 1.0 - cos_a;
        double rx = rotation_axis[0], ry = rotation_axis[1], rz = rotation_axis[2];

        cv::Matx33f rot_matrix(
            t_*rx*rx + cos_a,    t_*rx*ry - sin_a*rz, t_*rx*rz + sin_a*ry,
            t_*rx*ry + sin_a*rz, t_*ry*ry + cos_a,    t_*ry*rz - sin_a*rx,
            t_*rx*rz - sin_a*ry, t_*ry*rz + sin_a*rx, t_*rz*rz + cos_a
        );
        cv::Vec3f gaze_local(0.0f, 0.0f, inner_radius);
        out_gaze_direction = cv::normalize(rot_matrix * gaze_local);
    }

    if (cv::norm(out_gaze_direction) > 0) {
        std::lock_guard<std::mutex> lock(gaze_mutex);
        latest_gaze_vector = out_gaze_direction;
    }
}

// --- 5.4 Consolidated Frame Processing ---
void process_eye_frame(cv::Mat& frame) {
    g_eye_found = false;   // 默认无瞳孔(可能在眨眼)，成功拟合后才置 true
    frame = crop_to_aspect_ratio(frame);
    cv::Point darkest_point = get_darkest_area(frame);

    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    int darkest_pixel_value = gray_frame.at<uchar>(darkest_point.y, darkest_point.x);

    cv::Mat thresh_strict = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 5), darkest_point, 250);
    cv::Mat thresh_med = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 15), darkest_point, 250);
    cv::Mat thresh_relaxed = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 25), darkest_point, 250);

    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
    cv::Mat image_array[] = {thresh_relaxed, thresh_med, thresh_strict};

    std::vector<cv::Point> final_contours;
    double max_goodness = 0;
    bool ellipse_found = false;

    for (int i = 0; i < 3; ++i) {
        cv::Mat dilated_image;
        cv::dilate(image_array[i], dilated_image, kernel, cv::Point(-1,-1), 2);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        auto reduced = filter_contours_by_area_and_return_largest(contours, 1000, 3.0);
        if (!reduced.empty() && reduced[0].size() >= 5) {
            auto goodness = check_ellipse_goodness(dilated_image, reduced[0]);
            auto pixels = check_contour_pixels(reduced[0], dilated_image.size());
            double final_goodness = goodness[0] * pixels.absolute_thick * pixels.absolute_thick * pixels.ratio_under_ellipse;

            if (final_goodness > max_goodness) {
                max_goodness = final_goodness; final_contours = reduced[0]; ellipse_found = true;
            }
        }
    }

    if (!ellipse_found) return;

    final_contours = optimize_contours_by_angle(final_contours, gray_frame);
    if (final_contours.size() < 5) return;

    cv::RotatedRect final_rotated_rect = cv::fitEllipse(final_contours);

    // [🚨 SAFETY GUARD] 与 check_contour_pixels/check_ellipse_goodness 一致的防护，
    // 但这里之前缺失: fitEllipse 在退化输入(如 optimize_contours_by_angle 筛完后
    // 点几乎共线)下可能返回负数/NaN 的 size，若不拦截会在下方 cv::ellipse() 里
    // 触发 "box.size.width >= 0 && box.size.height >= 0" 断言，直接崩溃整个进程。
    if (final_rotated_rect.size.width <= 0 || final_rotated_rect.size.height <= 0 ||
        std::isnan(final_rotated_rect.size.width) || std::isnan(final_rotated_rect.size.height)) {
        return;  // 本帧瞳孔拟合退化，直接放弃(等同未检测到瞳孔)，不污染 ray_lines
    }

    ray_lines.push_back(final_rotated_rect);
    if (ray_lines.size() > (size_t)max_rays) ray_lines.erase(ray_lines.begin());

    cv::Point model_center_average(320, 240);
    cv::Point model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5);

    if (model_center.x != 0 || model_center.y != 0) {
        model_center_average = update_and_average_point(model_centers, model_center, 200);
    }
    if (model_center_average.x == 320) model_center_average = prev_model_center_avg;
    if (model_center_average.x != 0) prev_model_center_avg = model_center_average;

    // Draw visual tracking data on the eye frame for the UDP stream
    cv::ellipse(frame, final_rotated_rect, cv::Scalar(0, 255, 255), 2);
    cv::circle(frame, model_center_average, 6, cv::Scalar(255, 50, 50), -1);
    cv::line(frame, model_center_average, cv::Point((int)final_rotated_rect.center.x, (int)final_rotated_rect.center.y), cv::Scalar(0, 0, 255), 2);

    // === 垂直通道调试：记录瞳孔/模型中心，并在眼图上叠加 ===
    {
        std::lock_guard<std::mutex> lock(eye_dbg_mutex);
        dbg_pupil_x = final_rotated_rect.center.x;
        dbg_pupil_y = final_rotated_rect.center.y;
        dbg_model_x = (float)model_center_average.x;
        dbg_model_y = (float)model_center_average.y;
    }
    // 灰色水平参考线 = 模型中心Y；瞳孔越偏离它，垂直信号越强
    cv::line(frame, cv::Point(0, model_center_average.y), cv::Point(frame.cols, model_center_average.y), cv::Scalar(80, 80, 80), 1);
    char eye_dbg[96];
    snprintf(eye_dbg, sizeof(eye_dbg), "pY=%.0f mY=%.0f dY=%+.0f",
             final_rotated_rect.center.y, (float)model_center_average.y,
             final_rotated_rect.center.y - (float)model_center_average.y);
    cv::putText(frame, eye_dbg, cv::Point(5, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    g_eye_found = true;   // 成功拟合到瞳孔椭圆 => 眼睛睁开
    cv::Vec3f dummy_sphere_center, dummy_gaze_dir;
    compute_gaze_vector(final_rotated_rect.center.x, final_rotated_rect.center.y,
                        model_center_average.x, model_center_average.y,
                        dummy_sphere_center, dummy_gaze_dir);
}
// ==========================================
// 6. MAIN INTEGRATION LOOP
// ==========================================
static int open_reuse_calib_listener() {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cerr << "[CALIB UDP] socket() failed: " << strerror(errno) << std::endl;
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(LOCAL_UDP_PORT_REUSE_CALIB);
    addr.sin_addr.s_addr = inet_addr(LOCAL_IP_ADDRESS);
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[CALIB UDP] bind(5011) failed: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        std::cerr << "[CALIB UDP] nonblocking setup failed: " << strerror(errno) << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

int main() {
    // A dropped remote terminal must not terminate the camera/tracking loop.
    signal(SIGHUP, SIG_IGN);
    signal(SIGPIPE, SIG_IGN);
    std::cout << "🚀 Starting RV1126B Dual-Camera Vision System..." << std::endl;
    try {

    // 提前绑定"复用标定"监听端口 (在相机/ToF等慢速初始化之前)，
    // 避免用户在初始化阶段按下的按键因端口尚未监听而被内核丢弃。
    // 设为非阻塞：真正的监听窗口是"整个阶段A的生命周期"(见下方 Phase A 循环体内的轮询)，
    // 而不是启动后的固定时长，因此这里不需要任何超时等待。
    int reuse_calib_sock = open_reuse_calib_listener();

    // 1. Start ToF Sensor Thread
    std::thread tof_thread(tof_i2c_thread);
    tof_thread.detach();

    // 初始化扬声器
    init_audio_system();

    // 2. Setup UDP Stream
    setup_udp_video_stream();

    // 3. Initialize Dual Cameras
    std::string world_pipeline =
    "v4l2src device=/dev/video23 ! "
    "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false";

    cv::VideoCapture cap_eye;
    cv::VideoCapture cap_world;
    auto open_dual_cameras = [&]() -> bool {
        cap_eye.release();
        cap_world.release();
        try {
            bool eye_ok = cap_eye.open("/dev/video52", cv::CAP_V4L2);
            if (eye_ok) {
                cap_eye.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap_eye.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            }
            bool world_ok = cap_world.open(world_pipeline, cv::CAP_GSTREAMER);
            if (!eye_ok || !world_ok) {
                std::cerr << "❌ Camera open failed (eye=" << eye_ok
                          << ", world=" << world_ok << "), retrying." << std::endl;
                cap_eye.release();
                cap_world.release();
                return false;
            }
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "❌ OpenCV camera initialization exception: " << e.what() << std::endl;
            cap_eye.release();
            cap_world.release();
            return false;
        }
    };

    while (!open_dual_cameras()) sleep(1);

    std::cout << "✅ Dual Cameras Initialized." << std::endl;
    vision_heartbeat_ms.store(monotonic_ms(), std::memory_order_relaxed);
    std::thread vision_watchdog(vision_watchdog_thread);
    vision_watchdog.detach();

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    NativeGazeMapper gaze_mapper;

    std::vector<std::vector<float>> calib_features;
    std::vector<std::vector<float>> calib_targets;
    std::vector<cv::Point2f> captured_raw_centers;

    cv::Mat frame_world, frame_eye;

    // ==========================================
    // 复用标定候选数据：若上次保存过完整标定，Phase A 的整个生命周期内
    // (从进入循环起，直到用户举牌 3 秒解锁进入 Phase B 为止) 短按按键都可
    // 触发复用，不再局限于开机后的固定时长。真正的监听在下面 Phase A 循环体内。
    // ==========================================
    std::vector<std::vector<float>> saved_calib_features, saved_calib_targets;
    bool reuse_calib_available = load_calib_data(saved_calib_features, saved_calib_targets);
    if (reuse_calib_available) {
        std::cout << "⏳ [CALIB] Found saved calibration (" << saved_calib_features.size()
                  << " pts). Short-press button any time during Phase A (waiting for trigger marker) to reuse."
                  << std::endl;
        // 注: "REUSE_TIP" 语音提示已从 Python 端 VOICE_MAP 移除(不在当前可用语音资源内),
        // 这里不再调用 send_voice。复用功能本身不依赖语音提示，仅少了开局的口头提醒；
        // 复用成功后仍会播报 "REUSE_OK"(该键保留在 VOICE_MAP 中)。
    } else {
        if (reuse_calib_sock >= 0) close(reuse_calib_sock);   // 没有可复用的存档，直接释放该 socket
        reuse_calib_sock = -1;
    }

    while (true) {
        vision_heartbeat_ms.store(monotonic_ms(), std::memory_order_relaxed);
        // ==========================================
        // 定义当前帧的调试状态变量
        // ==========================================
        std::string debug_phase = "UNKNOWN";
        std::string debug_msg = "NORMAL";
        double debug_z = -1.0;

        // cap_eye >> frame_eye;
        // cap_world >> frame_world;
        // 第 1 步：快速抓取 (Grab)
        // grab() 只是向硬件/驱动发送一个“快门”指令，将当前最新的一帧锁定到内存，
        // 而不进行任何耗时的解码或矩阵拷贝。它的执行时间极短（微秒级）。
        bool eye_grabbed = false;
        bool world_grabbed = false;
        bool eye_retrieved = false;
        bool world_retrieved = false;
        try {
            eye_grabbed = cap_eye.grab();
            world_grabbed = cap_world.grab();

        // 第 2 步：提取解码 (Retrieve)
        // retrieve() 才是真正去刚才锁定的内存中，把裸流数据解码为 OpenCV 可用的 BGR 矩阵。
            eye_retrieved = eye_grabbed && cap_eye.retrieve(frame_eye);
            world_retrieved = world_grabbed && cap_world.retrieve(frame_world);
        } catch (const cv::Exception& e) {
            std::cerr << "❌ Camera capture exception, reopening: " << e.what() << std::endl;
            while (!open_dual_cameras()) sleep(1);
            continue;
        }

        static int consecutive_camera_failures = 0;
        if (!eye_retrieved || !world_retrieved || frame_eye.empty() || frame_world.empty()) {
            ++consecutive_camera_failures;
            if (consecutive_camera_failures == 1 || consecutive_camera_failures % 30 == 0) {
                std::cerr << "⚠️ Dropped frame from camera(s), consecutive="
                          << consecutive_camera_failures << std::endl;
            }
            if (consecutive_camera_failures >= 30) {
                std::cerr << "♻️ Camera stream stalled, reopening both cameras." << std::endl;
                while (!open_dual_cameras()) sleep(1);
                consecutive_camera_failures = 0;
            } else {
                usleep(10000);
            }
            continue;
        }
        consecutive_camera_failures = 0;

        // 干净世界帧快照：必须在下方任何 cv::putText/rectangle/circle 绘制到
        // frame_world 之前拷贝，专供 Python 侧 VLM 拍照 ROI 使用，
        // 保证不含 ArUco 标定框/举牌进度条/"Z: xxxmm"深度文本等 HUD 叠加。
        cv::Mat frame_world_clean = frame_world.clone();

        // Process Eye Tracking
        process_eye_frame(frame_eye);

        cv::Vec3f current_gaze;
        {
            std::lock_guard<std::mutex> lock(gaze_mutex);
            current_gaze = latest_gaze_vector;
        }

        // Update gaze history for stability check
        gaze_history.push_back(current_gaze);
        if (gaze_history.size() > 5) gaze_history.pop_front();

        // === 垂直通道调试量 (供 JSON/HUD) ===
        double debug_gy = current_gaze[1];
        double debug_eyedy = 0.0;
        {
            std::lock_guard<std::mutex> lock(eye_dbg_mutex);
            debug_eyedy = (double)(dbg_pupil_y - dbg_model_y);
        }

        // Process World & Calibration
        if (!gaze_mapper.trained()) {
            cv::Mat gray;
            cv::cvtColor(frame_world, gray, cv::COLOR_BGR2GRAY);
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::aruco::detectMarkers(gray, dictionary, corners, ids, params);

            if (!calibration_started) {
                // ==========================================
                // PHASE A: WAITING FOR AUTOMATIC TRIGGER
                // ==========================================
                debug_phase = "WAIT_TRIGGER";  //Debug

                // --- 复用标定检测：整个阶段A生命周期内每帧非阻塞轮询一次，
                //     不再局限于开机后的固定时长。命中后立刻训练并跳过标定，
                //     下一帧起 gaze_mapper.trained()==true 自动进入 Phase C。---
                if (reuse_calib_available) {
                    if (reuse_calib_sock < 0) {
                        static double next_reuse_socket_retry = 0.0;
                        double now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
                        if (now >= next_reuse_socket_retry) {
                            reuse_calib_sock = open_reuse_calib_listener();
                            next_reuse_socket_retry = now + 1.0;
                        }
                    }
                    static bool reuse_listen_announced = false;
                    if (!reuse_listen_announced) {
                        std::cout << "👂 [CALIB][PhaseA] 正在监听短按复用信号 (UDP " << LOCAL_UDP_PORT_REUSE_CALIB
                                  << "), 举牌解锁标定前随时可短按复用..." << std::endl;
                        reuse_listen_announced = true;
                    }
                    // 注: 原本这里会周期性重播 "REUSE_TIP" 语音提示，该键已从 Python
                    // 端 VOICE_MAP 移除(不在当前可用语音资源内)，故不再调用 send_voice；
                    // 复用监听功能本身不受影响，仅少了口头提醒，用户仍可随时短按复用。

                    char reuse_buf[64] = {0};
                    ssize_t reuse_n = -1;
                    struct sockaddr_in reuse_sender;
                    memset(&reuse_sender, 0, sizeof(reuse_sender));
                    socklen_t reuse_sender_len = sizeof(reuse_sender);
                    if (reuse_calib_sock >= 0) {
                        reuse_n = recvfrom(
                            reuse_calib_sock,
                            reuse_buf,
                            sizeof(reuse_buf) - 1,
                            0,
                            reinterpret_cast<struct sockaddr*>(&reuse_sender),
                            &reuse_sender_len);
                        if (reuse_n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                            std::cerr << "[CALIB UDP] recv failed, rebuilding listener: "
                                      << strerror(errno) << std::endl;
                            close(reuse_calib_sock);
                            reuse_calib_sock = -1;
                        }
                    }
                    if (reuse_n > 0) {
                        std::cout << "📨 [CALIB][PhaseA] 收到复用端口数据 " << reuse_n << " 字节: \""
                                  << reuse_buf << "\"" << std::endl;
                        if (strncmp(reuse_buf, "REUSE_CALIB", 11) == 0) {
                            // 先确认本次按键已被标定复用消费。batch_doubao.py 收到
                            // REUSE_ACCEPTED 后不会再向 YOLO 发送 SNAP，从协议层保证
                            // 一次短按只能执行“复用”或“拍照”中的一个动作。
                            const char reuse_ack[] = "REUSE_ACCEPTED";
                            ssize_t ack_n = sendto(
                                reuse_calib_sock,
                                reuse_ack,
                                sizeof(reuse_ack) - 1,
                                0,
                                reinterpret_cast<const struct sockaddr*>(&reuse_sender),
                                reuse_sender_len);
                            if (ack_n != static_cast<ssize_t>(sizeof(reuse_ack) - 1)) {
                                std::cerr << "[CALIB UDP] failed to send REUSE_ACCEPTED: "
                                          << strerror(errno) << std::endl;
                            }
                            calib_features = saved_calib_features;
                            calib_targets = saved_calib_targets;
                            gaze_mapper.train(calib_features, calib_targets);
                            reuse_calib_available = false;
                            if (reuse_calib_sock >= 0) close(reuse_calib_sock);
                            reuse_calib_sock = -1;
                            std::cout << "✅ [CALIB][PhaseA-复用触发] 收到 REUSE_CALIB, 复用上次标定 -> 已加载 "
                                      << calib_features.size() << " 点并完成训练, 跳过本次标定, 进入识别(Phase C)。" << std::endl;
                            send_voice("REUSE_OK");
                            continue;   // 本帧到此结束，不再执行下方 Phase A 剩余逻辑
                        } else {
                            std::cout << "↩️ [CALIB][PhaseA] 数据非 REUSE_CALIB, 忽略。" << std::endl;
                        }
                    }
                }

                bool found_trigger = false;
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] == TRIGGER_ARUCO_ID) { found_trigger = true; break; }
                }

                if (found_trigger) {
                    double current_time = (double)cv::getTickCount() / cv::getTickFrequency();
                    if (!trigger_seen_continuous) {
                        trigger_seen_continuous = true;
                        trigger_start_time = current_time;
                        std::cout << "⏳ Trigger marker detected. Hold steady..." << std::endl;
                    } else {
                        double elapsed = current_time - trigger_start_time;
                        int progress_width = (int)(400 * (elapsed / TRIGGER_DURATION_SEC));
                        cv::rectangle(frame_world, cv::Rect(50, 50, progress_width, 30), cv::Scalar(0, 255, 0), -1);
                        cv::putText(frame_world, "HOLD TO UNLOCK CALIBRATION", cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
                        debug_msg = "HOLD TO UNLOCK CALIBRATION";   //Debug

                        if (elapsed >= TRIGGER_DURATION_SEC) {
                            calibration_started = true;
                            // 手动标定已经开始进入 Phase B：停止监听复用信号，
                            // 避免用户此后误按按键把正在进行的手动标定打断成"复用旧数据"。
                            if (reuse_calib_available) {
                                reuse_calib_available = false;
                                if (reuse_calib_sock >= 0) close(reuse_calib_sock);
                                reuse_calib_sock = -1;
                            }
                            std::cout << "✅ Calibration Unlocked! Proceeding to Phase 1..." << std::endl;
                        }
                    }
                } else {
                    if (trigger_seen_continuous) {
                        std::cout << "❌ Trigger lost. Resetting timer." << std::endl;
                        trigger_seen_continuous = false;
                    }
                    cv::putText(frame_world, "SHOW MARKER 0 TO UNLOCK", cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                    debug_msg = "SHOW MARKER 0 TO UNLOCK";   //Debug

                }
            } else {
                // ==========================================
                // PHASE B: DATA COLLECTION (4-Phase State Machine)
                // ==========================================
                debug_phase = "Calibration";  //Debug
                int active_target_id = -1;
                double target_Z_min = 0, target_Z_max = 0;
                std::string phase_name = "";

                // 1. 根据当前状态机获取目标参数
                if (current_calib_phase == 1) {
                    active_target_id = seq_45cm[target_idx];
                    target_Z_min = 350; target_Z_max = 500;
                    phase_name = "PHASE 1 (45cm - ToF)";
                } else if (current_calib_phase == 2) {
                    active_target_id = seq_90cm[target_idx];
                    target_Z_min = 800; target_Z_max = 1000;
                    phase_name = "PHASE 2 (90cm - ToF)";
                } else if (current_calib_phase == 3) {
                    active_target_id = seq_135cm[target_idx];
                    target_Z_min = 1200; target_Z_max = 1400;
                    phase_name = "PHASE 3 (135cm - ToF)";
                } else if (current_calib_phase == 4) {
                    active_target_id = seq_pnp[target_idx];
                    if (target_idx < 4) {
                        target_Z_min = 650; target_Z_max = 950; // 前4个点 80cm
                        phase_name = "PHASE 4 (80cm - PnP)";
                    } else {
                        target_Z_min = 1150; target_Z_max = 1550; // 后4个点 140cm
                        phase_name = "PHASE 4 (140cm - PnP)";
                    }
                }

                if (!ids.empty()) {
                    cv::aruco::drawDetectedMarkers(frame_world, corners, ids, cv::Scalar(100, 100, 100));

                    // 2. 寻找当前目标 ID
                    int target_corner_index = -1;
                    for (size_t i = 0; i < ids.size(); ++i) {
                        if (ids[i] == active_target_id) {
                            target_corner_index = i;
                            break;
                        }
                    }

                    if (target_corner_index != -1) {
                        cv::Point2f center = (corners[target_corner_index][0] + corners[target_corner_index][1] +
                                              corners[target_corner_index][2] + corners[target_corner_index][3]) / 4.0f;

                        // 目标孤立高亮
                        std::vector<std::vector<cv::Point2f>> single_corner = {corners[target_corner_index]};
                        cv::aruco::drawDetectedMarkers(frame_world, single_corner, cv::noArray(), cv::Scalar(0, 0, 255));

                        // 强制冷却期拦截：给用户转移视线的时间
                        double current_time = (double)cv::getTickCount() / cv::getTickFrequency();
                        if (current_time < cooldown_end_time) {
                            // UI 提示正在移动
                            cv::putText(frame_world, "MOVING TO NEXT...", cv::Point(center.x - 80, center.y - 30),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);

                            temporal_depth_buffer.clear(); // 清空旧数据
                            last_calib_time = current_time; // 压制 0.4 秒计时器
                            continue; // 拦截！不执行下方的检测
                        }

                        // ==========================================
                        // 动态深度源与智能边缘回退 (Smart PnP Fallback)
                        // ==========================================
                        double current_depth = -1.0;
                        std::string depth_source_str = "ToF"; // 默认标记为 ToF

                        if (current_calib_phase == 4) {
                            current_depth = get_pnp_depth(corners[target_corner_index], camera_matrix, dist_coeffs);
                            depth_source_str = "PnP"; // Phase 4 原生 PnP
                        } else {
                            // === PnP + ToF 交叉校验 (拒绝污染样本) ===
                            // 近/中点 PnP 精度高、ToF 在视场角内也可靠；两者一致才算干净样本。
                            // 不一致 => 该点深度被污染，直接否决(current_depth=-1)重新采。
                            double d_tof = get_depth_from_pixel(center.x, center.y, true, false);
                            double d_pnp = get_pnp_depth(corners[target_corner_index], camera_matrix, dist_coeffs);
                            bool tof_ok = (d_tof > 0 && d_tof <= 1500);
                            bool pnp_ok = (d_pnp > 0 && d_pnp <= 2000);

                            if (tof_ok && pnp_ok) {
                                double rel = std::abs(d_tof - d_pnp) / d_pnp;
                                if (rel <= 0.12) {                 // 一致(≤12%): 取均值, 最干净
                                    current_depth = 0.5 * (d_tof + d_pnp);
                                    depth_source_str = "ToF+PnP";
                                } else {                           // 冲突: 污染样本, 否决本次采样
                                    current_depth = -1.0;
                                    depth_source_str = "CONFLICT";
                                    cv::putText(frame_world, "ToF/PnP CONFLICT -> ADJUST ANGLE",
                                                cv::Point(center.x - 120, center.y + 60),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                                    send_voice("CONFLICT");
                                }
                            } else if (tof_ok) {
                                current_depth = d_tof; depth_source_str = "ToF";
                            } else if (pnp_ok && active_target_id >= 6 && active_target_id <= 11) {
                                current_depth = d_pnp; depth_source_str = "PnP(Fallback)"; // ToF 盲区边缘 -> PnP 救场
                                cv::putText(frame_world, "ToF BLINDSPOT -> PnP FALLBACK",
                                            cv::Point(center.x - 100, center.y + 60),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                            }
                        }

                        // ==========================================
                        // 四角物理边界限制 (Physical ROI Limiters)
                        // ==========================================
                        bool in_safe_zone = (current_depth > 0);
                        cv::Rect target_roi;
                        bool draw_roi = false;
                        std::string corner_name = "";

                        // 只在 Phase 4 (四角极限标定) 启用物理框限制
                        if (current_calib_phase == 4) {
                            int roi_w = 240, roi_h = 180; // 宽容的限制框大小 (基于 1280x720 画幅)

                            if (active_target_id == 6) { // 左上角
                                target_roi = cv::Rect(10, 10, roi_w, roi_h);
                                corner_name = "TOP-LEFT";
                                draw_roi = true;
                            } else if (active_target_id == 8) { // 右上角
                                target_roi = cv::Rect(FRAME_W - roi_w - 10, 10, roi_w, roi_h);
                                corner_name = "TOP-RIGHT";
                                draw_roi = true;
                            } else if (active_target_id == 9) { // 右下角
                                target_roi = cv::Rect(FRAME_W - roi_w - 10, FRAME_H - roi_h - 10, roi_w, roi_h);
                                corner_name = "BOTTOM-RIGHT";
                                draw_roi = true;
                            } else if (active_target_id == 11) { // 左下角
                                target_roi = cv::Rect(10, FRAME_H - roi_h - 10, roi_w, roi_h);
                                corner_name = "BOTTOM-LEFT";
                                draw_roi = true;
                            }

                            if (draw_roi) {
                                // 1. 在画面上画出醒目的目标区域框
                                cv::rectangle(frame_world, target_roi, cv::Scalar(0, 0, 255), 3, cv::LINE_8);
                                cv::putText(frame_world, "MOVE BOARD TO: " + corner_name,
                                            cv::Point(target_roi.x + 10, target_roi.y + 30),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

                                // 2. 严格空间拦截：目标码的中心点必须落入该物理框内！
                                if (!target_roi.contains(center)) {
                                    in_safe_zone = false; // 否决安全状态，阻断数据采集
                                    cv::putText(frame_world, "MOVE MARKER INTO RED BOX!",
                                                cv::Point(center.x - 120, center.y + 60),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                                    debug_msg = "MOVE BOARD TO RED BOX!";
                                    temporal_depth_buffer.clear(); // 清空脏数据
                                }
                            }
                        }

                        cv::Vec3f gaze_std = calculate_gaze_stddev(gaze_history);
                        float gxy_mag = std::sqrt(current_gaze[0]*current_gaze[0] + current_gaze[1]*current_gaze[1]);
                        bool is_gaze_stable = (gaze_history.size() == 5
                                            && gaze_std[0] < 0.04 && gaze_std[1] < 0.04
                                            && std::abs(current_gaze[0]) < 0.85
                                            && std::abs(current_gaze[1]) < 0.85
                                            && gxy_mag > 0.02);

                        char req_str[100];
                        sprintf(req_str, "%s | LOOK AT ID: %d", phase_name.c_str(), active_target_id);
                        cv::putText(frame_world, req_str, cv::Point(center.x - 80, center.y - 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

                        if (in_safe_zone && is_gaze_stable) {

                            // [关键修复] 不允许无效数据污染中位数缓冲池！
                            if (current_depth > 0) {
                                temporal_depth_buffer.push_back(current_depth);
                            }

                            double current_time = (double)cv::getTickCount() / cv::getTickFrequency();

                            if (current_time - last_calib_time > 0.5) {
                                // === 稳健深度门控：中位数 + MAD 剔除离群 + 抖动否决 ===
                                double final_stable_depth = current_depth;
                                bool   depth_is_clean = false;
                                bool   depth_measuring = false;      // 样本仍在累积, 别急着判 NOISY
                                size_t n = temporal_depth_buffer.size();
                                if (n < 6) {
                                    // [关键修复] 样本不足 6 帧: 继续累积, 绝不清空/判 NOISY,
                                    // 否则第一个点会每帧压1个又清空, 永远攒不够 -> 卡死在 NOISY。
                                    depth_measuring = true;
                                } else {
                                    std::sort(temporal_depth_buffer.begin(), temporal_depth_buffer.end());
                                    double med = temporal_depth_buffer[n / 2];
                                    std::vector<double> dev(n);
                                    for (size_t i = 0; i < n; i++) dev[i] = std::abs(temporal_depth_buffer[i] - med);
                                    std::sort(dev.begin(), dev.end());
                                    double robust_std = 1.4826 * dev[n / 2];
                                    double band = std::max(15.0, 2.0 * robust_std);
                                    double sum = 0; int cnt = 0;
                                    for (size_t i = 0; i < n; i++)
                                        if (std::abs(temporal_depth_buffer[i] - med) <= band) { sum += temporal_depth_buffer[i]; cnt++; }
                                    final_stable_depth = (cnt > 0) ? sum / cnt : med;
                                    double phase_nominal = 0.5 * (target_Z_min + target_Z_max);
                                    // 放宽: 面阵 ToF 本身有 15~40mm 正常波动(叠加 PnP 抖动更大)。
                                    // 只否决"真·剧烈跳变"(上百 mm), 允许至少 60mm 或 12% 标称的正常噪声。
                                    double max_allowed_std = std::max(60.0, 0.12 * phase_nominal);
                                    depth_is_clean = (robust_std <= max_allowed_std);
                                }

                                if (depth_measuring) {
                                    // 样本累积中: 提示但不清空, 让缓冲继续攒到 6 帧再评估
                                    cv::putText(frame_world, "MEASURING DEPTH...",
                                                cv::Point(center.x - 100, center.y + 30),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2);
                                    debug_msg = "MEASURING...";
                                } else if (!depth_is_clean) {
                                    cv::putText(frame_world, "DEPTH NOISY - HOLD STILL",
                                                cv::Point(center.x - 130, center.y + 30),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                                    debug_msg = "DEPTH NOISY";
                                    send_voice("NOISY");
                                    temporal_depth_buffer.clear();
                                } else if (final_stable_depth >= target_Z_min && final_stable_depth <= target_Z_max) {

                                    // 记录高精度数据
                                    std::vector<cv::Point2f> raw_pts = { center };
                                    std::vector<cv::Point2f> undist_pts;
                                    cv::undistortPoints(raw_pts, undist_pts, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);

                                    // 角点 PnP 深度不可信(小码/倾斜/边缘畸变, 跨度可达 292mm)，
                                    // 训练用标称距离作为 Z 特征，避免把 ±150mm 噪声灌进模型
                                    float feature_z = (float)final_stable_depth;
                                    if (current_calib_phase == 4)
                                        feature_z = (target_idx < 4) ? 800.0f : 1400.0f;
                                    calib_features.push_back({current_gaze[0], current_gaze[1], feature_z});
                                    calib_targets.push_back({undist_pts[0].x, undist_pts[0].y});
                                    last_calib_time = current_time;

                                    // 扩充输出：加入深度来源、眼球注视向量(gx, gy) 以及世界相机中的像素坐标(x, y)
                                    printf("[CALIB SUCCESS] %s | ID: %2d | Src: %-13s | Z: %6.1fmm | Gaze: (%5.2f, %5.2f) | Pix: (%4.0f, %4.0f) | Pts: %2d/%d\n",
                                            phase_name.c_str(),
                                            active_target_id,
                                            depth_source_str.c_str(),
                                            final_stable_depth,
                                            current_gaze[0], current_gaze[1],
                                            center.x, center.y,
                                            (int)calib_features.size(),
                                            TARGET_CALIB_POINTS);

                                    target_idx++;

                                    // ==========================================
                                    // 采集成功，立刻进入 2 秒冷却期
                                    // ==========================================
                                    cooldown_end_time = current_time + 2;

                                    // 阶段切换与音频反馈触发
                                    // 注: "PHASE_DONE"/"NEXT" 语音键已从 Python 端 VOICE_MAP
                                    // 移除(不在当前可用语音资源内), 这里不再调用 send_voice；
                                    // play_beep() 是独立的 WAV 音效播放(与 VOICE_MAP/TTS 无关)，
                                    // 保留不变，仍能提供阶段/单点完成的音频反馈。
                                    if (current_calib_phase == 1 && target_idx >= seq_45cm.size()) {
                                        current_calib_phase = 2; target_idx = 0;
                                        std::cout << "🚀 PHASE 1 COMPLETE! Step back to 90cm." << std::endl;
                                        play_beep(true); // 阶段完成长音
                                    } else if (current_calib_phase == 2 && target_idx >= seq_90cm.size()) {
                                        current_calib_phase = 3; target_idx = 0;
                                        std::cout << "🚀 PHASE 2 COMPLETE! Step back to 135cm." << std::endl;
                                        play_beep(true); // 阶段完成长音
                                    } else if (current_calib_phase == 3 && target_idx >= seq_135cm.size()) {
                                        current_calib_phase = 4; target_idx = 0;
                                        std::cout << "🚀 PHASE 3 COMPLETE! Move to 80cm for Corner PnP." << std::endl;
                                        play_beep(true); // 阶段完成长音
                                    } else if (current_calib_phase == 4 && target_idx >= seq_pnp.size()) {
                                        std::cout << "🎯 All " << TARGET_CALIB_POINTS << " points collected! Training SVR model..." << std::endl;
                                        cv::putText(frame_world, "TRAINING MODEL...", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 4);
                                        debug_msg = "TRAINING MODEL...";
                                        play_beep(true); send_voice("DONE"); // 全部完成长音(DONE 仍是有效语音键)

                                        gaze_mapper.train(calib_features, calib_targets);
                                        save_calib_data(calib_features, calib_targets);   // 断电不丢: 覆盖保存本次标定, 只留最新一份
                                        ef_u.reset(); ef_v.reset();
                                        gf_x.reset(); gf_y.reset(); gf_z.reset();
                                        std::cout << "✅ Training Complete! Entering Inference Phase." << std::endl;
                                    } else {
                                        play_beep(false); // 单点完成短音
                                    }

                                    // 清空缓冲池，为下一个目标做准备
                                    temporal_depth_buffer.clear();

                                } else {
                                    // 距离不对
                                    char dist_warn[100];
                                    if (final_stable_depth < target_Z_min) { sprintf(dist_warn, "MOVE FURTHER! (< %d)", (int)target_Z_min); send_voice("FAR"); }
                                    else { sprintf(dist_warn, "MOVE CLOSER! (> %d)", (int)target_Z_max); send_voice("NEAR"); }
                                    cv::putText(frame_world, dist_warn, cv::Point(center.x - 100, center.y + 30),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
                                    debug_msg = dist_warn;

                                    temporal_depth_buffer.clear(); // 距离不对也清空，重新采样
                                }
                            }
                        } else if (!is_gaze_stable) {
                            cv::putText(frame_world, "EYE UNSTABLE", cv::Point(center.x - 60, center.y + 30),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
                            debug_msg = "EYE UNSTABLE...";

                            temporal_depth_buffer.clear(); // 视线抖动，立刻清空脏数据
                        }
                    } else {
                        char missing_str[100];
                        sprintf(missing_str, "WHERE IS ID: %d?", active_target_id);
                        cv::putText(frame_world, missing_str, cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
                        debug_msg = missing_str;
                    }

                    if (!gaze_mapper.trained()) {
                        char status_text[100];
                        sprintf(status_text, "CALIB: %d/%d | PHASE %d", (int)calib_features.size(), TARGET_CALIB_POINTS, current_calib_phase);
                        cv::putText(frame_world, status_text, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 165, 255), 3);
                    }

                } else {
                    cv::putText(frame_world, "SEARCHING FOR BOARD...", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
                    debug_msg = "SEARCHING FOR BOARD...";
                }
            } // 结束 Phase B
        }
        else {
            // ==========================================
            // PHASE C: INFERENCE  (rebuilt)
            // ==========================================
            debug_phase = "Anchoring";  //Debug
            // 1) Pre-filter the upstream gaze: median-of-5 + 1-€ per axis.
            //    This is purely downstream of compute_gaze_vector — eye-tracking
            //    algorithms are untouched. We just clean the signal.
            cv::Vec3f g_med = gaze_median.push(current_gaze);
            cv::Vec3f g_filt(
                (float)gf_x.filter(g_med[0]),
                (float)gf_y.filter(g_med[1]),
                (float)gf_z.filter(g_med[2])
            );

            bool is_blinking = (std::abs(g_filt[0]) > 0.90f || std::abs(g_filt[1]) > 0.90f);
            if (is_blinking) {
                cv::circle(frame_world, cv::Point((int)smoothed_u, (int)smoothed_v), 20, cv::Scalar(128,128,128), 2);
                cv::putText(frame_world, "LOST (Blink)", cv::Point((int)smoothed_u + 25, (int)smoothed_v - 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);

            } else {

                static double depth_running = 650.0;
                static int    bad_probe_count = 0;
                const  double Z_FALLBACK = 650.0;   // mid of trained range

                // [MODIFICATION 2] Disable 'use_min_pool'.
                // Changing the 4th argument to 'false' turns the ToF sensor from a
                // "shotgun" back into a "sniper rifle". It will only lock onto the
                // depth of the exact pixel you are looking at, destroying the gravity well.
                int   tvalid = 0; float tstd = 0.0f;
                double sampled_a = get_depth_robust(smoothed_u, smoothed_v, tvalid, tstd);
                dbg_tof_valid = tvalid; dbg_tof_std = tstd;   // ToF 质量 -> JSON/HUD
                cv::Point2f rough = gaze_mapper.predict(g_filt[0], g_filt[1], (float)Z_FALLBACK);
                rough.x = std::max(0.f, std::min((float)(FRAME_W - 1), rough.x));
                rough.y = std::max(0.f, std::min((float)(FRAME_H - 1), rough.y));
                double sampled_b = get_depth_from_pixel(rough.x, rough.y, false, false);

                double sampled = -1.0;

                // [MODIFICATION 3] Stabilize dual-probes via Averaging.
                // If both the current pixel probe and the fallback probe return valid data,
                // average them. This naturally smooths edge boundaries without biasing the foreground.
                // ToF 质量门控：主探针需 有效格>=3 且 邻域std<200mm 才可信
                bool a_trust = (sampled_a > 250.0 && sampled_a < 1500.0 && tvalid >= 3 && tstd < 200.0f);
                bool b_ok    = (sampled_b > 250.0 && sampled_b < 1500.0);
                if (a_trust && b_ok)  sampled = 0.5 * (sampled_a + sampled_b);
                else if (a_trust)     sampled = sampled_a;
                else if (b_ok)        sampled = sampled_b;

                // [MODIFICATION 4] Implement a One-Euro Filter for Depth.
                // This replaces the old EMA. The One-Euro filter allows depth to snap
                // instantly during fast saccades (high velocity) but filters heavily
                // during fixation (low velocity), syncing perfectly with the eye vector.
                static OneEuroFilter ef_depth(30.0, 0.5, 0.05);

                static int  z_outlier_count = 0;
                static bool z_established = false;
                if (sampled > 0) {
                    // 时间离群门控：与已建立深度差 >300mm 且非持续 -> 判为 ToF 跳变, 丢弃保持;
                    // 连续 8 帧都偏离才认可(真的是场景变了)。这直接压制"忽大忽小"的落点漂移。
                    if (z_established && std::abs(sampled - depth_running) > 300.0) {
                        z_outlier_count++;
                        if (z_outlier_count >= 8) { depth_running = ef_depth.filter(sampled); z_outlier_count = 0; }
                    } else {
                        depth_running = ef_depth.filter(sampled);
                        z_outlier_count = 0; z_established = true;
                    }
                    bad_probe_count = 0;
                } else {
                    bad_probe_count++;
                    if (bad_probe_count > 30) {
                        depth_running = 0.97 * depth_running + 0.03 * Z_FALLBACK;
                        ef_depth.reset();
                        z_established = false;
                    }
                }
                double Z_used = std::max(300.0, std::min(1200.0, depth_running));

                cv::Point2f ideal = gaze_mapper.predict(g_filt[0], g_filt[1], (float)Z_used);
                // 3) Re-distort the ideal pixel exactly as before.
                std::vector<cv::Point3f> pts3 = { cv::Point3f(
                    (float)((ideal.x - camera_matrix.at<double>(0,2)) / camera_matrix.at<double>(0,0)),
                    (float)((ideal.y - camera_matrix.at<double>(1,2)) / camera_matrix.at<double>(1,1)),
                    1.0f) };
                std::vector<cv::Point2f> distorted;
                cv::projectPoints(pts3, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), camera_matrix, dist_coeffs, distorted);

                // 4) 1-€ filter on the OUTPUT pixel, with a hard jump guard.
                //    A single-frame change of more than JUMP_PX is almost
                //    certainly a stale-frame artefact (eye lost, depth probe
                //    misfire, ToF saturation). In that case, hold the previous
                //    filtered output for one frame instead of letting the
                //    spike propagate through the One-Euro filter.
                float t_u = distorted[0].x;
                float t_v = distorted[0].y;
                const float JUMP_PX = 250.0f;
                static float last_u = t_u, last_v = t_v;
                static bool  have_last = false;
                float du = t_u - last_u, dv = t_v - last_v;
                bool  is_jump = have_last && (std::sqrt(du*du + dv*dv) > JUMP_PX);
                if (!is_jump) {
                    smoothed_u = (float)ef_u.filter(t_u);
                    smoothed_v = (float)ef_v.filter(t_v);
                    last_u = t_u; last_v = t_v;
                    have_last = true;
                }
                // else: keep smoothed_u, smoothed_v at their previous values for one frame

                // Bounds clamp
                smoothed_u = std::max(0.0f, std::min((float)(FRAME_W - 1), smoothed_u));
                smoothed_v = std::max(0.0f, std::min((float)(FRAME_H - 1), smoothed_v));

                // --- 调试开关 k: 强制锚定点为画面正中心, 覆盖上面的眼动计算结果 ---
                // 放在 clamp 之后、诊断打印之前, 使后续绘制/UDP发送/YOLO吸附都统一
                // 使用该固定点, 而不影响 k=false 时的任何现有行为。
                //
                // [重要] 不能直接发送 (FRAME_W/2, FRAME_H/2) = (640, 360)：
                // YOLO8_test.py 的 udp_coord_receiver() 把 (640,360) 当作"未标定
                // 占位坐标"的哨兵值特殊处理(收到时不进入正常的已标定判定分支，
                // 甚至会把 IS_CALIBRATED 复位)，直接导致 YOLO 吸附/拍照识别失效。
                // 用 +1 像素偏移(画面中心附近，肉眼不可见)绕开该哨兵值，同时仍
                // 落在有效边界内 (< FRAME_W-1, FRAME_H-1，与上面的 clamp 一致)。
                if (k) {
                    smoothed_u = FRAME_W / 2.0f + 1.0f;
                    smoothed_v = FRAME_H / 2.0f + 1.0f;
                }

                // === 垂直通道实时诊断 (每 ~0.5s 打印一次) ===
                // 看三段是否同步变化：eyeDY(瞳孔-模型) -> gy -> v。
                //   * eyeDY 变、gy 不变 => compute_gaze_vector 垂直增益被压死
                //   * eyeDY 不变        => 眼图/瞳孔本身没有垂直信号(镜头/装配/阈值)
                //   * gy 变、v 漂移      => mapper 垂直映射问题
                static int vdbg_counter = 0;
                if (++vdbg_counter % 15 == 0) {
                    float py, my, px, mx;
                    {
                        std::lock_guard<std::mutex> lock(eye_dbg_mutex);
                        py = dbg_pupil_y; my = dbg_model_y; px = dbg_pupil_x; mx = dbg_model_x;
                    }
                    //printf("[VDBG] gy=%+.3f eyeDY=%+.1f (pY=%.0f mY=%.0f) | gx=%+.3f | Z=%.0f ToF=%d/9 std=%.0f -> v=%.0f u=%.0f\n",
                          //g_filt[1], py - my, py, my, g_filt[0], Z_used, tvalid, tstd, smoothed_v, smoothed_u);
                }

                // Draw
                cv::circle(frame_world, cv::Point((int)smoothed_u, (int)smoothed_v), 20, cv::Scalar(0,255,0), 3);

                cv::circle(frame_world, cv::Point((int)smoothed_u, (int)smoothed_v), 5,  cv::Scalar(0,0,255), -1);
                char depth_str[64];
                sprintf(depth_str, "Z: %dmm", (int)Z_used);
                cv::putText(frame_world, depth_str,
                            cv::Point((int)smoothed_u + 25, (int)smoothed_v - 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);

                debug_msg = depth_str;   //Debug

                debug_z = Z_used;

            }
        }// 结束 Phase C
        /*
        // ==========================================
        // 🚨 新增：ToF 8x8 深度矩阵可视化网格 (用于物理边界校准)
        // ==========================================
        {
            // 获取当前理论映射的边界（基于你之前代码中的逻辑）
            // 如果你想全屏寻找，可以直接设置 grid_w = FRAME_W / 8
            int grid_rows = 8;
            int grid_cols = 8;

            // 建议：基于我们之前的理论安全框来画，方便你对比“理论”与“实际”的偏移
            int theory_safe_w = (int)(camera_matrix.at<double>(0, 0) * 0.828);
            int theory_safe_h = (int)(camera_matrix.at<double>(1, 1) * 0.828);
            int start_x = (FRAME_W - theory_safe_w) / 2;
            int start_y = (FRAME_H - theory_safe_h) / 2;

            int cell_w = theory_safe_w / grid_cols;
            int cell_h = theory_safe_h / grid_rows;

            std::lock_guard<std::mutex> lock(tof_mutex); // 锁定数据，准备读取

            for (int i = 0; i < grid_rows; i++) {
                for (int j = 0; j < grid_cols; j++) {
                    // 计算当前格子的矩形区域
                    int x1 = start_x + j * cell_w;
                    int y1 = start_y + i * cell_h;
                    cv::Rect cell_rect(x1, y1, cell_w, cell_h);

                    // 获取深度值
                    int depth = latest_tof_matrix[i][j];

                    // 绘制格子边界 (使用半透明绿色或灰色)
                    cv::rectangle(frame_world, cell_rect, cv::Scalar(100, 100, 100), 1);

                    // 如果测到有效距离（比如有人脸或纸板经过），高亮该格子
                    if (depth > 50 && depth < 1000) {
                        // 绘制数值
                        char val_str[10];
                        sprintf(val_str, "%d", depth);
                        cv::putText(frame_world, val_str, cv::Point(x1 + 5, y1 + cell_h/2 + 5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

                        // 给测到近距离物体的格子加一个黄色内框
                        cv::rectangle(frame_world, cell_rect, cv::Scalar(0, 255, 255), 2);
                    }
                }
            }

            // 画一个醒目的中心十字，对齐参考
            cv::line(frame_world, cv::Point(FRAME_W/2-20, FRAME_H/2), cv::Point(FRAME_W/2+20, FRAME_H/2), cv::Scalar(0,0,255), 2);
            cv::line(frame_world, cv::Point(FRAME_W/2, FRAME_H/2-20), cv::Point(FRAME_W/2, FRAME_H/2+20), cv::Scalar(0,0,255), 2);
        }
        // ==========================================
        */
        // ==========================================
        // Send frames over Ethernet
        // ==========================================

        // --- 发送视线坐标给本机的 Python ---
        char coord_buf[64];
        // 发送格式："X,Y" (使用的是原始 1280x720 坐标系)
        snprintf(coord_buf, sizeof(coord_buf), "%d,%d", (int)smoothed_u, (int)smoothed_v);
        send_udp_datagram(udp_socket_coord, coord_addr, coord_buf, strlen(coord_buf), "coord");

        {
            char json_buf[512];
            // 注意这里的双引号和逗号，差一个 Python 都解析不了！
            // %s 对应 string.c_str(), %d 对应整数, %.1f 对应浮点数
            snprintf(json_buf, sizeof(json_buf),
                     "{\"phase\":\"%s\", \"msg\":\"%s\", \"pts\":%d, \"z\":%.1f, \"gy\":%.3f, \"eye_dy\":%.1f, \"tof_valid\":%d, \"tof_std\":%.0f}",
                     debug_phase.c_str(),
                     debug_msg.c_str(),
                     (int)calib_features.size(),
                     debug_z,
                     debug_gy,
                     debug_eyedy,
                     dbg_tof_valid,
                     dbg_tof_std);

            send_debug_info(json_buf);
        }

        send_frame_over_ethernet(udp_socket_world, world_addr, frame_world, 640, 360, 60);

        // 干净世界帧(原生分辨率, 未叠加HUD): 降频发送(约1/3主循环帧率)以控制 JPEG
        // 编码开销——VLM 拍照 ROI 只需"随时可用的较新画面", 不需要 30fps 的实时性。
        // 质量取 40 (而非之前的 70): 1280x720 @ q70 编码后经常超过 UDP 单包
        // 65500 字节上限, 被 send_frame_over_ethernet 静默丢弃, 是短按拍照
        // 一直报"暂无待识别帧"的真正根因。q40 在该分辨率下留有充分余量，
        // 供豆包视觉识别完全够用(无需摄影级画质)。
        static int clean_frame_send_counter = 0;
        if (++clean_frame_send_counter >= 3) {
            clean_frame_send_counter = 0;
            send_frame_over_ethernet(udp_socket_world_clean, world_clean_addr, frame_world_clean, FRAME_W, FRAME_H, 40);
        }

        if (!frame_eye.empty()) {
            send_frame_over_ethernet(udp_socket_eye, eye_addr, frame_eye, 320, 240, 50);
        }
    } // 结束 while(true)

        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "❌ Fatal OpenCV exception: " << e.what() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal exception: " << e.what() << std::endl;
        return 3;
    } catch (...) {
        std::cerr << "❌ Fatal unknown exception." << std::endl;
        return 4;
    }
}
