#include "tof_test.h"
#include "vl53l5cx_api.h"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

static TOF_Packet_t g_tof_packet;
VL53L5CX_Configuration Dev;

// Equivalent to HAL_GetTick()
uint32_t get_tick_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void TOF_Test_Init(void) {
    uint8_t status, isAlive;
    
    // Linux uses 7-bit addresses (0x52 >> 1 = 0x29)
    Dev.platform.address = 0x29; 

    Dev.platform.fd = open("/dev/i2c-4", O_RDWR);
    if (Dev.platform.fd < 0) {
        printf("Error: Failed to open /dev/i2c-4\n");
        return;
    }

    printf("[1] Resetting Sensor...\n");
    VL53L5CX_Reset_Sensor(&Dev.platform);

    printf("[2] Checking Alive...\n");
    vl53l5cx_is_alive(&Dev, &isAlive);
    if (!isAlive) {
        printf("Error: Sensor Not Found!\n");
        return;
    }

    printf("[3] Loading FW (Wait 3s)...\n");
    status = vl53l5cx_init(&Dev);
    if (status != VL53L5CX_STATUS_OK) {
        printf("Error: FW Load Fail!\n");
        return;
    }
    printf("[4] Init Success!\n");

    vl53l5cx_set_resolution(&Dev, VL53L5CX_RESOLUTION_8X8);
    vl53l5cx_set_ranging_frequency_hz(&Dev, 15);

    status = vl53l5cx_start_ranging(&Dev);
    if (status == VL53L5CX_STATUS_OK) {
        printf("[5] Ranging 8x8 @ 15Hz Started!\n");
    }
}

void TOF_Test_Loop(void) {
    uint8_t isDataReady = 0;
    VL53L5CX_ResultsData Results;

    // Check if data is ready
    vl53l5cx_check_data_ready(&Dev, &isDataReady);

    if (isDataReady) {
        vl53l5cx_get_ranging_data(&Dev, &Results);

        g_tof_packet.header = 0xA55A;
        g_tof_packet.timestamp_ms = get_tick_ms();
        g_tof_packet.checksum = 0;

        for (int i = 0; i < 64; i++) {
            uint8_t status = Results.target_status[VL53L5CX_NB_TARGET_PER_ZONE * i];
            
            // Replicating your STM32 reliability logic
            if (status == 5 || status == 9 || status == 6 || status == 10) {
                g_tof_packet.depth_data[i] = Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE * i];
            } else {
                g_tof_packet.depth_data[i] = 2000; 
            }
        }

        // Use low-level write to bypass buffering issues and prevent freezes.
        // This outputs the exact same binary packet as HAL_UART_Transmit
        write(STDOUT_FILENO, &g_tof_packet, sizeof(g_tof_packet));
        
        // Critical: Yield CPU after a successful read to maintain system stability
        usleep(10000); 
    } else {
        // Critical: If data is not ready, sleep for 5ms to prevent the while(1) 
        // loop from locking up the I2C driver and pegging the CPU to 100%
        usleep(5000); 
    }
}

int main() {
    TOF_Test_Init();
    while(1) {
        TOF_Test_Loop();
    }
    return 0;
}