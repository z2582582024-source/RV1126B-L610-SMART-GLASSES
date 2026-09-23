#include "platform.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <stdio.h>

uint8_t VL53L5CX_RdByte(VL53L5CX_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_value) {
    return VL53L5CX_RdMulti(p_platform, RegisterAdress, p_value, 1);
}

uint8_t VL53L5CX_WrByte(VL53L5CX_Platform *p_platform, uint16_t RegisterAdress, uint8_t value) {
    return VL53L5CX_WrMulti(p_platform, RegisterAdress, &value, 1);
}

#define I2C_CHUNK_SIZE 256

uint8_t VL53L5CX_WrMulti(VL53L5CX_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_values, uint32_t size) {
    uint32_t current_size;
    uint32_t offset = 0;

    while (size > 0) {
        // Send a maximum of 256 bytes per transaction
        current_size = (size > I2C_CHUNK_SIZE) ? I2C_CHUNK_SIZE : size;
        uint16_t current_reg = RegisterAdress + offset;

        uint8_t buf[current_size + 2];
        buf[0] = (current_reg >> 8) & 0xFF;
        buf[1] = current_reg & 0xFF;
        memcpy(&buf[2], p_values + offset, current_size);

        struct i2c_msg msg;
        msg.addr = p_platform->address;
        msg.flags = 0;
        msg.len = current_size + 2;
        msg.buf = buf;

        struct i2c_rdwr_ioctl_data rdwr;
        rdwr.msgs = &msg;
        rdwr.nmsgs = 1;

        if (ioctl(p_platform->fd, I2C_RDWR, &rdwr) < 0) {
            printf("\nI2C Write Error at reg 0x%04X, size %d\n", current_reg, current_size);
            return 1;
        }

        size -= current_size;
        offset += current_size;
    }
    return 0;
}

uint8_t VL53L5CX_RdMulti(VL53L5CX_Platform *p_platform, uint16_t RegisterAdress, uint8_t *p_values, uint32_t size) {
    uint32_t current_size;
    uint32_t offset = 0;

    while (size > 0) {
        current_size = (size > I2C_CHUNK_SIZE) ? I2C_CHUNK_SIZE : size;
        uint16_t current_reg = RegisterAdress + offset;

        uint8_t reg[2];
        reg[0] = (current_reg >> 8) & 0xFF;
        reg[1] = current_reg & 0xFF;

        struct i2c_msg msgs[2];
        msgs[0].addr = p_platform->address;
        msgs[0].flags = 0;
        msgs[0].len = 2;
        msgs[0].buf = reg;

        msgs[1].addr = p_platform->address;
        msgs[1].flags = I2C_M_RD;
        msgs[1].len = current_size;
        msgs[1].buf = p_values + offset;

        struct i2c_rdwr_ioctl_data rdwr;
        rdwr.msgs = msgs;
        rdwr.nmsgs = 2;

        if (ioctl(p_platform->fd, I2C_RDWR, &rdwr) < 0) {
             printf("\nI2C Read Error at reg 0x%04X, size %d\n", current_reg, current_size);
             return 1;
        }

        size -= current_size;
        offset += current_size;
    }
    return 0;
}
uint8_t VL53L5CX_Reset_Sensor(VL53L5CX_Platform *p_platform) {
    // If you haven't wired the reset pin to a Linux GPIO, just sleep.
    // If LPn is pulled high hardwired, the firmware will boot.
    usleep(100000); 
    return 0;
}

void VL53L5CX_SwapBuffer(uint8_t *buffer, uint16_t size) {
    uint32_t i, tmp;
    for(i = 0; i < size; i = i + 4) {
        tmp = (buffer[i]<<24) | (buffer[i+1]<<16) | (buffer[i+2]<<8) | (buffer[i+3]);
        memcpy(&(buffer[i]), &tmp, 4);
    }
}

uint8_t VL53L5CX_WaitMs(VL53L5CX_Platform *p_platform, uint32_t TimeMs) {
    usleep(TimeMs * 1000);
    return 0;
}