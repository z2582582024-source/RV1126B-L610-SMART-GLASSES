#ifndef TOF_TEST_H
#define TOF_TEST_H

#include <stdint.h>

// Define the packet structure for serial output
#pragma pack(push, 1)
typedef struct {
    uint16_t header;       // 0xA55A
    uint32_t timestamp_ms;
    uint16_t depth_data[64];
    uint16_t checksum;
} TOF_Packet_t;
#pragma pack(pop)

void TOF_Test_Init(void);
void TOF_Test_Loop(void);

#endif // TOF_TEST_H