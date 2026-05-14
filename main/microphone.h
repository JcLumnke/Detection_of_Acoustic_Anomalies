#ifndef MICROPHONE_H_
#define MICROPHONE_H_

#include <stdint.h>

void microphone_init();

bool microphone_read(float* buffer, int samples);

#endif