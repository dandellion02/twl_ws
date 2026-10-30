#ifndef COMPEN_FILTER_H
#define COMPEN_FILTER_H

#include <Arduino.h>
#include <math.h>
#include "Params.h"

class CompenFilter
{
private:
    float alpha;
    float roll;
    float pitch;
    bool initialized;

public:
    CompenFilter(float alpha_ = 0.98f)
        : alpha(alpha_),
          roll(0.0f),
          pitch(0.0f),
          initialized(false)
    {
    }

    inline void update(
        float ax,
        float ay,
        float az,
        float gx,
        float gy)
    {
        // Accelerometer 기반 attitude
        const float acc_roll =
            atan2f(ay, az);

        const float acc_pitch =
            atan2f(
                -ax,
                sqrtf(ay * ay + az * az)
            );

        if (!initialized)
        {
            roll = acc_roll;
            pitch = acc_pitch;
            initialized = true;
            return;
        }

        // Complementary filtering
        roll =
            alpha * (roll + gx * dt)
            + (1.0f - alpha) * acc_roll;

        pitch =
            alpha * (pitch + gy * dt)
            + (1.0f - alpha) * acc_pitch;
    }

    inline float getRoll() const
    {
        return roll;
    }

    inline float getPitch() const
    {
        return pitch;
    }

    inline void getProjectedGravity(
        float &gx,
        float &gy,
        float &gz) const
    {
        const float sin_roll  = sinf(roll);
        const float cos_roll  = cosf(roll);
        const float sin_pitch = sinf(pitch);
        const float cos_pitch = cosf(pitch);

        gx = sin_pitch;
        gy = -sin_roll * cos_pitch;
        gz = -cos_roll * cos_pitch;
    }

    inline void reset()
    {
        roll = 0.0f;
        pitch = 0.0f;
        initialized = false;
    }
};

#endif