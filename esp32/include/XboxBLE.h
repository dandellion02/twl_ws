#ifndef  XBOXBLE_H
#define XBOXBLE_H

#include <Arduino.h>
#include <XboxSeriesXControllerESP32_asukiaaa.hpp>

/* =========================================================
 * Xbox Series X Controller BLE
 * ESP32-S3 + Xbox Controller Model 1914
 * ========================================================= */

// Xbox Controller BLE MAC Address
//static constexpr char XBOX_MAC_ADDRESS[] = "F4:6A:D7:60:D5:7F";
static constexpr char XBOX_MAC_ADDRESS[] = "44:16:22:D2:8A:4C";

// Xbox controller object
static XboxSeriesXControllerESP32_asukiaaa::Core xboxController(
    XBOX_MAC_ADDRESS
);


/* =========================================================
 * Initialization
 * ========================================================= */

inline void xboxInit()
{
    Serial.println("[Xbox] Initializing BLE...");

    xboxController.begin();
}


/* =========================================================
 * Update
 *
 * Must be called repeatedly inside loop()
 * ========================================================= */

inline void xboxUpdate()
{
    xboxController.onLoop();
}


/* =========================================================
 * Connection state
 * ========================================================= */

inline bool xboxIsConnected()
{
    return xboxController.isConnected() &&
           !xboxController.isWaitingForFirstNotification();
}


/* =========================================================
 * Joystick normalization
 *
 * Xbox raw range:
 *      0 ~ 65535
 *
 * Converted range:
 *     -1.0 ~ +1.0
 * ========================================================= */

inline float xboxNormalizeJoystick(uint16_t value)
{
    constexpr float CENTER = 32767.5f;

    float normalized =
        (static_cast<float>(value) - CENTER) / CENTER;

    // Deadzone
    constexpr float DEADZONE = 0.05f;

    if (fabsf(normalized) < DEADZONE)
        return 0.0f;

    return constrain(normalized, -1.0f, 1.0f);
}


/* =========================================================
 * Left Stick
 * ========================================================= */

inline float xboxLeftX()
{
    if (!xboxIsConnected())
        return 0.0f;

    return xboxNormalizeJoystick(
        xboxController.xboxNotif.joyLHori
    );
}

inline float xboxLeftY()
{
    if (!xboxIsConnected())
        return 0.0f;

    return -xboxNormalizeJoystick(
        xboxController.xboxNotif.joyLVert
    );
}


/* =========================================================
 * Right Stick
 * ========================================================= */

inline float xboxRightX()
{
    if (!xboxIsConnected())
        return 0.0f;

    return xboxNormalizeJoystick(
        xboxController.xboxNotif.joyRHori
    );
}

inline float xboxRightY()
{
    if (!xboxIsConnected())
        return 0.0f;

    return xboxNormalizeJoystick(
        xboxController.xboxNotif.joyRVert
    );
}

#endif  // XBOXBLE_H