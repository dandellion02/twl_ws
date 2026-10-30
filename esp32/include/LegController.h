#ifndef LEG_CONTROLLER_H
#define LEG_CONTROLLER_H

#include <Arduino.h>
#include <ArduinoEigen.h>
#include <ESP32Servo.h>

class LegController
{
private:
    Servo left_leg_servo;
    Servo right_leg_servo;

    const float Left_Leg_Standing_Deg = 110.0f;
    const float Left_Leg_Sitting_Deg = 50.0f;
    const float Right_Leg_Standing_Deg = 70.0f;
    const float Right_Leg_Sitting_Deg = 130.0f;

public:
    LegController() {}
    
    bool attachLegServo(int left_pin, int right_pin)
    {
        left_leg_servo.setPeriodHertz(50); // 50Hz for standard servos
        right_leg_servo.setPeriodHertz(50); // 50Hz for standard servos
        
        left_leg_servo.attach(left_pin, 500, 2500);
        right_leg_servo.attach(right_pin, 500, 2500); 

        return true;
    }

    void sendPoseControl(float left_angle, float right_angle)       //degree
    {
        left_leg_servo.write(left_angle);
        right_leg_servo.write(right_angle);
    }
};


#endif // LEG_CONTROLLER_H  