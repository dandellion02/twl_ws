#include "Params.h"        // 모든 물리 상수가 정의된 헤더
//#include "Bluetooth.h"     // 블루투스 커맨드 수신 클래스 (Receiver 대체)
#include "HRController.h"  // 고관절(Hip) 서보 제어기
#include "LegController.h" // 다리 제어기
#include "VYBController.h" // 바퀴(Wheel) 밸런싱 제어기
#include "MGServo.h"       // RS485 통신 기반 휠 모터 드라이버
#include "IMU.h"           // MPU6050 센서 데이터 처리
#include "POL.h"           // 로봇의 물리 모델(Physics of Locomotion)
#include "EKF.h"           // 확장 칼만 필터(상태 추정기)
#include "CompenFilter.h"   // 보상 필터(Compensation Filter)
#include "Logger.h"        // WiFi 기반 데이터 로거
#include "Timer.h"         // 샘플링 주기를 위한 정밀 타이머
#include "rl_model.h"      // PPO 강화학습 모델
#include "policy_network.h" // PPO 강화학습 모델 헤더
#include "XboxBLE.h"      // Xbox Controller BLE 통신 헤더

// --- 객체 생성 및 물리 설정 ---
const Properties properties = createDefaultProperties(); // 로봇의 질량, 길이 등 기본 파라미터
POL Pol(properties);                                     // 물리 모델에 파라미터 주입
HardwareSerial RS485(1);                                 // RS485 통신을 위한 시리얼 포트 설정
MGServo ServoLW(1, RS485);                               // 1번 모터: 왼쪽 바퀴
MGServo ServoRW(2, RS485);                               // 2번 모터: 오른쪽 바퀴
//WBRBluetooth ble;                                        // 블루투스 수신용 객체 (기존 Receiver 대체)
IMU MPU6050;                                             // 자이로/가속도 센서 객체
CompenFilter compenFilter;                                 // 보상 필터 객체

HRController HR_controller;                     // 고관절 제어 인스턴스
LegController Leg_controller;                   // 다리 제어 인스턴스
VYBController VYB_controller(ServoRW, ServoLW); // 바퀴 제어 인스턴스
Logger WIFI_Logger(ssid, password);             // 데이터 전송용 와이파이 설정
EKF Estimator(Pol);                             // 모델 기반 상태 추정기
PolicyNetwork policy;                           // 강화학습 기반 정책 신경망


// --- 행렬 선언 (Eigen 라이브러리 활용) ---

/* 현재 상태 벡터 x = [θ, θ̇ , v, ψ̇ ]
θ, θ̇ : IMU (MPU6050) 기반 추정된 Pitch 각도/각속도
v    : 좌우 바퀴 속도 합
ψ̇    : 좌우 바퀴 속도 차 */
Eigen::Matrix<float, 4, 1> x = Eigen::Matrix<float, 4, 1>::Zero();
// 목표 상태 벡터 x_d
Eigen::Matrix<float, 4, 1> x_d = Eigen::Matrix<float, 4, 1>::Zero();
// 시간 지연 보상을 위한 이전 토크 u_prev
Eigen::Matrix<float, 2, 1> u_prev = Eigen::Matrix<float, 2, 1>::Zero();
// 현재 토크 u 입력
Eigen::Matrix<float, 2, 1> u = Eigen::Matrix<float, 2, 1>::Zero();
// 센서 측정값 벡터 z (가속도, 각속도 등 8개 항목)
Eigen::Matrix<float, 8, 1> z = Eigen::Matrix<float, 8, 1>::Zero();
// 모터의 실제 전류 피드백값
Eigen::Matrix<float, 2, 1> iq_vec = Eigen::Matrix<float, 2, 1>::Zero();

int i = 0;
Timer log_timer(Timer::TimerType::Millis);
Timer sampling_timer(Timer::TimerType::Millis);
Timer temp_timer(Timer::TimerType::Millis);
float h_d = HEIGHT_MAX, phi_d = 0;
float v_d = 0, dpsi_d = 0;

// Policy Network 입출력 차원 수에 맞춰 버퍼 할당
float obs_buffer[10];
float action_buffer[2] = {0.0f, 0.0f}; 

void serialPrintStates();
void Ready2Log();
void LogData();

void motorResponseTest();
void motorFrictionTest();
void testSpeedStep(MGServo& motor);

// ==============================================================================
//                                    DEBUG
// ==============================================================================
// Off 모드에서 Height 반전문제 디버깅
void sim_off_mode_gain();
void dbgHeight();

// ==============================================================================
//                                    SETUP
// ==============================================================================
void setup()
{
  // Serial 통신, Bluetooth 초기화
  Serial.begin(115200);
  delay(1000);                                                        // 디버깅용 PC 시리얼 통신 시작

  /*-------Servo motor setup-------*/
  if (!Leg_controller.attachLegServo(LH_PIN, RH_PIN))
  {
    Serial.println("[ERROR] Fail to attach Leg Servos.");
    while (1)
    {
      delay(1000);
    }
  }
  Serial.println("[2] Leg Servo OK");
  Leg_controller.sendPoseControl(110.0f, 70.0f); // 초기 자세: 양쪽 다리 0도

  //ble.begin("ESP32_WBR_Control", &ServoLW, &ServoRW, &HR_controller); // 블루투스 수신 대기 (기존 receiver.begin)

  // IMU (MPU6050) 초기화
  if (!MPU6050.begin())
  {
    Serial.println("[ERROR] Fail to initialize IMU.");
    while (1)
    {
      delay(1000);
    } // 초기화 실패 시 무한 루프
  }

  // RS485 통신 초기화
  pinMode(RS485_DE_RE, OUTPUT);   // RS485 송수신 방향 제어 핀 설정
  digitalWrite(RS485_DE_RE, LOW); // 기본 상태를 수신(Read) 모드로 설정
  
  // 모터와 통신할 고속 라인 개통
  RS485.begin(460800, SERIAL_8N1, RS485_RX_PIN, RS485_TX_PIN);

  // WIFI 연결 (CSV 로깅 서비스 시작)
  //WIFI_Logger.begin();

  // PSRAM 상태 확인 및 초기화
  // --- ESP32 PSRAM(외부 메모리) 체크 ---
  if (psramFound())
  { // 대용량 로깅을 위해 PSRAM이 있는지 확인
    if (!psramInit())
    {
      while (1)
      {
        Serial.println("[ERROR] PSRAM Fail!");
      } // 메모리 초기화 실패 시 멈춤
    }
  }
  else
  {
    while (1)
    {
      Serial.println("PSRAM not available.");
    } // PSRAM 없으면 구동 불가
  }


  if (!policy.begin(policy_float32_tflite)) 
  {
    Serial.println("[ERROR] AI Model Loading Failed...");
    while (1) // 실패 시 무한 루프 대기
      delay(1000);
  }
  
  //xboxInit(); // Xbox Controller BLE 초기화

  // CSV 컬럼 (모니터링 타겟) 리스트 준비
  //Ready2Log();

  // 시간 측정 시작
  //log_timer.start();
  sampling_timer.start();
}

// ==============================================================================
//                                    LOOP
// ==============================================================================

void loop()
{
  //testSpeedStep(ServoLW);
  //testSpeedStep(ServoRW);
  //motorFrictionTest();
#if 1
  // ---  주기 제어 (Sampling Time Check) ---
  // dt(0.008) * 1000 = 8ms가 경과했는지 확인합니다.
  if (sampling_timer.getDuration() >= dt * 1000)
  {
    sampling_timer.start(); // sampling timer 초기화

    //static unsigned long start_time = 0;
    //static unsigned long elapsed_time = 0;
    //start_time = micros();

    
    /*------------------observation------------------*/

    // 3축 가속도(m/s²) & 각속도(rad/s) 저장
    if (MPU6050.readData())
    {
      // body angular velocity
      obs_buffer[0] = MPU6050.gyr_vec[0];
      obs_buffer[1] = MPU6050.gyr_vec[1];
      obs_buffer[2] = MPU6050.gyr_vec[2];
    
      // attitude update
      compenFilter.update(
          MPU6050.acc_vec[0],
          MPU6050.acc_vec[1],
          MPU6050.acc_vec[2],
          MPU6050.gyr_vec[0],
          MPU6050.gyr_vec[1]
      );
      
      // projected gravity를 obs_buffer에 직접 기록
      compenFilter.getProjectedGravity(
          obs_buffer[3],
          obs_buffer[4],
          obs_buffer[5]
      );
    }

    // 휠 모터 각속도(rad/s) 저장
    if(ServoLW.sendCommandReadMotorState2() && ServoRW.sendCommandReadMotorState2())
    {
      obs_buffer[6] = ServoLW.getMotorSpeed() * DEG2RAD;  // rad/s 단위로 변환
      obs_buffer[7] = -ServoRW.getMotorSpeed() * DEG2RAD;  // rad/s 단위로 변환
    }

    //직전 action 값 저장
    //obs_buffer[8] = action_buffer[0];
    //obs_buffer[9] = action_buffer[1];

    //Xbox controller command 저장
#if 0
    static unsigned long start_time = 0;
    static unsigned long elapsed_time = 0;
    start_time = micros();
    xboxUpdate();
    elapsed_time = micros() - start_time;
    if(xboxIsConnected())
    {
      Serial.printf(
        "LX: %.2f | LY: %.2f | RX: %.2f | RY: %.2f | Time: %lu us\n",
        xboxLeftX(),
        xboxLeftY(),
        xboxRightX(),
        xboxRightY(),
        elapsed_time
      );
    }
#endif
    /*------------------model reasoning------------------*/
    //static unsigned long start_time = 0;
    //static unsigned long elapsed_time = 0;
    //start_time = micros();

    // 모델 추론
    if (policy.predict(obs_buffer, action_buffer)) 
    {
      //start_time = micros();

      static float action_filtered[2] = {0.0f, 0.0f};

      static const float alpha = 0.925f;
      static const float beta = 150.0f;
#if 1
      action_filtered[0] =
          alpha * action_filtered[0]
          + (1.0f - alpha) * action_buffer[0];

      action_filtered[1] =
          alpha * action_filtered[1]
          + (1.0f - alpha) * action_buffer[1];
#endif          
      //elapsed_time = micros() - start_time;
      // 결과 출력 및 모터 제어 적용
      //ServoLW.sendTorqueControlCommand(action_buffer[0] * 1.351f);
      //ServoRW.sendTorqueControlCommand(-action_buffer[1] * 1.351f);
      ServoLW.sendSpeedControlCommand(action_filtered[0] * beta);
      ServoRW.sendSpeedControlCommand(-action_filtered[1] * beta);
      //elapsed_time = micros() - start_time;
      obs_buffer[8] = action_buffer[0];
      obs_buffer[9] = action_buffer[1];
      
      //Serial.print("Action Outputs: ");
      //for(int i = 0; i < policy.getNumOutputs(); i++) {
      //    Serial.printf("%.4f ", action_buffer[i]/2.5f);
      //}
      //Serial.println();
      
      //Serial.printf(" | Inference Time: %lu us (%.2f ms)\n", elapsed_time, elapsed_time / 1000.0f);
    }
#if 0    
    Serial.printf(
    "OBS: "
    "gyro=[%.4f %.4f %.4f] "
    "acc=[%.4f %.4f %.4f] "
    "wheel=[%.4f %.4f] "
    "prev=[%.4f %.4f]\n",
    obs_buffer[0],
    obs_buffer[1],
    obs_buffer[2],
    obs_buffer[3],
    obs_buffer[4],
    obs_buffer[5],
    obs_buffer[6],
    obs_buffer[7],
    obs_buffer[8],
    obs_buffer[9]
    );
#endif    
  }
#endif
}

void testSpeedStep(MGServo& motor)
{
    const float testSpeeds[] = {5.0f, 10.0f, 15.0f, 20.0f};
    const int numTests = sizeof(testSpeeds) / sizeof(testSpeeds[0]);

    const unsigned long testDurationUs = 1000000; // 1 s
    const unsigned long zeroDurationUs = 500000;  // 0.5 s

    Serial.println("time_us,target_rad_s,measured_rad_s,iq_A");

    for (int i = 0; i < numTests; i++)
    {
        float target = testSpeeds[i];

        unsigned long startTime = micros();

        while ((micros() - startTime) < testDurationUs)
        {
            unsigned long elapsed = micros() - startTime;

            if (motor.sendSpeedControlCommand(target))
            {
                float measured_rad_s =
                    motor.getMotorSpeed() * DEG_TO_RAD;

                Serial.print(elapsed);
                Serial.print(",");
                Serial.print(target, 4);
                Serial.print(",");
                Serial.print(measured_rad_s, 4);
                Serial.print(",");
                Serial.println(motor.getMotorIq(), 4);
            }
        }

        // 0 rad/s 구간
        unsigned long zeroStart = micros();

        while ((micros() - zeroStart) < zeroDurationUs)
        {
            motor.sendSpeedControlCommand(0.0f);
        }
    }

    motor.sendSpeedControlCommand(0.0f);

    Serial.println("Speed step test finished.");
}

void motorFrictionTest()
{
  // ============================================================
  // Test configuration
  // ============================================================

  constexpr float DRIVE_TORQUE = 0.30f * 1.35f;              // [Nm]

  constexpr unsigned long PRE_DELAY_US   = 500000;   // 0.5 s
  constexpr unsigned long DRIVE_TIME_US  = 2000000;  // 2.0 s
  constexpr unsigned long COAST_TIME_US  = 3000000;  // 3.0 s

  // 약 0.62 ms/read이므로 3초를 전부 고속 저장하면 너무 많음.
  // 따라서 약 8 ms 간격으로 기록.
  constexpr unsigned long SAMPLE_PERIOD_US = 8000;

  constexpr int MAX_SAMPLES = 700;


  // ============================================================
  // Test state
  // ============================================================

  enum TestState
  {
    TEST_INIT,
    TEST_WAIT,
    TEST_DRIVE,
    TEST_COAST,
    TEST_FINISHED,
    TEST_PRINTED
  };

  static TestState state = TEST_INIT;

  static unsigned long state_start_time = 0;
  static unsigned long test_start_time = 0;
  static unsigned long last_sample_time = 0;

  static int sample_count = 0;


  // ============================================================
  // RAM buffer
  // ============================================================

  static unsigned long time_buffer[MAX_SAMPLES];

  static float command_buffer[MAX_SAMPLES];
  static float iq_buffer[MAX_SAMPLES];
  static float torque_buffer[MAX_SAMPLES];
  static float speed_buffer[MAX_SAMPLES];

  static unsigned long response_time_buffer[MAX_SAMPLES];


  // ============================================================
  // Helper: Motor state sampling
  // ============================================================

  auto sampleMotor = [&](float command_torque)
  {
    if (sample_count >= MAX_SAMPLES)
      return;

    unsigned long now = micros();

    if ((now - last_sample_time) < SAMPLE_PERIOD_US)
      return;

    last_sample_time = now;


    unsigned long read_start = micros();

    bool success =
        ServoRW.sendCommandReadMotorState2();

    unsigned long read_end = micros();


    if (!success)
      return;


    float iq =
        ServoRW.getMotorIq();

    float estimated_torque =
        iq * 0.7f;

    float speed_rad_s =
        ServoRW.getMotorSpeed() * DEG2RAD;


    time_buffer[sample_count] =
        read_end - test_start_time;

    command_buffer[sample_count] =
        command_torque;

    iq_buffer[sample_count] =
        iq;

    torque_buffer[sample_count] =
        estimated_torque;

    speed_buffer[sample_count] =
        speed_rad_s;

    response_time_buffer[sample_count] =
        read_end - read_start;

    sample_count++;
  };


  // ============================================================
  // 1. Initialization
  // ============================================================

  if (state == TEST_INIT)
  {
    Serial.println();
    Serial.println("==========================================");
    Serial.println(" Motor Friction / Coast-down Test");
    Serial.println("==========================================");

    Serial.printf(
      "Drive torque : %.3f Nm\n",
      DRIVE_TORQUE
    );


    sample_count = 0;


    // 시작 전에 0 Nm
    ServoRW.sendTorqueControlCommand(0.0f);


    state_start_time = micros();

    state = TEST_WAIT;

    return;
  }


  // ============================================================
  // 2. 정지 상태 대기
  // ============================================================

  if (state == TEST_WAIT)
  {
    if ((micros() - state_start_time) < PRE_DELAY_US)
      return;


    // ==========================================================
    // 테스트 시간 기준점
    // ==========================================================

    test_start_time = micros();
    last_sample_time = test_start_time;


    // ==========================================================
    // 일정 torque command 인가
    // ==========================================================

    bool success =
        ServoRW.sendTorqueControlCommand(DRIVE_TORQUE);


    if (!success)
    {
      Serial.println("[ERROR] Drive torque command failed.");
      return;
    }


    state_start_time = micros();

    state = TEST_DRIVE;

    return;
  }


  // ============================================================
  // 3. DRIVE phase
  //
  // +0.1 Nm를 2초 동안 유지
  // ============================================================

  if (state == TEST_DRIVE)
  {
    sampleMotor(DRIVE_TORQUE);


    if ((micros() - state_start_time) >= DRIVE_TIME_US)
    {
      // --------------------------------------------------------
      // Torque 제거
      // --------------------------------------------------------

      ServoRW.sendTorqueControlCommand(0.0f);


      state_start_time = micros();

      state = TEST_COAST;

      return;
    }


    return;
  }


  // ============================================================
  // 4. COAST phase
  //
  // 0 Nm 상태에서 자연 감속 기록
  // ============================================================

  if (state == TEST_COAST)
  {
    sampleMotor(0.0f);


    if (
        (micros() - state_start_time) >= COAST_TIME_US ||
        sample_count >= MAX_SAMPLES
       )
    {
      // 안전상 다시 0 Nm
      ServoRW.sendTorqueControlCommand(0.0f);


      state = TEST_FINISHED;

      return;
    }


    return;
  }


  // ============================================================
  // 5. CSV 출력
  // ============================================================

  if (state == TEST_FINISHED)
  {
    Serial.println();
    Serial.println("# TEST FINISHED");

    Serial.printf(
      "# Sample count = %d\n",
      sample_count
    );


    Serial.println(
      "time_us,"
      "cmd_torque_Nm,"
      "iq_A,"
      "est_torque_Nm,"
      "speed_rad_s,"
      "response_us"
    );


    for (int i = 0; i < sample_count; i++)
    {
      Serial.printf(
        "%lu,"
        "%.4f,"
        "%.6f,"
        "%.6f,"
        "%.6f,"
        "%lu\n",

        time_buffer[i],
        command_buffer[i],
        iq_buffer[i],
        torque_buffer[i],
        speed_buffer[i],
        response_time_buffer[i]
      );
    }


    Serial.println("# END");


    state = TEST_PRINTED;

    return;
  }


  // ============================================================
  // 완료 후 아무것도 하지 않음
  // ============================================================

  if (state == TEST_PRINTED)
  {
    return;
  }
}

void motorResponseTest()
{
  // ============================================================
  // Test configuration
  // ============================================================

  constexpr float TEST_TORQUE = -0.3f;              // [Nm]
  constexpr unsigned long PRE_DELAY_US = 500000;   // 0.5 s
  constexpr unsigned long RECORD_TIME_US = 30000;  // 30 ms

  constexpr int MAX_SAMPLES = 64;


  // ============================================================
  // State
  // ============================================================

  static bool initialized = false;
  static bool step_sent = false;
  static bool recording_finished = false;
  static bool printed = false;

  static unsigned long init_time = 0;
  static unsigned long step_time = 0;

  static int sample_count = 0;


  // ============================================================
  // RAM logging buffer
  // ============================================================

  static unsigned long time_buffer[MAX_SAMPLES];

  static float iq_buffer[MAX_SAMPLES];
  static float torque_buffer[MAX_SAMPLES];
  static float speed_buffer[MAX_SAMPLES];

  static unsigned long read_time_buffer[MAX_SAMPLES];


  // ============================================================
  // 1. 최초 초기화
  // ============================================================

  if (!initialized)
  {
    initialized = true;

    sample_count = 0;

    Serial.println();
    Serial.println("==========================================");
    Serial.println(" Motor Current Step Response Test");
    Serial.println("==========================================");

    Serial.printf(
      "Test torque : %.3f Nm\n",
      TEST_TORQUE
    );

    Serial.println("Waiting before step...");


    // 처음에는 토크 0
    ServoRW.sendTorqueControlCommand(0.0f);

    init_time = micros();

    return;
  }


  // ============================================================
  // 2. Step 입력 전 대기
  // ============================================================

  if (!step_sent)
  {
    if ((micros() - init_time) < PRE_DELAY_US)
      return;


    // ----------------------------------------------------------
    // STEP 명령 시작 시간
    // ----------------------------------------------------------

    unsigned long command_start =
        micros();


    // ----------------------------------------------------------
    // +0.3 Nm step을 딱 한 번 전송
    // ----------------------------------------------------------

    bool success =
        ServoRW.sendTorqueControlCommand(TEST_TORQUE);


    unsigned long command_end =
        micros();


    if (!success)
    {
      Serial.println(
        "[ERROR] Step torque command failed."
      );

      return;
    }


    // ==========================================================
    // command_start를 step 기준 t = 0 으로 사용
    // ==========================================================

    step_time = command_start;


    // ==========================================================
    // 중요:
    //
    // sendTorqueControlCommand() 응답 자체에 포함된
    // Iq / Speed를 첫 번째 sample로 저장
    // ==========================================================

    if (sample_count < MAX_SAMPLES)
    {
      time_buffer[sample_count] =
          command_end - step_time;


      float iq =
          ServoRW.getMotorIq();


      iq_buffer[sample_count] =
          iq;


      torque_buffer[sample_count] =
          iq * 0.7f;


      speed_buffer[sample_count] =
          ServoRW.getMotorSpeed() * DEG2RAD;


      // 첫 sample의 response time은
      // torque command 전체 round-trip 시간
      read_time_buffer[sample_count] =
          command_end - command_start;


      sample_count++;
    }


    step_sent = true;

    return;
  }


  // ============================================================
  // 3. Step 이후 Motor State 고속 Polling
  // ============================================================

  if (!recording_finished)
  {
    unsigned long now =
        micros();


    unsigned long elapsed =
        now - step_time;


    // ----------------------------------------------------------
    // 종료 조건
    // ----------------------------------------------------------

    if (
        elapsed >= RECORD_TIME_US ||
        sample_count >= MAX_SAMPLES
       )
    {
      recording_finished = true;


      // 테스트 끝난 후 토크 제거
      ServoRW.sendTorqueControlCommand(0.0f);

      return;
    }


    // ==========================================================
    // Motor State Read
    // ==========================================================

    unsigned long read_start =
        micros();


    bool success =
        ServoRW.sendCommandReadMotorState2();


    unsigned long read_end =
        micros();


    // ==========================================================
    // 응답 성공 시 저장
    // ==========================================================

    if (success)
    {
      unsigned long sample_time =
          read_end - step_time;


      float iq =
          ServoRW.getMotorIq();


      float estimated_torque =
          iq * 0.7f;


      float speed_rad_s =
          ServoRW.getMotorSpeed() * DEG2RAD;


      // --------------------------------------------------------
      // RAM 저장
      // --------------------------------------------------------

      time_buffer[sample_count] =
          sample_time;


      iq_buffer[sample_count] =
          iq;


      torque_buffer[sample_count] =
          estimated_torque;


      speed_buffer[sample_count] =
          speed_rad_s;


      read_time_buffer[sample_count] =
          read_end - read_start;


      sample_count++;
    }


    // 측정 중에는 Serial 출력 X
    return;
  }


  // ============================================================
  // 4. 측정 완료 후 CSV 출력
  // ============================================================

  if (!printed)
  {
    printed = true;


    Serial.println();
    Serial.println("# TEST FINISHED");


    Serial.printf(
      "# Sample count = %d\n",
      sample_count
    );


    Serial.println(
      "time_us,"
      "iq_A,"
      "est_torque_Nm,"
      "speed_rad_s,"
      "response_us"
    );


    for (int i = 0; i < sample_count; i++)
    {
      Serial.printf(
        "%lu,"
        "%.6f,"
        "%.6f,"
        "%.6f,"
        "%lu\n",

        time_buffer[i],
        iq_buffer[i],
        torque_buffer[i],
        speed_buffer[i],
        read_time_buffer[i]
      );
    }


    Serial.println("# END");

    return;
  }
}


void serialPrintStates()
{
  Serial.print("theta:");
  Serial.print(x(0) * 180 / M_PI, 6);
  Serial.print(" ");
  Serial.print("theta_dot:");
  Serial.print(x(1) * 180 / M_PI, 6);
  Serial.print(" ");
  Serial.print("v:");
  Serial.print(x(2), 6);
  Serial.print(" ");
  Serial.print("psi_dot:");
  Serial.print(x(3) * 180 / M_PI, 6);
  Serial.print(" ");
  Serial.print("u_RW:");
  Serial.print(u(0), 6);
  Serial.print(" ");
  Serial.print("u_LW:");
  Serial.print(u(1), 6);
  Serial.print(" ");
}

void Ready2Log()
{
  WIFI_Logger.readyToLogValue("cal_time");
  WIFI_Logger.readyToLogTimeStamp(); // 시간 기록
  WIFI_Logger.readyToLogValue("h_d");
  WIFI_Logger.readyToLogValue("theta_d");
  WIFI_Logger.readyToLogValue("v_d");
  WIFI_Logger.readyToLogValue("psi_dot_d");
  WIFI_Logger.readyToLogValue("theta_hat");
  WIFI_Logger.readyToLogValue("theta_dot_hat");
  WIFI_Logger.readyToLogValue("v_hat");
  WIFI_Logger.readyToLogValue("psi_dot_hat");
  WIFI_Logger.readyToLogValue("tau_RW");
  WIFI_Logger.readyToLogValue("tau_LW");
  WIFI_Logger.readyToLogValue("acc_x");
  WIFI_Logger.readyToLogValue("acc_y");
  WIFI_Logger.readyToLogValue("acc_z");
  WIFI_Logger.readyToLogValue("gyr_x");
  WIFI_Logger.readyToLogValue("gyr_y");
  WIFI_Logger.readyToLogValue("gyr_z");
  WIFI_Logger.readyToLogValue("theta_dot_RW");
  WIFI_Logger.readyToLogValue("theta_dot_LW");
  WIFI_Logger.readyToLogValue("current_RW");
  WIFI_Logger.readyToLogValue("current_LW");
  WIFI_Logger.readyToLogValue("deg_R");
  WIFI_Logger.readyToLogValue("deg_L");
  WIFI_Logger.readyToLogValue("angle_RH");
  WIFI_Logger.readyToLogValue("angle_LH");
  WIFI_Logger.readyToLogValue("u_P");
  WIFI_Logger.readyToLogValue("u_D");
  WIFI_Logger.readyToLogValue("u_V");
  WIFI_Logger.readyToLogValue("u_Y");
}

void LogData()
{
  // hip 각도 계산 결과
  Eigen::Vector2f hips = Pol.get_theta_hips();

  // 클램핑 전 raw 값 계산
  float deg_R = hips(0) * 180.0f / M_PI;
  float deg_L = hips(1) * 180.0f / M_PI;
  float angle_RH = 90 - (int)(deg_R * 90.0f / 135.0f);
  float angle_LH = 90 - (int)(deg_L * 90.0f / 135.0f);

  WIFI_Logger.logValue("cal_time", sampling_timer.getDuration());
  WIFI_Logger.logTimeStamp(log_timer.getDuration());
  WIFI_Logger.logValue("h_d", h_d);
  WIFI_Logger.logValue("theta_d", x_d(0));
  WIFI_Logger.logValue("v_d", x_d(2));
  WIFI_Logger.logValue("psi_dot_d", x_d(3));
  WIFI_Logger.logValue("theta_hat", x(0));
  WIFI_Logger.logValue("theta_dot_hat", x(1));
  WIFI_Logger.logValue("v_hat", x(2));
  WIFI_Logger.logValue("psi_dot_hat", x(3));
  WIFI_Logger.logValue("tau_RW", u(0));
  WIFI_Logger.logValue("tau_LW", u(1));
  WIFI_Logger.logValue("acc_x", z(0));
  WIFI_Logger.logValue("acc_y", z(1));
  WIFI_Logger.logValue("acc_z", z(2));
  WIFI_Logger.logValue("gyr_x", z(3));
  WIFI_Logger.logValue("gyr_y", z(4));
  WIFI_Logger.logValue("gyr_z", z(5));
  WIFI_Logger.logValue("theta_dot_RW", z(6));
  WIFI_Logger.logValue("theta_dot_LW", z(7));
  WIFI_Logger.logValue("current_RW", iq_vec(0));
  WIFI_Logger.logValue("current_LW", iq_vec(1));
  WIFI_Logger.logValue("deg_R", deg_R);
  WIFI_Logger.logValue("deg_L", deg_L);
  WIFI_Logger.logValue("angle_RH", angle_RH);
  WIFI_Logger.logValue("angle_LH", angle_LH);
  WIFI_Logger.logValue("u_P", VYB_controller.u_P);
  WIFI_Logger.logValue("u_D", VYB_controller.u_D);
  WIFI_Logger.logValue("u_V", VYB_controller.u_V);
  WIFI_Logger.logValue("u_Y", VYB_controller.u_Y);
}


void sim_off_mode_ekf_check()
/* OFF 모드에서 EKF의 개입(왜곡) 여부 시리얼 관찰
1. RAW 데이터(가속도, 자이로, 엔코더)가 올바른 부호를 내는지 확인
2. EKF를 거친 결과(x)가 RAW 데이터의 경향성(부호)을 그대로 잘 따라가는지 확인
*/
{
  static unsigned long last = 0;
  if (millis() - last > 500) 
  {
    last = millis();

    // ----------------------------------------------------
    // [ Before EKF ] : 센서 원시 데이터 (z 벡터) 기반 임시 계산
    // ----------------------------------------------------
    // 1. 가속도계 기반 Pitch 추정
    // z[8]: { acc[3], gyro[3], V_right, V_left };
    // OFF 모드여도, 손으로 직접 밀면서 (V_right, V_left != 0) 을 만들거니까 의미가 있음
    float raw_theta = atan2(-z(0), z(2)); 

    // 2. 자이로스코프 Y축 (Pitch Rate)
    float raw_theta_dot = z(4); 

    // 3. 엔코더 기반 속도 추정
    // 모터 속도 = 로봇 회전 각속도 - (로봇 전진 속도 / 바퀴 반지름)
    // 따라서 v_raw = R * (raw_theta_dot - 평균 모터 회전속도)
    float avg_motor_speed = (z(6) + z(7)) / 2.0f;
    float raw_v = Pol.R * (raw_theta_dot - avg_motor_speed);

    // ----------------------------------------------------
    // [ 시리얼 출력 ] : Before vs After 정면 비교
    // ----------------------------------------------------
    Serial.println("========================================");
    
    // 1. 각도 비교 (가속도계 vs EKF)
    Serial.println("[ 1. 각도 (Theta) ]");
    Serial.printf("RAW(가속도): %+.2f deg | EKF: %+.2f deg\n", 
                  raw_theta * 180 / M_PI, x(0) * 180 / M_PI);
    
    // 2. 각속도 비교 (자이로 vs EKF)
    Serial.println("[ 2. 각속도 (Theta_dot) ]");
    Serial.printf("RAW(자이로): %+.2f deg/s | EKF: %+.2f deg/s\n", 
                  raw_theta_dot * 180 / M_PI, x(1) * 180 / M_PI);

    // 3. 속도 비교 (엔코더 vs EKF)
    Serial.println("[ 3. 전진 속도 (Velocity) ]");
    Serial.printf("RAW(엔코더): %+.3f m/s | EKF: %+.3f m/s\n", 
                  raw_v, x(2));
                  
    Serial.println("========================================");
  }
}


void sim_off_mode_gain()
/* OFF 모드에서 시리얼 관찰
Step 1: 입력단(센서)의 '앞' 통일하기 
1) 각도/각속도 확인: 로봇의 상체를 손으로 잡고 앞으로 기울입니다.
- Pass 조건: theta 와 theta_dot 이 모두 양수(+)가 찍혀야 합니다.
2) 엔코더 속도 확인: 상체는 세워둔 채, 바퀴를 바닥에 대고 앞으로 죽 밀어줍니다. 
- Pass 조건: v 가 양수(+)가 찍혀야 합니다.


Step 2: LQR 확인하기
1) P/D항 확인: 상체를 앞으로 확 엎어뜨려 봅니다. (theta > 0, theta_dot > 0)
- Pass 조건: u_P < 0 , u_D < 0  
2) V항 확인: 바퀴를 앞으로 죽 밀어봅니다. 
- Pass 조건: u_V < 0

*/
{
  static unsigned long last = 0;
  if (millis() - last > 500) 
  {
    last = millis();
    
    // 목표 상태 (보통 x_d = 0)
    float err_theta = x_d(0) - x(0);
    float err_theta_dot = x_d(1) - x(1);
    float err_v = x_d(2) - x(2);

    // K 배열의 0번째 인덱스 (h=0.13 근처의 하드코딩된 RW 게인 대략적용)
    // offMode 실험이므로, RUN mode 게인을 대입해서 실제 거동 시뮬레이션 
    float K0 = 1.63f; 
    float K1 = 0.18f; 
    float K2 = 0.23f; 

    float u_P = K0 * err_theta;      // P 제어력 (각도)
    float u_D = K1 * err_theta_dot;  // D 제어력 (각속도 댐핑)
    float u_V = K2 * err_v;          // V 제어력 (속도 억제)

    Serial.println("========================================");
    Serial.printf("[상태 변수] theta: %+.2f | theta_dot: %+.2f | v: %+.2f\n", 
                  x(0) * 180 / M_PI, x(1) * 180 / M_PI, x(2));
                  
    // P와 D가 같은 방향의 힘을 내는지(음의 피드백), 반대로 싸우는지(양의 피드백) 확인
    /* v_d = 0인 상태입니다. 기체를 억지로 앞으로 굴렸으므로 (v > 0), 
       LQR은 u_V 항에 음수(-)가 찍혀야 정상입니다. */
    Serial.printf("[RW 토크 기여도] u_P(각도): %+.2f | u_D(댐핑): %+.2f | u_V(속도): %+.2f\n", 
                  u_P, u_D, u_V);
    Serial.println("========================================");
  }
}
#if 0
void dbgHeight()
// params.h SERVO_LINK_ANGLE 결정을 위한 디버깅 함수
{
  static unsigned long last = 0;
  if (millis() - last > 1000)
  {
    last = millis();
    h_d = ble.getH();
    Pol.setHR(h_d, phi_d);
    Pol.calculate_com_and_inertia();

    Eigen::Vector2f hips = Pol.get_theta_hips();
    Serial.printf("h=%.3f RH=%.2f LH=%.2f\n",
                  h_d, hips(0) * 180 / M_PI, hips(1) * 180 / M_PI);
    float theta_eq_val;
    Pol.get_theta_eq(theta_eq_val);
    Serial.printf("h=%.3f theta_eq=%.3f theta_hat=%.3f\n",
                  h_d,
                  theta_eq_val * 180 / M_PI,
                  x(0) * 180 / M_PI);
    // IMU 원시값
    Serial.printf("acc: %.3f %.3f %.3f | gyr: %.3f %.3f %.3f\n",
        MPU6050.acc_vec(0), MPU6050.acc_vec(1), MPU6050.acc_vec(2),
        MPU6050.gyr_vec(0), MPU6050.gyr_vec(1), MPU6050.gyr_vec(2));
    // EKF 상태
    Serial.printf("eq=%.3f hat=%.3f hat_dot=%.3f v=%.4f\n",
        (theta_eq_val) * 180/M_PI,
        x(0) * 180/M_PI,
        x(1) * 180/M_PI,
        x(2));
    // z 측정벡터
    Serial.printf("z: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
        z(0),z(1),z(2),z(3),z(4),z(5),z(6),z(7));
  }
}
#endif
