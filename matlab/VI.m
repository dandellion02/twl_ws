clear; clc;

%% 1. 설정값 및 하이퍼파라미터 (논문 Section IV-B 기준)
n = 3; % 상태 변수 개수 [theta, theta_dot, v_error]
m = 1; % 입력 변수 개수 [torque]

% 비용 함수 가중치
Q = diag([1400, 70, 70]); 
r = 70;

% VI 알고리즘 파라미터
h = 1e-7;      % 알고리즘 시간 s의 스텝 사이즈
epsilon = 0.001; % 수렴 임계값
sf = 0;         % 시작 시간

%% 2. 데이터 로드 및 전처리
% [주의] CSV 파일은 [time, theta, theta_dot, v_error, torque] 컬럼을 가진다고 가정합니다.
% 만약 파일이 없다면 아래 'readmatrix' 부분을 실제 파일명으로 수정하세요.
% tau_LW, tau_RW : 양쪽 모터 토크
% theta_hat : 피치 각도
% theta_dot_hat : 피치 각속도
% v_hat : 선속도
% v_d : 목표선속도

try
    data = readmatrix('logdata/logdata1_fwd_bwd.csv'); 
    t_raw = data(:, 1);          % 시간 (s)
    theta = data(:, 24);      % 피치 각도 (rad)
    theta_dot = data(:, 23);  % 피치 각속도 (rad/s)
    %theta = data(:, 24) * 180 / pi;      % 피치 각도 (degree)
    %theta_dot = data(:, 23) * 180 / pi;  % 피치 각속도 (degree/s)
    v_error = data(:, 26) - data(:, 25);    % 선속도 오차 (m/s)
    tau = data(:, 18) + data(:, 19); % tau_LW + tau_RW, 바퀴 토크 (N.m) 
    

    N = size(data, 1);
    Ts = 0.008;        % 샘플링 주기 (0.008s)
    
catch
    warning('CSV 파일을 찾을 수 없습니다. 예시용 더미 데이터를 생성합니다.');
    N = 1000; Ts = 0.01;
    t = (0:N-1)' * Ts;
    theta = randn(N, 1) * 0.1; theta_dot = randn(N, 1) * 0.1;
    v_error = randn(N, 1) * 0.1; tau = randn(N, 1) * 0.5;
end

%% 3. 핵심 행렬 Dxx, Ixx, Ix_tau 구성 (식 8 및 Algorithm 1)
fprintf('데이터 전처리(사다리꼴 적분) 시작...\n');

Dxx = []; Ixx = []; Ix_tau = [];

for i = 1:N-1
    x_i = [theta(i); theta_dot(i); v_error(i)];
    x_next = [theta(i+1); theta_dot(i+1); v_error(i+1)];
    tau_i = tau(i);
    tau_next = tau(i+1);
    
    % Dxx: vecv(x)의 차이값
    Dxx = [Dxx; (vecv(x_next) - vecv(x_i))'];
    
    % Ixx: vecv(x)의 적분 (사다리꼴 방식)
    Ixx_val = (vecv(x_i) + vecv(x_next)) * Ts / 2;
    Ixx = [Ixx; Ixx_val'];
    
    % Ix_tau: tau * x의 적분 (사다리꼴 방식)
    Ix_tau_val = (tau_i * x_i + tau_next * x_next) * Ts / 2;
    Ix_tau = [Ix_tau; Ix_tau_val'];   
end

% M 행렬 구성: M = [Ixx, 2*r*Ix_tau] (식 9)
M = [Ixx, 2 * r * Ix_tau];
%% [추가 코드] 행렬 M의 Rank 확인 및 PE 조건 검증

% 1. 이론적으로 요구되는 Full Column Rank 계산 (n=3, m=1 기준 9)
required_rank = n*(n+1)/2 + m*n; 
actual_rank = rank(M);

fprintf('\n--- 행렬 M 상태 점검 ---\n');
fprintf('요구되는 Rank: %d\n', required_rank);
fprintf('실제 측정된 Rank: %d\n', actual_rank);

% 2. Rank 부족 시 경고 및 원인 출력
if actual_rank < required_rank
    warning('경고: 행렬 M이 Full Column Rank가 아닙니다 (Rank 부족: %d < %d).', actual_rank, required_rank);
    fprintf('원인 분석:\n');
    fprintf('1. 데이터 수집 시 탐색 노이즈(beta)가 부족하여 시스템이 충분히 자극되지 않았을 수 있습니다(PE 조건 미달) [1, 2].\n');
    fprintf('2. 데이터 세트 수(N)가 너무 적을 수 있습니다 (최소 700~1200개 권장) [3].\n');
    fprintf('3. 센서 데이터에 변화가 거의 없는 정지 상태의 데이터일 가능성이 큽니다 [2].\n');
    fprintf('결과 예측: K 값이 NaN으로 나오거나 계산 결과가 발산할 수 있습니다.\n');
else
    fprintf('확인: 행렬 M이 Full Rank를 만족합니다. 학습을 진행합니다.\n');
end
M_dagger = pinv(M, 1e-6); % 의사 역행렬 계산 (Step 4), 1e-6보다 작은 고윳값은 무시하여 수치적 안정성 확보

%% 4. 가치 반복(VI) 루프 실행 (Step 5-8)
fprintf('VI 루프 연산 중...\n');

% y = [vecs(H); vec(K)] 초기화 (Step 4)
y = zeros(n*(n+1)/2 + m*n, 1); 
y_old = ones(size(y)) * inf;
iter = 0;

while norm(y - y_old) > epsilon
    y_old = y;
    iter = iter + 1;
    
    % y로부터 H(s)와 K(s) 복원 (식 13)
    vecs_H = y(1:6);
    vec_K = y(7:9);
    
    H_s = [vecs_H(1), vecs_H(2)/2, vecs_H(3)/2;
           vecs_H(2)/2, vecs_H(4), vecs_H(5)/2;
           vecs_H(3)/2, vecs_H(5)/2, vecs_H(6)];
    K_s = reshape(vec_K, [m, n]);
    
    % f(y) 계산 (식 13)
    term_inside = -H_s + r * (K_s' * K_s) - Q;
    f_y = M_dagger * Dxx * vecs(term_inside);
    
    % 오일러 업데이트 (Step 7)
    y = y - h * f_y; 
    
    % [디버깅 추가] NaN 발생 시 즉시 멈추고 직전 값 확인
    if any(isnan(y))
        fprintf('iter %d에서 NaN. 직전 y값 (index:value):\n', iter);
        for k = 1:numel(y_old)
            fprintf('  %d: %g\n', k, y_old(k));
        end
        error('중단: NaN 검출');
    end


    if iter > 100000 % 안전 장치
        break;
    end
end

%% 5. 결과 재구성 및 출력 (Step 9-10)
K_optimal = reshape(y(7:9), [m, n]);

fprintf('\n=== 학습 결과 ===\n');
fprintf('최종 반복 횟수: %d\n', iter);
fprintf('추출된 최적 Gain K:\n');
disp(K_optimal);

%% 헬퍼 함수 정의
function v = vecv(x)
    % 식 (12): 상태 벡터를 조합하여 6차원 벡터 생성
    v = [x(1)^2; x(1)*x(2); x(1)*x(3); x(2)^2; x(2)*x(3); x(3)^2];
end

function vs = vecs(W)
    % 식 (12): 대칭 행렬을 6차원 벡터로 변환
    vs = [W(1,1); 2*W(1,2); 2*W(1,3); W(2,2); 2*W(2,3); W(3,3)];
end