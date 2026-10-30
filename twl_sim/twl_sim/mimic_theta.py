import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

class Theta3Plotter(Node):
    def __init__(self):
        super().__init__('theta3_plotter')
        
        # 링크 길이 설정 (단위: mm)
        self.l1 = 75.0
        self.l2 = 77.0
        self.l3 = 50.0
        self.l4 = 106.0

        self.plot_theta3()

    def calculate_theta3(self, theta2_deg):
        # 각도를 라디안으로 변환
        theta2 = np.radians(theta2_deg)

        # 티스토리 공식 기반 Freudenstein's Equation 변수
        K1 = (self.l4**2 - self.l2**2 - self.l3**2 - self.l1**2) / (2 * self.l2 * self.l3)
        K2 = self.l1 / self.l3
        K3 = self.l1 / self.l2

        A = K1 + (K2+1)*np.cos(theta2) - K3
        B = -2 * np.sin(theta2)
        C = K1 + (K2-1)*np.cos(theta2) + K3

        # B^2 - 4AC 판별식 체크가 필요할 수 있음 (조립 불가능 영역)
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return None
        
        half_tan_theta3 = (-B - np.sqrt(discriminant)) / (2 * A)
        theta3 = 2 * np.arctan(half_tan_theta3)
        return np.degrees(theta3)

    def plot_theta3(self):
        # theta2 범위를 0도에서 180도까지 테스트 (기구학적 한계 고려 필요)
        theta2_range = np.linspace(0, 180, 360)
        theta3_values = []
        valid_theta2 = []

        for t2 in theta2_range:
            t3 = self.calculate_theta3(t2)
            if t3 is not None:
                theta3_values.append(t3)
                valid_theta2.append(t2)

        plt.figure(figsize=(10, 6))
        plt.plot(valid_theta2, theta3_values, label=r'$\theta_3$ Value')
        plt.title('Theta 3 vs Theta 2 (4-Bar Linkage)')
        plt.xlabel('Theta 2 (Active Thigh Angle) [deg]')
        plt.ylabel('Theta 3 (Coupler Angle) [deg]')
        plt.grid(True)
        plt.legend()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = Theta3Plotter()
    # 그래프 창을 닫으면 노드 종료
    rclpy.shutdown()

if __name__ == '__main__':
    main()