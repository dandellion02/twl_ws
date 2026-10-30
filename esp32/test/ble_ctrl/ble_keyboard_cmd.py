import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from bleak import BleakClient, BleakScanner
#virtual environment 3.14.4 terminal command: uv venv && uv pip install bleak && python c:/Users/wjddb/Downloads/WBR_MAIN-tunning/test/ble_ctrl/ble_keyboard_cmd.py

DEFAULT_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
DEFAULT_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
PREFERRED_DEVICE_NAME = "ESP32_WBR_Control"
YAW_HW_BIAS = 0.03

class AsyncBleWorker:
    def __init__(self, ui_log, ui_status, ui_devices):
        self.ui_log = ui_log
        self.ui_status = ui_status
        self.ui_devices = ui_devices
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.client = None
        self.rx_uuid = None
        self.tx_uuid = None

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def scan(self):
        self.ui_status("Scanning...")
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            rows = []
            seen = set()
            for d in devices:
                label = f"{d.name or '(No name)'} | {d.address}"
                if label not in seen:
                    seen.add(label)
                    rows.append((d.name or "(No name)", d.address))
            rows.sort(key=lambda x: 0 if x[0] == PREFERRED_DEVICE_NAME else 1)
            self.ui_devices(rows)
            self.ui_status("Scan complete")
        except Exception as e:
            self.ui_status("Scan failed")
            self.ui_log(f"[ERROR] {e}")

    async def connect(self, address, rx_uuid, tx_uuid):
        self.ui_status("Connecting...")
        try:
            client = BleakClient(address)
            await client.connect()
            self.client = client
            self.rx_uuid = rx_uuid
            self.tx_uuid = tx_uuid
            await client.start_notify(tx_uuid, self._handle_notify)
            self.ui_status("Connected")
            self.ui_log(f"Connected: {address}")
        except Exception as e:
            self.ui_status("Connect failed")
            self.ui_log(f"[ERROR] {e}")
            self.client = None

    async def disconnect(self):
        if not self.client:
            return
        try:
            if self.client.is_connected:
                await self.client.stop_notify(self.tx_uuid)
                await self.client.disconnect()
        except Exception:
            pass
        finally:
            self.client = None
            self.ui_status("Disconnected")
            self.ui_log("Disconnected")

    def _handle_notify(self, sender, data):
        self.ui_log(f"[RX] {data.decode('utf-8', errors='replace').strip()}")

    async def send_command(self, command):
        if not self.client or not self.client.is_connected:
            self.ui_log("Not connected")
            return
        try:
            await self.client.write_gatt_char(
                self.rx_uuid,
                (command + "\n").encode("utf-8"),
                response=False
            )
            self.ui_log(f"[TX] {command}")
        except Exception as e:
            self.ui_log(f"[ERROR] {e}")


class BleMotorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ESP32 WBR BLE Controller")
        self.geometry("850x950")

        self.worker = AsyncBleWorker(self.ui_log, self.ui_status, self.ui_set_devices)

        self.status_var   = tk.StringVar(value="Disconnected")
        self.device_var   = tk.StringVar()
        self.rx_uuid_var  = tk.StringVar(value=DEFAULT_RX_UUID)
        self.tx_uuid_var  = tk.StringVar(value=DEFAULT_TX_UUID)
        self.custom_var   = tk.StringVar()
        self.torque_var   = tk.IntVar(value=0)
        self.devices_map  = {}

        # 주행(Locomotion) 제어용 변수
        self.v_var = tk.DoubleVar(value=0.0)
        self.h_var = tk.DoubleVar(value=0.18) 
        self.yaw_var = tk.DoubleVar(value=0.0)

        # 라벨 위젯 레퍼런스 저장을 위한 변수
        self.v_label = None
        self.h_label = None
        self.yaw_label = None

        self._build_ui()
        self._bind_keys() # 키보드 이벤트 바인딩
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main_canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas, padding=12)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(
                scrollregion=main_canvas.bbox("all")
            )
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        outer = scrollable_frame

        # ── 연결 ──
        conn = ttk.LabelFrame(outer, text="Connection", padding=8)
        conn.pack(fill="x", expand=True)

        ttk.Label(conn, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(conn, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(6, 20))
        ttk.Button(conn, text="Scan",       command=self.scan_devices).grid(row=0, column=2, padx=4)
        ttk.Button(conn, text="Connect",    command=self.connect_device).grid(row=0, column=3, padx=4)
        ttk.Button(conn, text="Disconnect", command=self.disconnect_device).grid(row=0, column=4, padx=4)

        ttk.Label(conn, text="Device").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.device_combo = ttk.Combobox(conn, textvariable=self.device_var, state="readonly", width=58)
        self.device_combo.grid(row=1, column=1, columnspan=4, sticky="we", padx=(6, 0), pady=(8, 0))

        ttk.Label(conn, text="RX UUID").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(conn, textvariable=self.rx_uuid_var, width=52).grid(row=2, column=1, columnspan=4, sticky="we", padx=(6, 0), pady=(6, 0))

        ttk.Label(conn, text="TX UUID").grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(conn, textvariable=self.tx_uuid_var, width=52).grid(row=3, column=1, columnspan=4, sticky="we", padx=(6, 0), pady=(4, 0))

        # ── 퀵 커맨드 ──
        quick = ttk.LabelFrame(outer, text="Quick Commands", padding=8)
        quick.pack(fill="x", pady=(10, 0))

        quick_cmds = [
            ("STOP",    "stop"),
            ("RUN",     "run"),
            ("RESET",   "reset"),
            ("STATE",   "state"),
            ("IMU",     "imu"),
        ]
        for label, cmd in quick_cmds:
            ttk.Button(quick, text=label, width=8,
                       command=lambda c=cmd: self.send(c)).pack(side="left", padx=4)

# ── 로봇 주행(Locomotion) 제어 & 키보드 가이드 UI ──
        loco_frame = ttk.LabelFrame(outer, text="Keyboard Control", padding=12)
        loco_frame.pack(fill="x", pady=(10, 0))

        # 좌측: 키보드 타일 배치 프레임
        keys_frame = ttk.Frame(loco_frame)
        keys_frame.pack(side="left", padx=(0, 20))

        # 타일 스타일 지정을 위한 설정 (단단하고 깔끔한 보더 표현)
        style = ttk.Style()
        style.configure("KeyTile.TFrame", relief="solid", borderwidth=1, background="#f0f0f0")
        style.configure("KeyLabel.TLabel", background="#f0f0f0", font=("Helvetica", 10))
        style.configure("KeyMain.TLabel", background="#f0f0f0", font=("Helvetica", 16, "bold"))
        style.configure("Desc.TLabel", font=("Helvetica", 10))

        # 키보드 버튼 타일을 생성하는 헬퍼 함수
        def create_key_tile(parent, r, c, key_char, title, sub_text):
            tile = ttk.Frame(parent, style="KeyTile.TFrame", padding=6)
            tile.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            
            lbl_key = ttk.Label(tile, text=key_char, style="KeyMain.TLabel", anchor="center")
            lbl_key.pack(fill="x")
            
            lbl_title = ttk.Label(tile, text=title, style="KeyLabel.TLabel", anchor="center")
            lbl_title.pack(fill="x")
            
            lbl_sub = ttk.Label(tile, text=sub_text, style="KeyLabel.TLabel", anchor="center", foreground="#777777")
            lbl_sub.pack(fill="x")
            return tile

        # 1행: Q, W, E
        create_key_tile(keys_frame, 0, 0, "Q", "높이 ↑", "h +0.1")
        create_key_tile(keys_frame, 0, 1, "W", "속도 ↑", "v +0.1")
        create_key_tile(keys_frame, 0, 2, "E", "높이 ↓", "h -0.1")

        # 2행: A, S, D
        create_key_tile(keys_frame, 1, 0, "A", "좌측 바퀴 ↑\n우회전", "")
        create_key_tile(keys_frame, 1, 1, "S", "정지값\nv, yaw = 0", "")
        create_key_tile(keys_frame, 1, 2, "D", "우측 바퀴 ↑\n좌회전", "")

        # 3행: X (중앙에 배치하기 위해 column=1)
        create_key_tile(keys_frame, 2, 1, "X", "속도 ↓", "v -0.1")

        # 4행: R, F, C (구분을 위해 y축 공백 추가)
        ttk.Frame(keys_frame, height=10).grid(row=3, column=0, columnspan=3)
        create_key_tile(keys_frame, 4, 0, "R", "RUN\n주행 시작", "")
        create_key_tile(keys_frame, 4, 1, "F", "STOP\n즉시 정지", "")
        create_key_tile(keys_frame, 4, 2, "C", "RESET\n초기화", "")

        # 우측: 설명 글자 배치 프레임
        desc_frame = ttk.Frame(loco_frame)
        desc_frame.pack(side="left", fill="y", padx=10, pady=10)

        descriptions = [
            ("Q / E", "로봇 높이 증가 / 감소"),
            ("W / X", "전진 속도 증가 / 감소"),
            ("A", "좌측 바퀴 토크 증가 성향 → 우회전"),
            ("D", "우측 바퀴 토크 증가 성향 → 좌회전"),
            ("S", "속도와 yaw를 0으로 초기화"),
            ("R / F / C", "RUN / STOP / RESET")
        ]

        for i, (key_desc, action_desc) in enumerate(descriptions):
            ttk.Label(desc_frame, text=key_desc, font=("Helvetica", 11, "bold"), width=12).grid(row=i, column=0, sticky="w", pady=6)
            ttk.Label(desc_frame, text=action_desc, style="Desc.TLabel").grid(row=i, column=1, sticky="w", pady=6)

        # 백엔드 데이터 업데이트용 더미 라벨 레퍼런스 생성 (기존 첫 번째 코드의 대입 로직 버그 방지)
        self.v_label = ttk.Label()
        self.h_label = ttk.Label()
        self.yaw_label = ttk.Label()

        # ── 토크 슬라이더 ──
        torque_frame = ttk.LabelFrame(outer, text="Direct Torque Command (both wheels)", padding=8)
        torque_frame.pack(fill="x", pady=(10, 0))

        self.torque_label = ttk.Label(torque_frame, text="0", width=6, anchor="center")
        self.torque_label.pack(side="right", padx=(0, 8))

        self.slider = ttk.Scale(
            torque_frame, from_=-200, to=200,
            orient="horizontal", variable=self.torque_var,
            command=self._on_slider
        )
        self.slider.pack(fill="x", padx=8)

        btn_row = ttk.Frame(torque_frame)
        btn_row.pack(pady=(6, 0))
        for val in (-100, -50, -10, 0, 10, 50, 100):
            ttk.Button(
                btn_row, text=str(val), width=6,
                command=lambda v=val: self._set_torque(v)
            ).pack(side="left", padx=2)

        # ── 커스텀 커맨드 ──
        custom = ttk.LabelFrame(outer, text="Custom Command", padding=8)
        custom.pack(fill="x", pady=(10, 0))
        ttk.Entry(custom, textvariable=self.custom_var, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(custom, text="Send", command=self.send_custom).pack(side="left", padx=(8, 0))

        # ── 로그 ──
        log_frame = ttk.LabelFrame(outer, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    # ── 키보드 바인딩 및 처리 ──
    def _bind_keys(self):
        self.bind("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event):
        # Entry, Text 박스에 포커스가 있을 때는 단축키를 무시
        widget = self.focus_get()
        if isinstance(widget, (tk.Entry, tk.Text, scrolledtext.ScrolledText)):
            return

        char = event.char.lower()
        if not char:
            return

        # --------------------------------------------------------
        # 스텝 단위(민감도) 일괄 설정 (모두 0.1 단위로 통일)
        # --------------------------------------------------------
        v_step = 0.1
        yaw_step = 0.1
        h_step = 0.1

        # ── 1. 속도 (Velocity) ──
        if char == 'w':
            self._set_v(min(1.0, self.v_var.get() + v_step))
        elif char == 'x':
            self._set_v(max(-1.0, self.v_var.get() - v_step))
        
        # ── 2. 조향 (Yaw) - 좌우 반전 적용 완료 ──
        elif char == 'a':
            self._set_yaw(max(-0.26, self.yaw_var.get() - yaw_step))
        elif char == 'd':
            self._set_yaw(min(0.26, self.yaw_var.get() + yaw_step))
            
        # ── 3. 높이 (Height) ──
        elif char == 'q':
            self._set_h(min(0.20, self.h_var.get() + h_step))
        elif char == 'e':
            self._set_h(max(0.07, self.h_var.get() - h_step))
            
        # ── 4. 퀵 커맨드 (정지 및 모드 전환) ──
        elif char == 's':
            self._set_v(0.0)
            self._set_yaw(0.0)
        elif char == 'r':
            self.send("run")
        elif char == 'f':
            self.send("stop")
        elif char == 'c':
            self.send("reset")

    # ── Locomotion 슬라이더 리팩토링 (통합 제어) ──
    def _create_loco_slider(self, parent, label_text, var, min_val, max_val, update_method, default_val):
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=4)
        
        ttk.Label(row_frame, text=label_text, width=12).pack(side="left")
        
        val_label = ttk.Label(row_frame, text=f"{default_val:.2f}", width=6, anchor="e")
        val_label.pack(side="right", padx=4)

        slider = ttk.Scale(row_frame, from_=min_val, to=max_val, orient="horizontal", variable=var, 
                           command=lambda val: update_method(float(val)))
        slider.pack(side="left", fill="x", expand=True, padx=8)

        ttk.Button(row_frame, text="0 / Def", width=6, 
                   command=lambda: update_method(default_val)).pack(side="right", padx=4)
        
        return val_label 

    # 값을 세팅하고 UI업데이트 및 BLE 전송을 한 번에 처리하는 함수들
    def _set_v(self, val):
        self.v_var.set(val)
        self.v_label.config(text=f"{val:.2f}")
        self.send(f"v {-val:.2f}") 

    def _set_h(self, val):
        self.h_var.set(val)
        self.h_label.config(text=f"{val:.2f}")
        self.send(f"h {val:.2f}")

    def _set_yaw(self, val):
        self.yaw_var.set(val)
        self.yaw_label.config(text=f"{val:.2f}")
        calibrated_yaw = val + YAW_HW_BIAS
        self.send(f"yaw {val:.2f}")

    # ── 기존 토크 제어 슬라이더 ──
    def _on_slider(self, val):
        v = int(float(val))
        self.torque_var.set(v)
        self.torque_label.config(text=str(v))
        self.send(f"both t {v}")

    def _set_torque(self, val):
        self.torque_var.set(val)
        self.torque_label.config(text=str(val))
        self.send(f"both t {val}")

    # ── BLE 통신부 ──
    def scan_devices(self):
        self.worker.call(self.worker.scan())

    def connect_device(self):
        label = self.device_var.get().strip()
        if not label or label not in self.devices_map:
            messagebox.showwarning("Device", "Scan 후 기기를 선택하세요.")
            return
        self.worker.call(self.worker.connect(
            self.devices_map[label],
            self.rx_uuid_var.get().strip(),
            self.tx_uuid_var.get().strip()
        ))

    def disconnect_device(self):
        self.worker.call(self.worker.disconnect())

    def send(self, cmd):
        self.worker.call(self.worker.send_command(cmd))

    def send_custom(self):
        cmd = self.custom_var.get().strip()
        if cmd:
            self.send(cmd)

    # ── UI 콜백 ──
    def ui_log(self, msg):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def ui_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def ui_set_devices(self, rows):
        def update():
            self.devices_map.clear()
            labels = []
            for name, address in rows:
                label = f"{name} | {address}"
                labels.append(label)
                self.devices_map[label] = address
            self.device_combo["values"] = labels
            if labels:
                self.device_combo.current(0)
                self.device_var.set(labels[0])
        self.after(0, update)

    def on_close(self):
        try:
            self.worker.call(self.worker.disconnect()).result(timeout=3)
        except Exception:
            pass
        try:
            self.worker.loop.call_soon_threadsafe(self.worker.loop.stop)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = BleMotorApp()
    app.mainloop()