
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小智 AI 串口监听程序
监听 ESP32-S3 发送的小智 AI 回答并显示
"""

import serial
import serial.tools.list_ports
import time
import json
from datetime import datetime

class XiaoZhiListener:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        """
        初始化串口监听器

        参数:
            port: 串口设备路径
            baudrate: 波特率（默认 115200）
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def list_serial_ports(self):
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        print("=" * 60)
        print("可用的串口设备：")
        print("=" * 60)
        for port, desc, hwid in sorted(ports):
            print(f"设备: {port}")
            print(f"描述: {desc}")
            print(f"ID: {hwid}")
            print("-" * 60)

    def connect(self):
        """连接到串口"""
        try:
            print(f"正在连接到 {self.port} (波特率: {self.baudrate})...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)  # 等待串口稳定
            print(f"✅ 成功连接到 {self.port}")
            return True
        except serial.SerialException as e:
            print(f"❌ 连接失败: {e}")
            print("\n请检查：")
            print("1. 串口设备路径是否正确")
            print("2. USB 线是否连接正常")
            print("3. 当前用户是否有串口访问权限")
            print("   解决方法: sudo usermod -a -G dialout $USER")
            return False

    def parse_message(self, line):
        """
        解析接收到的消息

        支持的格式:
        1. 简单格式: [XIAOZHI]: 消息内容
        2. JSON 格式: {"role":"assistant","content":"消息内容"}
        """
        line = line.strip()

        # 尝试解析 JSON 格式
        if line.startswith("{"):
            try:
                data = json.loads(line)
                return data.get("role", "unknown"), data.get("content", "")
            except json.JSONDecodeError:
                pass

        # 解析简单格式 [ROLE]: content
        if line.startswith("[") and "]:" in line:
            role_end = line.index("]:")
            role = line[1:role_end]
            content = line[role_end + 2:].strip()
            return role, content

        # 无法识别的格式
        return "unknown", line

    def format_output(self, role, content):
        """格式化输出"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 根据角色设置不同的样式
        role_icons = {
            "XIAOZHI": "🤖",
            "USER": "👤",
            "assistant": "🤖",
            "user": "👤"
        }

        icon = role_icons.get(role, "💬")

        return f"[{timestamp}] {icon} [{role}]: {content}"

    def listen(self):
        """开始监听串口"""
        if not self.ser:
            print("❌ 串口未连接，请先调用 connect()")
            return

        print("\n" + "=" * 60)
        print("🎧 开始监听小智 AI 消息...")
        print("按 Ctrl+C 退出")
        print("=" * 60 + "\n")

        buffer = ""

        try:
            while True:
                if self.ser.in_waiting > 0:
                    # 读取数据
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    # 按行处理
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)

                        if line.strip():
                            # 解析消息
                            role, content = self.parse_message(line)

                            # 过滤非相关消息
                            if role in ["XIAOZHI", "USER", "assistant", "user"]:
                                # 格式化并输出
                                output = self.format_output(role, content)
                                print(output)

                                # 可选：同时保存到文件
                                self.save_to_file(role, content)

                time.sleep(0.01)  # 减少 CPU 占用

        except KeyboardInterrupt:
            print("\n\n⏹️  监听已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")

    def save_to_file(self, role, content):
        """保存消息到文件（可选）"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"xiaozhi_log_{timestamp}.txt"

            with open(filename, 'a', encoding='utf-8') as f:
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{role}]: {content}\n"
                f.write(log_entry)
        except Exception as e:
            print(f"保存文件失败: {e}")

    def close(self):
        """关闭串口连接"""
        if self.ser:
            self.ser.close()
            print("串口已关闭")


def main():
    """主函数"""
    # 创建监听器实例
    listener = XiaoZhiListener(
        port='/dev/ttyACM0',  # 根据实际情况修改
        baudrate=115200
    )

    # 列出所有可用串口
    listener.list_serial_ports()

    # 连接到串口
    if listener.connect():
        # 开始监听
        listener.listen()
        # 关闭连接
        listener.close()


if __name__ == "__main__":
    main()
