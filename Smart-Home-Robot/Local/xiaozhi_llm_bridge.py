#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小智AI串口监听桥接程序
直接调用项目中的 multimodal_llm 模块
"""

import serial
import serial.tools.list_ports
import time
import json
from datetime import datetime
import argparse

# 导入项目现有模块
from multimodal_llm import create_llm, DEFAULT_PROMPTS
from minimax_tts import MiniMaxTTS
from audio_player import StreamingAudioPlayer
from config import load_env_file, print_config_status


class XiaoZhiSerialBridge:
    """小智AI串口桥接器 - 复用现有模块"""

    def __init__(self, port='/dev/ttyACM0', baudrate=115200, prompt_type="智能家居"):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.system_prompt = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["通用"])

        # 使用现有的LLM和TTS
        self.llm = create_llm()
        self.tts = MiniMaxTTS()
        self.conversation_history = []

    def list_ports(self):
        """列出可用串口"""
        ports = serial.tools.list_ports.comports()
        print("=" * 50)
        print("可用串口:")
        for p in sorted(ports):
            print(f"  {p.device}: {p.description}")
        print("=" * 50)

    def connect(self):
        """连接串口"""
        try:
            print(f"连接串口: {self.port} ({self.baudrate})")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)
            print("✅ 连接成功")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    def parse_line(self, line):
        """解析串口消息"""
        line = line.strip()
        if "[" in line and "]:" in line:
            idx = line.index("]:")
            role = line[1:idx]
            content = line[idx+2:].strip()
            return role, content
        return "unknown", line

    def call_llm(self, text):
        """调用LLM"""
        self.conversation_history.append({"role": "user", "content": text})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])

        print(f"\n[LLM请求] {text}")
        print("[LLM回复] ", end='', flush=True)

        response = self.llm.chat_with_text_stream(
            text=text,
            system_prompt=self.system_prompt,
            callback=lambda c: print(c, end='', flush=True)
        )
        print()

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def play_tts(self, text):
        """TTS播放"""
        if not self.tts.api_key:
            print("\n[TTS] 未配置，跳过")
            return

        print("\n[TTS] 播放中...")
        player = StreamingAudioPlayer()

        player.has_more_data = True
        self.tts.text_to_speech(
            text=text,
            voice_id="Chinese (Mandarin)_Warm_Bestie",
            stream=True,
            output_format="hex",
            on_stream_chunk=lambda h: player.write(bytes.fromhex(h)),
            output_path="response.pcm"
        )
        player.has_more_data = False
        player.wait_for_finish(timeout=100000.0)
        player.stop()

    def run(self):
        """运行监听循环"""
        if not self.ser:
            print("错误: 串口未连接")
            return

        # 启动运行标志
        self.running = True

        print("\n" + "=" * 50)
        print("🎧 小智AI串口监听启动 (Ctrl+C退出)")
        print("=" * 50)

        buffer = ""
        last_msg = ""

        try:
            while self.running:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue

                        role, content = self.parse_line(line)

                        if role == "USER" and content and content != last_msg:
                            last_msg = content
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"\n[{ts}] 👤 [USER]: {content}")

                            response = self.call_llm(content)
                            self.play_tts(response)

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n⏹️ 已停止")
        finally:
            self.running = False

    def close(self):
        """关闭串口"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


def main():
    parser = argparse.ArgumentParser(description='小智AI串口桥接')
    parser.add_argument('--port', default='/dev/ttyACM0', help='串口路径')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--list', action='store_true', help='列出串口')
    parser.add_argument('--prompt', default='智能家居', choices=['智能家居', '康养', '客服', '通用'],
                        help='系统提示词类型')

    args = parser.parse_args()

    # 先加载环境变量，再显示配置状态
    load_env_file()
    print_config_status()

    bridge = XiaoZhiSerialBridge(
        port=args.port,
        baudrate=args.baudrate,
        prompt_type=args.prompt
    )

    if args.list:
        bridge.list_ports()
        return

    bridge.list_ports()

    if bridge.connect():
        try:
            bridge.run()
        finally:
            bridge.close()


if __name__ == "__main__":
    main()
