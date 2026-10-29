# 用于模拟客户端发送Protobuf消息到server，不部署在我们的系统中
# -*- coding: utf-8 -*-
import time

import Pose_pb2
import socket
import random
import os


def data():
    # 1. 创建Protobuf消息对象并填充数据
    my_msg = Pose_pb2.MainCarInfo()
    # x=7969.7074556988155, y=-2922.589950684925, yaw=1.9608628455457437,camera_id=CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99
    my_msg.x = 7987.08639
    my_msg.y = -2964.6539
    my_msg.z = -396.08292
    my_msg.yaw = 1.9608628455457437
    my_msg.roll = 0.000000
    my_msg.pitch = 0.000000
    my_msg.timestamp = 1761382841.0010262
    my_msg.camera_id = "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60"
    print(my_msg.x, my_msg.y, my_msg.z, my_msg.yaw)
    return my_msg

class TCPClient:
    #def __init__(self, host='0.0.0.0', port=9999):
    def __init__(self, host='0.0.0.0', port=22):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_message_pb(self):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            # 2. 序列化为字节串
            my_msg = data()
            serialized_data = my_msg.SerializeToString()
            print(serialized_data)
            # 3. 计算长度并打包为4字节的消息头 (使用大端字节序)
            data_length = len(serialized_data)
            length_header = data_length.to_bytes(4, byteorder='big')  # 4字节头
            # 先发送长度头，再发送序列化数据
            self.socket.sendall(length_header + serialized_data)
            #self.socket.send(message.encode('utf-8'))

            #time.sleep(100)
            # 接收响应
            #response = self.socket.recv(1024)
            #print(f"服务器响应: {response.decode('utf-8')}")
            print(f"服务器响应 OK")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def recv_all(self, n: int) -> bytes:
        """从 socket 中严格读取 n 字节，若连接提前关闭则抛出异常。"""
        data = b""
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                raise ConnectionError("连接在读取时被关闭")
            data += packet
        return data

    def _save_image(self, data_bytes: bytes, name: str, index: int, img_dir: str) -> str:
        ext = ''
        if '.' in name:
            ext = os.path.splitext(name)[1]
        else:
            if data_bytes[:4] == b'\x89PNG':
                ext = '.png'
            elif data_bytes[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            else:
                ext = '.bin'
        fname = os.path.join(img_dir, f"{int(time.time())}_{index}{ext}")
        with open(fname, 'wb') as wf:
            wf.write(data_bytes)
        return fname

    def _save_lidar(self, data_bytes: bytes, name: str, lidar_dir: str) -> str:
        fname = os.path.join(lidar_dir, name)
        with open(fname, 'wb') as wf:
            wf.write(data_bytes)
        return fname

    def receive_items(self) -> bool:
        """接收并保存服务器发来的 items（type/name/len/data）"""
        try:
            header = self.recv_all(4)
            num_items = int.from_bytes(header, 'big')
            print(f"将接收 {num_items} 项目")

            out_dir = './realtime_output/received'
            img_dir = os.path.join(out_dir, 'images')
            lidar_dir = os.path.join(out_dir, 'lidar')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lidar_dir, exist_ok=True)

            for i in range(num_items):
                typ = self.recv_all(1)[0]
                name_len = int.from_bytes(self.recv_all(4), 'big')
                name = self.recv_all(name_len).decode('utf-8')
                size = int.from_bytes(self.recv_all(4), 'big')
                data_bytes = self.recv_all(size)

                if typ == 1:
                    fname = self._save_image(data_bytes, name, i, img_dir)
                    print(f"收到图片并保存为: {fname}")
                elif typ == 2:
                    fname = self._save_lidar(data_bytes, name, lidar_dir)
                    print(f"收到Lidar文件并保存为: {fname}")
                else:
                    print(f"未知类型 {typ} 接收到，保存为原始文件")
                    fname = os.path.join(out_dir, f"item_{i}.bin")
                    with open(fname, 'wb') as wf:
                        wf.write(data_bytes)

            # 读取结束状态（长度前缀 + 内容）
            status_len = int.from_bytes(self.recv_all(4), 'big')
            status = self.recv_all(status_len)
            try:
                print(f"服务器响应: {status.decode('utf-8')}")
            except Exception:
                print(f"服务器响应 (binary): {status}")

            return True
        except Exception as e:
            print(f"接收项目时出错: {e}")
            return False

    def send_message(self, message):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            self.socket.send(message.encode('utf-8'))

            # 接收响应
            response = self.socket.recv(1024)
            print(f"服务器响应: {response.decode('utf-8')}")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            print("连接已关闭")

if __name__ == "__main__":
    client = TCPClient()

    if client.connect():
        # 发送测试消息
        messages = ["Hello, Server!", "测试消息2", "测试消息3"]

        for msg in range(1,11):
            # 发送 protobuf 消息
            ok = client.send_message_pb()
            if not ok:
                print("发送失败，跳出循环")
                break

            # receive
            client.receive_items()

        client.close()

    
