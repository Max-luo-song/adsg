# server.py
# -*- coding: utf-8 -*-
import argparse
import struct
import traceback
import socket
import threading
import Pose_pb2

import inference
import os
import glob

import time

class TCPServer:
    #def __init__(self, host='0.0.0.0', port=9999):
    def __init__(self, host='0.0.0.0', port=22):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def start(self):
        """启动TCP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True

            print(f"[Socket] Server listening on {self.host}:{self.port}")

            while self.running:
                client_socket, client_address = self.socket.accept()
                print(f"[Socket] New connection: {client_address}")

                t = threading.Thread(
                    target=self.handle_client_pb,
                    args=(client_socket, client_address)
                )
                t.daemon = True
                t.start()

        except Exception as e:
            print(f"[Socket] Server start error: {e}")

    def handle_client_pb(self, conn, addr):
        """接收 protobuf pose 并渲染"""
        try:
            while True:
                header = conn.recv(4)
                if not header:
                    return
                data_length = int.from_bytes(header, "big")

                data = b""
                while len(data) < data_length:
                    packet = conn.recv(data_length - len(data))
                    if not packet:
                        break
                    data += packet

                pose_msg = Pose_pb2.MainCarInfo()
                pose_msg.ParseFromString(data)

                print(f"[Socket] Received pose from {addr}: "
                      f"x={pose_msg.x}, y={pose_msg.y}, z={pose_msg.z}, yaw={pose_msg.yaw}, roll={pose_msg.roll}, pitch={pose_msg.pitch}, timestamp={pose_msg.timestamp}, camera_id={pose_msg.camera_id}")
                print("pose_msg", pose_msg)
                start_time = time.time()
                # render cam images
                try:                
                    output = inference.cam_renderer_manager.render_from_pose(pose_msg)
                except Exception as e:
                    traceback.print_exc()
                    output = None
                # output = {0: "realtime_output/cam0_1766045546.420.png"}
                print("output:", output)
                # time.sleep(5)
                end_time_model = time.time()
                print(f"render_from_pose模型执行时间: {end_time_model - start_time:.4f} 秒")

                # render lidar points (lidar还没调通，暂时注释)
                # lidar_output = inference.lidar_renderer_manager.render_from_pose(pose_msg)
                # lidar_render_result_path = os.path.join(lidar_output, "renders")
                # lidar_txt_files = glob.glob(os.path.join(lidar_render_result_path, "*.txt"))

                try:
                    if output is not None:
                        # 构建要发送的 items 列表，每项是 (type_code, name, bytes)
                        # type_code: 1=image, 2=lidar
                        for cam_id, image_path in output.items():
                            print("image_path ==>", image_path)
                            with open(image_path, 'rb') as f:
                                image_data = f.read()
                            
                            response_proto = Pose_pb2.MainCarInfo()
                            response_proto.timestamp = pose_msg.timestamp
                            response_proto.camera_id = pose_msg.camera_id
                            response_proto.image_data = image_data
                            
                            data_bytes = response_proto.SerializeToString()
                            
                            print("data_bytes length is ==>", len(data_bytes))
                            # 发送数据长度
                            #conn.sendall(struct.pack('>I', len(image_data)))
                            #conn.sendall(image_data)

                            conn.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)  
                            #conn.sendall(data_bytes)
                        """
                        items = []

                        for cam_id, image_path in output.items():
                            with open(image_path, 'rb') as f:
                                image_data = f.read()
                            name = str(cam_id)
                            items.append((1, name, image_data))

                        # for txt_file in lidar_txt_files:
                        #     with open(txt_file, 'rb') as f:
                        #         txt_data = f.read()
                        #     name = os.path.basename(txt_file)
                        #     items.append((2, name, txt_data))

                        # 先发送 item 数量
                        conn.sendall(struct.pack('>I', len(items)))

                        # 发送每一项：1字节type + 4字节name_len + name + 4字节data_len + data
                        for typ, name, data_bytes in items:
                            try:
                                conn.sendall(bytes([typ]))
                                name_b = name.encode('utf-8')
                                conn.sendall(struct.pack('>I', len(name_b)))
                                conn.sendall(name_b)
                                conn.sendall(struct.pack('>I', len(data_bytes)))
                                conn.sendall(data_bytes)
                            except (ConnectionResetError, BrokenPipeError) as e:
                                print(f"发送时连接被重置或断开: {e}")
                                break

                            if typ == 1:
                                print(f"图片发送成功: {name}")
                            else:
                                print(f"Lidar数据发送成功: {name}")
                            """
                    else:
                        response_proto = Pose_pb2.MainCarInfo()
                        response_proto.timestamp = pose_msg.timestamp
                        response_proto.camera_id = pose_msg.camera_id
                        response_proto.image_data = b""
                            
                        data_bytes = response_proto.SerializeToString()
                            
                        print("data_bytes length is ==>", len(data_bytes))
                            # 发送数据长度
                        conn.sendall(struct.pack('>I', len(data_bytes)) + data_bytes)  
                    
                    end_time_trans_data = time.time()
                    print(f"传输图片的时间: {end_time_trans_data - end_time_model:.4f} 秒")
                except Exception as e:
                    print(f"发送失败: {e}")
                    traceback.print_exc()
                
                """
                # 发送结束状态（长度前缀 + 内容）
                try:
                    status = b"OK"
                    conn.sendall(struct.pack('>I', len(status)))
                    conn.sendall(status)
                except Exception:
                    pass
                """

        except Exception as e:
            print(f"[Socket] Error with {addr}: {e}")
        finally:
            conn.close()
            print(f"[Socket] Connection closed: {addr}")

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        print("[Socket] Server stopped")


def main(args):
    # init renderer
    inference.init_renderer_cam(
        resume_from=args.resume_from,
        source_path=args.source_path
    )
    inference.init_renderer_lidar(
        lidar_checkpoint_path=args.lidar_checkpoint_path,
        source_path=args.source_path,
        output_dir=args.output_dir
    )

    server = TCPServer()
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...", add_help=True)
    # eval
    parser.add_argument(
        "--resume_from",
        default="/nas_thoru/oldbak/zyj/workspace/scene_reconstruction/output/qcraft_20251025_163358_QCOYSD504206_1595_1610/20260104_lidar+cam0_1_2_3_5_6_7_9_10_11_12depth_loss/checkpoint_final.pth",
        help="path to checkpoint to resume from",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--source_path",
        default="/nas_thoru/scenario_output/zhangyingjun/processed/training/20251025_163358_QCOYSD504206_1595_1610",
        help="data source path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--lidar_checkpoint_path",
        default="/home/data/lidar/20251025_163358_QCOYSD504206_1595_1610",
        help="path to LiDAR checkpoint to resume from",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        default="./realtime_output",
        help="output directory for rendered results",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    main(args)
