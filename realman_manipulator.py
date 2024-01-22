import socket
import json
from typing import Self


class RealManManipulator:
    def __init__(self, target_ip: str = '192.168.1.18', target_port: int = 8080) -> None:
        self.__target_ip = target_ip
        self.__target_port = target_port
        self.__socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def connect(self) -> None:
        self.__socket_client.connect((self.__target_ip, self.__target_port))

    def close(self) -> None:
        self.__socket_client.close()

    def flush(self) -> None:
        self.__socket_client.setblocking(False)
        try:
            while True:
                data = self.__socket_client.recv(1024)
                if not data:
                    break
        except socket.error:
            pass
        finally:
            self.__socket_client.setblocking(True)

    def json_receive(self, command: str | None) -> dict:
        file_io = self.__socket_client.makefile()
        while True:
            msg = json.loads(file_io.readline())
            if msg.get('command') == command:
                return msg

    def json_send(self, msg: dict) -> None:
        self.__socket_client.sendall(json.dumps(msg).encode('ascii') + b'\r\n')

    def json_call(self, msg: dict) -> dict:
        self.json_send(msg)
        return self.json_receive(msg.get('command'))

    def movej(self, joint: list[int] | tuple[int], v: int, r: int) -> dict:
        return self.json_call({
            'command': 'movej',
            'joint': joint,
            'v': v,
            'r': r
        })

    def movel(self, pose: list[int] | tuple[int], v: int, r: int) -> dict:
        return self.json_call({
            'command': 'movel',
            'pose': pose,
            'v': v,
            'r': r
        })

    def movec(self, pose_via: list[int] | tuple[int], pose_to: list[int] | tuple[int], v: int, r: int, loop: int) -> dict:
        return self.json_call({
            'command': 'movec',
            'pose': {
                'pose_via': pose_via,
                'pose_to': pose_to
            },
            'v': v,
            'r': r,
            'loop': loop
        })

    def movej_canfd(self, joint: list[int] | tuple[int], follow: bool = True, expand: None | int = None) -> dict:
        msg = {
            'command': 'movej_canfd',
            'joint': joint,
            'follow': follow
        }
        if expand is not None:
            msg['expand'] = expand
        return self.json_call(msg)

    def movep_canfd(self, pose: list[int] | tuple[int] | None = None, pose_quat: list[int] | tuple[int] | None = None, follow: bool = True) -> dict:
        assert bool(pose) != bool(pose_quat)
        if pose is not None:
            return self.json_call({
                'command': 'movep_canfd',
                'pose': pose,
                'follow': follow
            })
        if pose_quat is not None:
            return self.json_call({
                'command': 'movep_canfd',
                'pose_quat': pose_quat,
                'follow': follow
            })

    def movej_p(self, pose: list[int] | tuple[int], v: int, r: int) -> dict:
        return self.json_call({
            'command': 'movec',
            'pose': pose,
            'v': v,
            'r': r
        })
