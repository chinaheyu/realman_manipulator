import os
import socket
import json
import time
from typing import Self, Iterable, Literal, Callable, Optional
import math
from pytransform3d.urdf import UrdfTransformManager, parse_urdf
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import transform_from, concat, invert_transform
from pytransform3d.rotations import euler_from_matrix, matrix_from_euler
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


def rad_to_deg(rad: float) -> float:
    return 180 * rad / math.pi


def deg_to_rad(deg: float) -> float:
    return math.pi * deg / 180


def clamp(value, min_value, max_value):
    if value < min_value:
        value = min_value
    if value > max_value:
        value = max_value
    return value


@dataclass
class StateFeedback:
    joints: list[float]
    pose: list[float]


@dataclass
class ForceData:
    force_data: list[float]
    zero_force_data: list[float]


class RealManManipulator:
    def __init__(self, model: Literal["ECO65", "ECO65_6F", "RM65", "RM65_6F", "RM75", "RM75_6F", "RML63", "RML63_6F"] = "RM75_6F", target_ip: str = '192.168.1.18', target_port: int = 8080, debug: bool = False) -> None:
        self.debug = debug
        self.target_ip = target_ip
        self.target_port = target_port
        self.line_buffer = b''
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.load_urdf(model)

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        self.close()

    def load_urdf(self, model: str) -> None:
        self.transform_manager = UrdfTransformManager()
        self.package_path = os.path.dirname(__file__)
        if not self.package_path.endswith('/'):
            self.package_path += '/'

        with open(os.path.join(self.package_path, f'rm_description/urdf/{model}.urdf')) as fp:
            raw_urdf_data = fp.read()

        robot_name, links, joints = parse_urdf(raw_urdf_data)
        self.links = [i.name for i in links]
        self.joints = [i.joint_name for i in joints]

        self.transform_manager.load_urdf(raw_urdf_data, package_dir=self.package_path)
        for v in self.transform_manager.visuals:
            v.color = np.array([0.0, 0.0, 0.0, 1.0])

    def connect(self) -> None:
        self.socket_client.connect((self.target_ip, self.target_port))
        self.flush()

    def close(self) -> None:
        self.socket_client.close()

    def flush(self) -> None:
        self.socket_client.setblocking(False)
        try:
            while True:
                data = self.socket_client.recv(4096)
                if not data:
                    break
        except socket.error:
            pass
        finally:
            self.socket_client.setblocking(True)

    def receive_json(self, timeout: Optional[float] = None) -> dict | None:
        self.socket_client.settimeout(timeout)
        begin = time.time()
        while timeout is None or time.time() - begin < timeout:
            try:
                line_sep = self.line_buffer.index(b'\n')
            except ValueError:
                pass
            else:
                line_data = self.line_buffer[:line_sep]
                self.line_buffer = self.line_buffer[line_sep + 1:]
                try:
                    response = json.loads(line_data)
                except json.decoder.JSONDecodeError:
                    if self.debug:
                        print(f'unable to parse json: {line_data}')
                    return None
                else:
                    if self.debug:
                        print(f'received: {response}')
                    return response
            try:
                self.line_buffer += self.socket_client.recv(4096)
            except TimeoutError:
                pass
        return None

    def send_json(self, msg: dict) -> None:
        if self.debug:
            print(f'send: {msg}')
        self.socket_client.sendall(json.dumps(msg).encode('ascii') + b'\r\n')

    def json_call(self, msg: dict, response_filter: Optional[Callable[[dict], bool]] = None) -> dict:
        self.flush()
        self.send_json(msg)
        while True:
            response = self.receive_json()
            if response_filter is None or response_filter(response):
                break
        return response

    @staticmethod
    def int_joints(joints: Iterable[float]) -> list[int]:
        return [int(rad_to_deg(i) * 1e3) for i in joints]

    @staticmethod
    def int_pose(pose: Iterable[float]) -> list[int]:
        return [int(j * 1e6 if i < 3 else j * 1e3) for i, j in enumerate(pose)]

    @staticmethod
    def int_pose_quaternion(pose_quaternion: Iterable[float]) -> list[int]:
        return [int(i * 1e6) for i in pose_quaternion]

    @staticmethod
    def int_velocity(velocity: float) -> int:
        return clamp(int(100 * velocity), 0, 100)

    @staticmethod
    def int_hand_angle(hand_angle: Iterable[Optional[float]]) -> list[int]:
        return [-1 if i is None else clamp(int(1e3 * i), 0, 1000) for i in hand_angle]

    @staticmethod
    def float_force_data(force_data: Iterable[int]) -> list[float]:
        return  [i / 1e3 for i in force_data]

    def get_current_state(self) -> StateFeedback:
        msg = self.json_call({"command": "get_current_arm_state"})
        return StateFeedback(
            joints=[deg_to_rad(i / 1e3) for i in msg['arm_state']['joint']],
            pose=[j / 1e6 if i < 3 else j / 1e3 for i, j in enumerate(msg['arm_state']['pose'])]
        )

    def get_current_force_data(self) -> ForceData:
        msg = self.json_call({"command": "get_force_data"})
        force_data = self.float_force_data(msg['force_data'])
        zero_force_data = self.float_force_data(msg['zero_force_data'])
        return ForceData(
            force_data=force_data,
            zero_force_data=zero_force_data
        )

    def get_current_transform(self) -> np.ndarray:
        self.update_transform()
        return self.transform_manager.get_transform('end_effector', self.links[0])

    def update_transform(self) -> None:
        state = self.get_current_state()
        for i, j in enumerate(state.joints):
            self.transform_manager.set_joint(self.joints[i], j)
        end_effector_to_base_link = transform_from(
            matrix_from_euler(state.pose[3:], 0, 1, 2, True),
            state.pose[:3]
        )
        base_link_to_last_link = self.transform_manager.get_transform(self.links[0], self.links[-1])
        end_effector_to_last_link = concat(end_effector_to_base_link, base_link_to_last_link)
        self.transform_manager.add_transform('end_effector', self.links[-1], end_effector_to_last_link)

    @staticmethod
    def trajectory_finished_filter(response: dict) -> bool:
        if response.get('state', '') != 'current_trajectory_state':
            return False
        if response.get('device', -1) != 0:
            return False
        return True

    def move_j(self, joints: Iterable[float], velocity: float) -> dict:
        return self.json_call({
            'command': 'movej',
            'joint': self.int_joints(joints),
            'v': self.int_velocity(velocity),
            'r': 0
        }, self.trajectory_finished_filter)

    def move_l(self, pose: Iterable[float], velocity: float) -> dict:
        return self.json_call({
            'command': 'movel',
            'pose': self.int_pose(pose),
            'v': self.int_velocity(velocity),
            'r': 0
        }, self.trajectory_finished_filter)

    def move_c(self, pose_via: Iterable[float], pose_to: Iterable[float], velocity: float, loop: int = 0) -> dict:
        return self.json_call({
            'command': 'movec',
            'pose': {
                'pose_via': self.int_pose(pose_via),
                'pose_to': self.int_pose(pose_to)
            },
            'v': self.int_velocity(velocity),
            'r': 0,
            'loop': loop
        }, self.trajectory_finished_filter)

    def move_j_continuous(self, joints: Iterable[float], follow: bool = True, expand: Optional[int] = None) -> dict:
        msg = {
            'command': 'movej_canfd',
            'joint': self.int_joints(joints),
            'follow': follow
        }
        if expand is not None:
            msg['expand'] = expand
        return self.json_call(msg)

    def move_p_continuous(self, pose: Optional[Iterable[float]] = None, pose_quaternion: Optional[Iterable[float]] = None, follow: bool = True) -> dict:
        assert bool(pose) != bool(pose_quaternion)
        if pose is not None:
            return self.json_call({
                'command': 'movep_canfd',
                'pose': self.int_pose(pose),
                'follow': follow
            })
        if pose_quaternion is not None:
            return self.json_call({
                'command': 'movep_canfd',
                'pose_quat': self.int_pose_quaternion(pose_quaternion),
                'follow': follow
            })

    def move_j_p(self, pose: Iterable[float], velocity: float) -> dict:
        return self.json_call({
            'command': 'movej_p',
            'pose': self.int_pose(pose),
            'v': self.int_velocity(velocity),
            'r': 0
        }, self.trajectory_finished_filter)

    def step_joint(self, joint: int, angle: float, velocity: float) -> dict:
        return self.json_call({
            'command': 'set_joint_step',
            'joint_step': [joint, int(rad_to_deg(angle) * 1e3)],
            'v': self.int_velocity(velocity),
            'r': 0
        })

    @staticmethod
    def pose_to_transform(pose: list[float]) -> np.ndarray:
        return transform_from(
            matrix_from_euler(pose[3:], 0, 1, 2, True),
            pose[:3]
        )

    def resolve_relative_pose(self, pose: list[float], offset: Optional[list[float]] = None) -> list[float]:
        if offset is None:
            offset_to_end_effector = np.eye(4)
        else:
            offset_to_end_effector = self.pose_to_transform(offset)
        self.update_transform()
        end_effector_to_base_link = self.transform_manager.get_transform('end_effector', self.links[0])
        target_pose_to_offset = self.pose_to_transform(pose)
        target_pose_to_base_link = end_effector_to_base_link @ offset_to_end_effector @ target_pose_to_offset @ invert_transform(offset_to_end_effector)
        xyz = target_pose_to_base_link[:3, 3]
        rpy = euler_from_matrix(target_pose_to_base_link[:3, :3], 0, 1, 2, True)
        return [*xyz, *rpy]

    def step_end_effector_pose(self, pose: list[float], velocity: float, offset: Optional[list[float]] = None) -> dict:
        target_pose = self.resolve_relative_pose(pose, offset)
        state = self.move_l(target_pose, velocity)
        if state['trajectory_state']:
            return state
        return self.move_j_p(target_pose, velocity)

    def set_hand_posture(self, posture_number: int) -> dict:
        response = self.json_call({
            'command': 'set_hand_posture',
            'posture_num': posture_number
        })
        self.receive_json(5.0)
        return response

    def set_hand_sequence(self, sequence_number: int) -> dict:
        response = self.json_call({
            'command': 'set_hand_seq',
            'seq_num': sequence_number
        })
        self.receive_json(5.0)
        return response

    def set_hand_angle(self, hand_angle: Iterable[Optional[float]]) -> dict:
        response = self.json_call({
            'command': 'set_hand_angle',
            'hand_angle': self.int_hand_angle(hand_angle)
        })
        self.receive_json(5.0)
        return response

    def set_hand_speed(self, hand_speed: float) -> dict:
        response = self.json_call({
            'command': 'set_hand_speed',
            'hand_speed': clamp(int(1e3 * hand_speed), 1, 1000)
        })
        self.receive_json(0.5)
        return response

    def set_hand_force(self, hand_force: float) -> dict:
        response = self.json_call({
            'command': 'set_hand_force',
            'hand_force': clamp(int(1e3 * hand_force), 1, 1000)
        })
        self.receive_json(0.5)
        return response

    def clear_force_data(self) -> dict:
        return self.json_call({"command": "clear_force_data"})

    def set_force_sensor(self) -> dict:
        return self.json_call({"command": "set_force_sensor"}, lambda x: x.get("command", "") == "set_force_sensor")

    def set_force_position(self, sensor: int, mode: int, direction: int, force: float) -> dict:
        response = self.json_call({
            "command": "set_force_position",
            "sensor": sensor,
            "mode": mode,
            "direction": direction,
            "N": int(10 * force)
        }, lambda x: x.get("command", "") == "set_force_position")
        return response

    def stop_force_position(self) -> dict:
        return self.json_call({"command": "stop_force_position"}, lambda x: x.get("command", "") == "stop_force_position")

    def start_force_position_continuous(self) -> dict:
        return self.json_call({"command": "Start_Force_Position_Move"}, lambda x: x.get("command", "") == "Start_Force_Position_Move")

    def force_position_continuous(self, sensor: int, mode: int, direction: int, force: float, pose: Optional[Iterable[float]] = None, pose_quaternion: Optional[Iterable[float]] = None, joints: Optional[Iterable[float]] = None, follow: bool = True) -> dict:
        assert any([pose is not None, pose_quaternion is not None, joints is not None])
        if pose is not None:
            return self.json_call({
                'command': 'Force_Position_Move',
                'pose': self.int_pose(pose),
                'sensor': sensor,
                'mode': mode,
                'dir': direction,
                'force': int(10 * force),
                'follow': follow
            })
        if pose_quaternion is not None:
            return self.json_call({
                'command': 'Force_Position_Move',
                'pose_quat': self.int_pose_quaternion(pose_quaternion),
                'sensor': sensor,
                'mode': mode,
                'dir': direction,
                'force': int(10 * force),
                'follow': follow
            })
        if joints is not None:
            return self.json_call({
                'command': 'Force_Position_Move',
                'joints': self.int_joints(joints),
                'sensor': sensor,
                'mode': mode,
                'dir': direction,
                'force': int(10 * force),
                'follow': follow
            })

    def stop_force_position_move(self) -> dict:
        return self.json_call({"command": "Stop_Force_Position_Move"}, lambda x: x.get("command", "") == "Stop_Force_Position_Move")

    def plot_transform(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        self.update_transform()
        if ax is None:
            ax = make_3d_axis(ax_s=1.0, unit='m', n_ticks=5)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 1])
        self.transform_manager.plot_frames_in('base_link', ax=ax, s=0.05, show_name=False)
        self.transform_manager.plot_visuals('base_link', ax=ax)
        return ax


def test():
    with RealManManipulator(debug=True) as manipulator:
        manipulator.set_hand_angle([1, 1, 1, 1, 1, 1])
        manipulator.set_hand_force(1)
        manipulator.set_hand_speed(1)
        manipulator.set_hand_angle([1, 1, 1, 1, 1, 1])

        manipulator.set_force_sensor()

        manipulator.move_j([-math.pi / 6, 1.2, 0.0, math.pi - 1.2, 0, -math.pi / 2, math.pi / 2], 0.2)

        pose = manipulator.get_current_state().pose

        pose[0] += 0.1
        pose[1] += 0.1

        manipulator.set_force_position(1, 0, 2, 10)

        time.sleep(2)

        manipulator.move_l(pose, 0.2)

        manipulator.stop_force_position()

        manipulator.plot_transform()
        plt.show()


if __name__ == '__main__':
    test()
