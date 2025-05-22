from typing import List
import numpy as np
import pickle
import os


class Record(object):
    def __init__(self, observation: np.array, action: int, info: dict):
        self.observation = observation
        self.action = action
        self.info = info


class GamePlay(object):
    def __init__(self, path: str, name: str):
        """

        :param path: directory to where the game will be saved or is stored
        :param name: name of the binary file (no extension)
        """
        self.path: str = path
        self.name: str = name
        self.last_update: str | None = None
        self.trajectory: List[Record] = list()
        if os.path.isfile(f'{self.path}/{self.name}.bin'):
            self.load(f'{self.path}/gameplay.bin')

    def update(self, new_record: Record, time: str):
        self.trajectory.append(new_record)
        self.last_update = time

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            gameplay = pickle.load(f)
            self.path = gameplay.path
            self.last_update = gameplay.last_update
            self.trajectory = gameplay.trajectory
