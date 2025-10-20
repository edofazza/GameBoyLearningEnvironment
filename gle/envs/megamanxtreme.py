from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class MegaManXtreme(Env):
    Y_POS_ENEMY1_ADDR = 0xB404
    X_POS_ENEMY1_ADDR = 0xB408
    ENEMY1_ACTIVE_ADDR = 0xB40D
    CHARACTER_HEALTH_ADDR = 0xCADC
    CHARACTER_LIVES_ADDR = 0xCADD
    BOSS_HEALTH_ADDR = 0xCAE1
    CHARACTER_CAPACITY_ADDR = 0xCAE2
    CHARACTER_X_POS_ADDR = 0xCC5B
    CHARACTER_X_POS_SUBPIXEL_ADDR = 0xCC5A
    CHARACTER_Y_POS_ADDR = 0xCC5E
    CHARACTER_Y_POS_SUBPIXEL_ADDR = 0xCC5D
    SUBTANK_HEALTH_ADDR = 0xD368    # (+80) /12
    FRAME_ADDR = 0xD39C
    SECONDS_ADDR = 0xD39D
    MINUTES_ADDR = 0xD39E
    HOURS_ADDR = 0xD39F

    def __init__(self, window_type: str = 'null', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False, return_sound: bool = False, rgba: bool = False,):
        super().__init__()
        self.prev_action_idx = None
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type
        self.rgba = rgba
        # Sound
        self.return_sound = return_sound

        with importlib.resources.path('gle.roms', "Mega Man Xtreme (U) [C][!].gbc") as rom_path:
            self.pyboy = PyBoy(
                str(rom_path),
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title}')

        if all_actions:
            self.actions = ALL_ACTIONS
            self.release_actions = ALL_RELEASE_ACTIONS
        else:
            self.actions = [
                WindowEvent.PRESS_BUTTON_A,
                WindowEvent.PRESS_BUTTON_B,
                WindowEvent.PRESS_ARROW_UP,
                WindowEvent.PRESS_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PASS,
                [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
                [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]
            ]

            self.release_actions = [
                WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.RELEASE_ARROW_RIGHT,
                WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PASS,
                [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A],
                [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B],
                [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
                [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A],
                [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B],
                [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B]
            ]

        if self.rgba:
            self.observation_space = spaces.Box(low=0, high=255, shape=(4, 144, 160), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

        self.screen = self.pyboy.screen

        self.reset()

    #   ******************************************************
    #               GYMNASIUM OVERRIDING FUNCTION
    #   ******************************************************
    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]] | tuple[ObsType, np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.take_action(action)
        obs = self.render()
        info = self.get_info()

        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True
        if info['lives'] == 0:
            done = True

        if self.return_sound:
            return obs, self.screen.sound, 1.0, done, False, info
        else:
            return obs, 1.0, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.actions_taken = 0

        if self.load_path is None:
            self.skip_game_initial_video()

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.rgba:
            screen_obs = self.screen.ndarray  # (144, 160, 4) RGBA
        else:
            screen_obs = self.screen.ndarray[:, :, :-1]  # (144, 160, 3) RGB
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.roms', "Mega Man Xtreme (U) [C][!].gbc") as rom_path:
            self.pyboy = PyBoy(
                str(rom_path),
                window_type=self.window_type
            )
        if self.load_path is not None:
            self.load()
        self.screen = self.pyboy.screen

    #   ******************************************************
    #                FUNCTION FOR MOVING IN THE GAME
    #   ******************************************************
    def skip_game_initial_video(self):
        while not self.pyboy.tick():
            if self.pyboy.frame_count == 550:
                self.take_action2(0)
                for _ in range(3):
                    self.take_action2(6)
                self.take_action2(0)
                self.take_action2(6)
                self.take_action2(0)
                for _ in range(2):
                    self.take_action2(6)
                for _ in range(5):
                    self.take_action2(0)
                    self.take_action2(6)
                for _ in range(2):
                    self.take_action2(6)
                for _ in range(3):
                    self.take_action2(0)
                for _ in range(3):
                    self.take_action2(6)
                break

    #   ******************************************************
    #                  SAVE AND LOAD FUNCTIONS
    #   ******************************************************
    def save(self) -> None:
        with open(self.save_path, "wb") as f:
            self.pyboy.save_state(f)

    def load(self) -> None:
        with open(self.load_path, "rb") as f:
            self.pyboy.load_state(f)

    #   ******************************************************
    #              UTILITY FUNCTIONS USED IN OVERRIDING
    #   ******************************************************
    def take_action(self, action_idx: int):
        if self.prev_action_idx is None or self.prev_action_idx == action_idx:
            self.prev_action_idx = action_idx
            selected_action = self.actions[self.prev_action_idx]
            if isinstance(selected_action, list):
                for action in selected_action:
                    self.pyboy.send_input(action)
                    self.pyboy.tick(5)
            else:
                self.pyboy.send_input(selected_action)
                self.pyboy.tick(10)
        else:  # different action
            # release previous actions
            old_actions_to_be_released = self.release_actions[self.prev_action_idx]
            if isinstance(old_actions_to_be_released, list):
                for action in old_actions_to_be_released[::-1]:
                    self.pyboy.send_input(action)
                    self.pyboy.tick()
            else:
                self.pyboy.send_input(old_actions_to_be_released)
                self.pyboy.tick()
            # Take new action
            self.prev_action_idx = action_idx
            selected_action = self.actions[self.prev_action_idx]
            if isinstance(selected_action, list):
                for action in selected_action:
                    self.pyboy.send_input(action)
                    self.pyboy.tick(5)
            else:
                self.pyboy.send_input(selected_action)
                self.pyboy.tick(10)

    def take_action2(self, action_idx: int):
        self.pyboy.send_input(self.actions[action_idx])
        self.pyboy.tick(7)
        self.pyboy.send_input(self.release_actions[action_idx])
        self.pyboy.tick(8)

    def get_info(self) -> dict:
        info = dict()
        info['health'] = self.pyboy.memory[self.CHARACTER_HEALTH_ADDR]
        info['lives'] = self.pyboy.memory[self.CHARACTER_LIVES_ADDR]
        info['x_pos'] = self.pyboy.memory[self.CHARACTER_X_POS_ADDR]
        info['x_pos_subpixel'] = self.pyboy.memory[self.CHARACTER_X_POS_SUBPIXEL_ADDR]
        info['y_pos'] = self.pyboy.memory[self.CHARACTER_Y_POS_ADDR]
        info['y_pos_subpixel'] = self.pyboy.memory[self.CHARACTER_X_POS_SUBPIXEL_ADDR]
        info['capacity'] = self.pyboy.memory[self.CHARACTER_CAPACITY_ADDR]
        info['subtank_health'] = self.pyboy.memory[self.SUBTANK_HEALTH_ADDR]
        info['boss_health'] = self.pyboy.memory[self.BOSS_HEALTH_ADDR]
        info['enemy1_active'] = self.pyboy.memory[self.ENEMY1_ACTIVE_ADDR]
        info['enemy1_x_pos'] = self.pyboy.memory[self.X_POS_ENEMY1_ADDR]
        info['enemy1_y_pos'] = self.pyboy.memory[self.Y_POS_ENEMY1_ADDR]
        info['frames'] = self.pyboy.memory[self.FRAME_ADDR]
        info['hours'] = self.pyboy.memory[self.HOURS_ADDR]
        info['minutes'] = self.pyboy.memory[self.MINUTES_ADDR]
        info['seconds'] = self.pyboy.memory[self.SECONDS_ADDR]
        return info
