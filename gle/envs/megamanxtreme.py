from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
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

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        with importlib.resources.path('gle.roms', "Mega Man Xtreme (U) [C][!].gbc") as rom_path:
            self.pyboy = PyBoy(
                str(rom_path),
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title()}')

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

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

        self.screen = self.pyboy.botsupport_manager().screen()

        self.reset()

    #   ******************************************************
    #               GYMNASIUM OVERRIDING FUNCTION
    #   ******************************************************
    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.take_action(action)
        obs = self.render()
        info = self.get_info()

        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True
        if info['lives'] == 0:
            done = True

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
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
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
        self.screen = self.pyboy.botsupport_manager().screen()

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
                    for _ in range(5):
                        self.pyboy.tick()
            else:
                self.pyboy.send_input(selected_action)
                for _ in range(10):
                    self.pyboy.tick()
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
                    for _ in range(5):
                        self.pyboy.tick()
            else:
                self.pyboy.send_input(selected_action)
                for _ in range(10):
                    self.pyboy.tick()

    def take_action2(self, action_idx: int):
        self.pyboy.send_input(self.actions[action_idx])
        for i in range(24):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action_idx])
            self.pyboy.tick()

    def get_info(self) -> dict:
        info = dict()
        info['health'] = self.pyboy.get_memory_value(self.CHARACTER_HEALTH_ADDR)
        info['lives'] = self.pyboy.get_memory_value(self.CHARACTER_LIVES_ADDR)
        info['x_pos'] = self.pyboy.get_memory_value(self.CHARACTER_X_POS_ADDR)
        info['x_pos_subpixel'] = self.pyboy.get_memory_value(self.CHARACTER_X_POS_SUBPIXEL_ADDR)
        info['y_pos'] = self.pyboy.get_memory_value(self.CHARACTER_Y_POS_ADDR)
        info['y_pos_subpixel'] = self.pyboy.get_memory_value(self.CHARACTER_X_POS_SUBPIXEL_ADDR)
        info['capacity'] = self.pyboy.get_memory_value(self.CHARACTER_CAPACITY_ADDR)
        info['subtank_health'] = self.pyboy.get_memory_value(self.SUBTANK_HEALTH_ADDR)
        info['boss_health'] = self.pyboy.get_memory_value(self.BOSS_HEALTH_ADDR)
        info['enemy1_active'] = self.pyboy.get_memory_value(self.ENEMY1_ACTIVE_ADDR)
        info['enemy1_x_pos'] = self.pyboy.get_memory_value(self.X_POS_ENEMY1_ADDR)
        info['enemy1_y_pos'] = self.pyboy.get_memory_value(self.Y_POS_ENEMY1_ADDR)
        info['frames'] = self.pyboy.get_memory_value(self.FRAME_ADDR)
        info['hours'] = self.pyboy.get_memory_value(self.HOURS_ADDR)
        info['minutes'] = self.pyboy.get_memory_value(self.MINUTES_ADDR)
        info['seconds'] = self.pyboy.get_memory_value(self.SECONDS_ADDR)
        return info
