from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class KirbysDreamLand(Env):
    GAME_STATE_ADDR = 0xD02C    # $01 = normal, $05 = drinking a bottle, $06 = warpstar or dying...
    X_POS_ON_SCREEN_ADDR = 0xD05C
    Y_POS_ON_SCREEN_ADDR = 0xD05D
    INHALE_TIMER_ADDR = 0xD066
    SCORE_10000S_PLACE_ADDR = 0xD070
    SCORE_1000S_PLACE_ADDR = 0xD071
    SCORE_100S_PLACE_ADDR = 0xD072
    SCORE_10S_PLACE_ADDR = 0xD073
    HP_ADDR = 0xD086
    LIVES_ADDR = 0xD089
    BOSS_HP_ADDR = 0xD093

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False, subtask: str | None = None,
                 return_sound: bool = False,):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.prev_score = 0
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type
        # Sound
        self.return_sound = return_sound

        with importlib.resources.path('gle.roms', "Kirby's Dream Land (USA, Europe).gb") as rom_path:
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

        self.subtask = subtask
        if self.subtask == 'boss_battle':
            self.prev_boss_hp = 0  # initial boss life
            self.initial_boss_hp = 0
        else:
            self.prev_score = 0
            self.original_score = 0

        _, info = self.reset()
        if self.subtask == 'boss_battle':
            self.boss_hp = info['boss_hp']
            self.boss_hp_original = info['boss_hp']
        else:
            self.prev_score = info['score']
            self.original_score = info['score']

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

        if self.subtask == 'boss_battle':
            reward = self.prev_boss_hp - info['boss_hp']
            self.prev_boss_hp = info['boss_hp']
            done = self.prev_boss_hp == 0 or done
        else:
            reward = info['score'] - self.prev_score
            self.prev_score = info['score']

        if self.return_sound:
            return obs, self.screen.sound, reward, done, False, info
        else:
            return obs, reward, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        if self.subtask == 'boss_battle':
            self.prev_boss_hp = self.initial_boss_hp
        else:
            self.prev_score = self.original_score
        self.actions_taken = 0

        if self.load_path is None:
            self.skip_game_initial_video()

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.roms', "Kirby's Dream Land (USA, Europe).gb") as rom_path:
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
            if self.pyboy.frame_count == 180:
                self.take_action2(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
                self.take_action2(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
                self.take_action2(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
                self.take_action2(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
                self.take_action2(WindowEvent.PASS, WindowEvent.PASS)
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

    def take_action2(self, action: WindowEvent, release: WindowEvent):
        self.pyboy.send_input(action)
        for i in range(24):
            if i == 8:
                self.pyboy.send_input(release)
            self.pyboy.tick()

    def get_info(self) -> dict:
        info = dict()
        info['hp'] = self.pyboy.get_memory_value(self.HP_ADDR)
        info['lives'] = self.pyboy.get_memory_value(self.LIVES_ADDR)
        info['score'] = self.get_score()
        info['state'] = self.pyboy.get_memory_value(self.GAME_STATE_ADDR)
        info['x_pos'] = self.pyboy.get_memory_value(self.X_POS_ON_SCREEN_ADDR)
        info['y_pos'] = self.pyboy.get_memory_value(self.Y_POS_ON_SCREEN_ADDR)
        info['inhale_timer'] = self.pyboy.get_memory_value(self.INHALE_TIMER_ADDR)
        info['boss_hp'] = self.pyboy.get_memory_value(self.BOSS_HP_ADDR)
        return info

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_score(self) -> int:
        return (10000 * self.read_bcd(self.pyboy.get_memory_value(self.SCORE_10000S_PLACE_ADDR))
                + 1000 * self.read_bcd(self.pyboy.get_memory_value(self.SCORE_1000S_PLACE_ADDR))
                + 100 * self.read_bcd(self.pyboy.get_memory_value(self.SCORE_100S_PLACE_ADDR))
                + 10 * self.read_bcd(self.pyboy.get_memory_value(self.SCORE_10S_PLACE_ADDR)))

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_bcd(self, value):
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
