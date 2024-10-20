from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces

ALL_ACTIONS = [
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT,
            WindowEvent.PASS,
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
            [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],

            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_DOWN],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_UP],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_DOWN],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_RIGHT],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_LEFT],

            [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_UP],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_UP],

            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_DOWN],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_DOWN],

            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_RIGHT],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT],

            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]
        ]

ALL_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_START,
    WindowEvent.RELEASE_BUTTON_SELECT,
    WindowEvent.PASS,
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_UP],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_DOWN],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_RIGHT],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_LEFT],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_UP],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_DOWN],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_RIGHT],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_LEFT],
    [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_UP],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_UP],
    [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_DOWN],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_DOWN],
    [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_RIGHT],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_RIGHT],
    [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B],
    [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_LEFT],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A],
    [WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_LEFT]
]


class GeneralGameBoyEnv(Env):
    def __init__(self, rom_path: str, window_type: str = 'headless', save_path: str | None = None,
                 load_path: str | None = None, max_actions: int | None = None,
                 ticks_between_actions: int = 5):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type
        self.rom_path = rom_path
        self.ticks_between_actions = ticks_between_actions

        self.pyboy = PyBoy(
            self.rom_path,
            window_type=self.window_type
        )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title()}')

        self.actions = ALL_ACTIONS

        self.release_actions = ALL_RELEASE_ACTIONS

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

        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True

        return obs, 0, done, False, dict()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.actions_taken = 0

        return self.render(), dict()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(
            self.rom_path,
            window_type=self.window_type
        )
        if self.load_path is not None:
            self.load()
        self.screen = self.pyboy.botsupport_manager().screen()

    #   ******************************************************
    #                  SAVE AND LOAD FUNCTIONS
    #   ******************************************************
    def save(self) -> None:
        with open(self.save_path, "wb") as f:
            self.pyboy.save_state(f)

    def load(self) -> None:
        with open(self.load_path, "rb") as f:
            self.pyboy.load_state(f)

    def take_action(self, action_idx: int):
        if self.prev_action_idx is None or self.prev_action_idx == action_idx:
            self.prev_action_idx = action_idx
            selected_action = self.actions[self.prev_action_idx]
            if isinstance(selected_action, list):
                for action in selected_action:
                    self.pyboy.send_input(action)
                    for _ in range(self.ticks_between_actions):
                        self.pyboy.tick()
            else:
                self.pyboy.send_input(selected_action)
                for _ in range(self.ticks_between_actions * 2):
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
                    for _ in range(self.ticks_between_actions):
                        self.pyboy.tick()
            else:
                self.pyboy.send_input(selected_action)
                for _ in range(self.ticks_between_actions * 2):
                    self.pyboy.tick()
