from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces


from general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class DonkeyKongLand(Env):
    CURRENT_CHARACTER_ADDR = 0xC64E     # 00 = Donkey Kong  01 = Diddy Kong
    LIVES_ADDR = 0xC66D
    LOCATION_ADDR = 0xC29A
    """
        0x00: Freezing Fun
        0x01: Jungle Jaunt
        0x02: Congo Carnage
        0x03: Tricky Temple
        0x04: Reef Rampage
        0x05: Riggin' Rumble
        0x06: Nautilus Chase
        0x07: Snake Charmer's Challenge
        0x08: Swirlwind Storm
        0x09: Simian Swing
        0x0A: Rope Ravine
        0x0B: Deck Trek
        0x0C: Button Barrel Blast bonus #2
        0x0D: Tyre Trail
        0x0E: File select menu
        0x0F: Kong Krazy
        0x10: Kremlantis
        0x11: Balloon Barrage (mislabeled "Construction Site Fight" in the manual)
        0x12: Pot Hole Panic
        0x13: Mountain Mayhem
        0x14: Skyscraper Caper
        0x15: Construction Site Fight (mislabeled "Balloon Barrage" in the manual)
        0x16: Track Attack
        0x17: Sky High Caper
        0x18: Arctic Barrel Arsenal
        0x19: Fast Barrel Blast
        0x1A: Collapsing Clouds
        0x1B: Oil Drum Slum
        0x1C: Landslide Leap
        0x1D: Chomp's Coliseum
        0x1E: Button Barrel Blast
        0x1F: Spiky Tyre Trail
        0x20: Mad Mole Holes
        0x21: Freezing Fun bonus
        0x22: Fast Barrel Blast bonus #2
        0x23: Deck Trek bonus #2
        0x24: Riggin' Rumble bonus #2
        0x25: Tyre Trail bonus #2
        0x26: Simian Swing bonus #1
        0x27: Tyre Trail bonus #3
        0x28: Jungle Jaunt bonus #1
        0x29: Congo Carnage bonus #1
        0x2A: Mountain Mayhem bonus #1
        0x2B: Wild Sting Fling
        0x2C: Tricky Temple bonus #1
        0x2D: Swirlwind Storm bonus #1
        0x2E: Arctic Barrel Arsenal bonus #2
        0x2F: Deck Trek bonus #1
        0x30: Balloon Barrage bonus #2
        0x31: Snake Charmer's Challenge bonus #1
        0x32: Balloon Barrage bonus #1
        0x33: Jungle Jaunt bonus #2
        0x34: Rope Ravine bonus #2
        0x35: Oil Drum Slum bonus #1
        0x36: Sky High Caper bonus #1
        0x37: Seabed Showdown
        0x38: Kong Token bonuses (shared by multiple levels):
                * Simian Swing bonus #2
                * Rope Ravine bonus #1
                * Tyre Trail bonus #1
                * Congo Carnage bonus #2
                * Arctic Barrel Arsenal bonus #1
                * Spiky Tyre Trail bonus #2
                * Collapsing Clouds bonus #2
                * Fast Barrel Blast bonus #1
        0x39: K. Rool's Kingdom
        0x3A: Pot Hole Panic bonus #1
        0x3B: Track Attack bonus #2
        0x3C: Landslide Leap bonus #2
        0x3D: Riggin' Rumble bonus #1
        0x3E: Construction Site Fight bonus #2
        0x3F: Spiky Tyre Trail bonus #1
        0x40: Track Attack bonus #1
        0x41: Pot Hole Panic bonus #2
        0x42: Kong Krazy bonus
        0x43: Skyscraper Caper bonus
        0x44: Mountain Mayhem bonus #2
        0x45: Construction Site Fight bonus #1
        0x46: Collapsing Clouds bonus #1
        0x47: Oil Drum Slum bonus #2
        0x48: Button Barrel Blast bonus #1
        0x49: Landslide Leap bonus #1
    """
    BONUS_EXIT_DESTINATION_ADDR = 0xCAAC    # Note that you should add this value by 0x01 to get the proper value in the list. Also, this value is 0x00 while not in a bonus stage.

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        self.pyboy = PyBoy(
            "roms/Donkey Kong Land (U) [S][!].gb",
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
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B]
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
                [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A],
                [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_B],
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
        self.pyboy = PyBoy(
            "roms/Donkey Kong Land (U) [S][!].gb",
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
            if self.pyboy.frame_count == 420:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
                for _ in range(5):
                    self.pyboy.tick()
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                self.take_action2(0)
                for _ in range(4):
                    self.take_action2(6)
                self.take_action2(0)
                for _ in range(12):
                    self.take_action2(6)
                self.take_action2(0)
                for _ in range(4):
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
        for i in range(15):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action_idx])
            self.pyboy.tick()

    def get_info(self) -> dict:
        info = dict()
        info['character'] = 'Donkey Kong' if self.pyboy.get_memory_value(self.CURRENT_CHARACTER_ADDR) == 0x00 else 'Diddy Kong'
        info['lives'] = self.pyboy.get_memory_value(self.LIVES_ADDR)
        info['location'] = self.pyboy.get_memory_value(self.LOCATION_ADDR)
        info['bonus_exit_dest'] = self.pyboy.get_memory_value(self.BONUS_EXIT_DESTINATION_ADDR) + 0x01 \
            if self.pyboy.get_memory_value(self.BONUS_EXIT_DESTINATION_ADDR) != 0x00 else 0x00
        return info
