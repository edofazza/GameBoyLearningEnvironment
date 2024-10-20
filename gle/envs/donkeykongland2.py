from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class DonkeyKongLand2(Env):
    BANANAS_ADDR = 0xDED3
    LIVES_ADDR = 0xDED4
    CURRENT_CHARACTER_ADDR = 0xDF00
    #   00 Diddy Kong
    #   01 = Dixie Kong
    #   02 = Rambi
    #   03 = Squawks
    #   04 = Enguarde
    #   05 = Squitter
    #   06 = Rattly
    #   07 = Roller Coaster
    TRACTION_ADDR = 0xDF2D  # 03 = Non-ice levels 34 = Ice levels
    HORIZONTAL_SCROLL_OFFSET_ADDRS = [0xDE80, 0xDE81]
    VERTICAL_SCROLL_OFFSET_ADDRS = [0xDE82, 0xDE83]
    HORIZONTAL_VELOCITY_ADDR = 0xDF10
    VERTICAL_VELOCITY_ADDR = 0xDF11
    CURRENT_STAGE_TYPE_ADDR = 0xC5F7
    #   00 = Find the Token!
    #   01 = Collect the Stars!
    #   02 = Destroy them All!
    #   03 = Warp
    #   FE = Regular stage (after exiting bonus stage)
    #   FF = Regular stage (from world map screen)
    BONUS_STAGE_TIMER_COUNTER_ADDR = 0xC5F8  # Bonus stage timer counter (remaining seconds)
    BONUS_STAGE_COUNTER_ADDR = 0xC5F9  # Bonus stage counter (remaining stars/Kremlings)

    # SAVED DATA (1)
    CURRENT_KONG_ADDR = 0xA002  # 00=Diddy, 01=Dixie
    NUMBER_HITS_REMAINING_ADDR = 0xA003
    LAST_WORLD_VISITED_ADDR = 0xA004
    HOURS_ADDR = 0xA006
    MINUTES_ADDR = 0xA007
    SECONDS_ADDR = 0xA008
    KREMKOINS_ADDR = 0xA00A
    DK_COINS_ADDR = 0xA00B
    WORLD_DATA_ADDRS = [0xA00E, 0xA014]
    # Bit 7: Supposed to be used to indicate the world was finished with Diddy or Dixie. Due to a bug that skips a check to determine who beats a world, this is always 0 (Diddy). Changing this bit to 1 will display that Dixie finished a world.
    # Bit 6: 0 if missing at least one Kremkoin, 1 if all Kremkoins have been found
    # Bit 5: ???
    # Bit 4: ???
    # Bit 3: Saved at Kong Kollege at least once
    # Bit 2: Used Funky's Flights at least once
    # Bit 1: 0 if DK Coin is not collected, 1 if DK Coin is collected
    # Bit 0: 0 if not cleared; 1 if cleared
    LEVEL_DATA_ADDRS = [0xA015, 0xA042]
    # Bit 7: 0 if cleared with Diddy the first time, 1 if cleared with Dixie for the first time
    # Bit 6: 0 if missing at least one Kremkoin, 1 if all Kremkoins have been found
    # Bit 5: Unused?
    # Bit 4: 0 if Bonus Coin #2 is missing, 1 if Bonus Coin #2 is collected
    # Bit 3: 0 if Bonus Coin #1 is missing, 1 if Bonus Coin #1 is collected
    # Bit 2: Unused?
    # Bit 1: 0 if DK Coin is not collected, 1 if DK Coin is collected
    # Bit 0: 0 if not cleared, 1 if cleared

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.prev_score = 0
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        with importlib.resources.path('gle.roms', "Donkey Kong Land 2 (UE) [S][!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
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

        if info['lives'] == 0:  # else 4
            done = True

        if info['bananas'] > 0:
            reward = info['bananas'] - self.prev_score
            self.prev_score = info['bananas']
        else:
            reward = 0
        return obs, reward, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.prev_score = 0
        self.actions_taken = 0

        if self.load_path is None:
            self.skip_game_initial_video()

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.roms', "Donkey Kong Land 2 (UE) [S][!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
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
            for _ in range(20):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(2):
                self.take_action2(2)
            self.take_action2(0)
            for _ in range(5):
                self.take_action2(2)
            self.take_action2(0)
            for _ in range(4):
                self.take_action2(2)
            self.take_action2(0)
            for _ in range(3):
                self.take_action2(2)
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
        info['bananas'] = self.pyboy.get_memory_value(self.BANANAS_ADDR)
        info['lives'] = self.pyboy.get_memory_value(self.LIVES_ADDR)
        info['character'] = self.get_current_character()
        info['ice_level'] = self.pyboy.get_memory_value(self.TRACTION_ADDR) == 0x34
        info['horizontal_scroll_offset'] = (self.pyboy.get_memory_value(self.HORIZONTAL_SCROLL_OFFSET_ADDRS[1]) * (2 ** 8)
                                            + self.pyboy.get_memory_value(self.HORIZONTAL_SCROLL_OFFSET_ADDRS[0]))
        info['vertical_scroll_offset'] = (self.pyboy.get_memory_value(self.VERTICAL_SCROLL_OFFSET_ADDRS[1]) * (2 ** 8)
                                          + self.pyboy.get_memory_value(self.VERTICAL_SCROLL_OFFSET_ADDRS[0]))
        info['horizontal_velocity'] = self.pyboy.get_memory_value(self.HORIZONTAL_VELOCITY_ADDR)
        info['vertical_velocity'] = self.pyboy.get_memory_value(self.VERTICAL_VELOCITY_ADDR)
        info['stage_type'] = self.pyboy.get_memory_value(self.CURRENT_STAGE_TYPE_ADDR)
        info['bonus_timer'] = self.pyboy.get_memory_value(self.BONUS_STAGE_TIMER_COUNTER_ADDR)
        info['bonus_counter'] = self.pyboy.get_memory_value(self.BONUS_STAGE_COUNTER_ADDR)
        info['saved_data1'] = self.get_saved_data1()
        return info

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_current_character(self) -> str:
        names = ['Diddy Kong', 'Dixie Kong', 'Rambi', 'Squawks', 'Enguarde', 'Squitter', 'Rattly', 'Roller Coaster']
        return names[self.pyboy.get_memory_value(self.CURRENT_CHARACTER_ADDR)]

    def get_saved_data1(self) -> dict:
        saved_info = dict()
        saved_info['current_kong'] = 'Diddy' if self.pyboy.get_memory_value(self.CURRENT_KONG_ADDR) == 0x00 else 'Dixie'
        saved_info['hits_remaining'] = self.pyboy.get_memory_value(self.NUMBER_HITS_REMAINING_ADDR)
        saved_info['last_world_visited'] = self.pyboy.get_memory_value(self.LAST_WORLD_VISITED_ADDR)
        saved_info['hours'] = self.pyboy.get_memory_value(self.HOURS_ADDR)
        saved_info['minutes'] = self.pyboy.get_memory_value(self.MINUTES_ADDR)
        saved_info['seconds'] = self.pyboy.get_memory_value(self.SECONDS_ADDR)
        saved_info['kremkoins'] = self.pyboy.get_memory_value(self.KREMKOINS_ADDR)
        saved_info['dk_coins'] = self.pyboy.get_memory_value(self.DK_COINS_ADDR)
        saved_info['world_data'] = self.decode_world_data()
        saved_info['level_data'] = self.decode_level_data()
        return saved_info

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def decode_world_data(self) -> dict:
        world_info = dict()
        for i, addr in enumerate(range(self.WORLD_DATA_ADDRS[0], self.WORLD_DATA_ADDRS[1] + 1)):
            tmp = dict()
            value = self.pyboy.get_memory_value(addr)
            tmp['cleared'] = value & (1 << 0) != 0
            tmp['dk_coin'] = value & (1 << 1) != 0
            tmp['used_funky_flights'] = value & (1 << 2) != 0
            tmp['save_at_kollege'] = value & (1 << 3) != 0
            tmp['all_kremkoins_found'] = value & (1 << 6) != 0
            tmp['dixie'] = value & (1 << 7) != 0
            world_info[f'world{i}'] = tmp
        return world_info

    def decode_level_data(self) -> dict:
        level_info = dict()
        for i, addr in enumerate(range(self.LEVEL_DATA_ADDRS[0], self.LEVEL_DATA_ADDRS[1] + 1)):
            tmp = dict()
            value = self.pyboy.get_memory_value(addr)
            tmp['cleared'] = value & (1 << 0) != 0
            tmp['dk_coin'] = value & (1 << 1) != 0
            tmp['bonus_coin1'] = value & (1 << 3) != 0
            tmp['bonus_coin2'] = value & (1 << 4) != 0
            tmp['all_kremkoins_found'] = value & (1 << 6) != 0
            tmp['dixie'] = value & (1 << 7) != 0
            level_info[f'level{i}'] = tmp
        return level_info
