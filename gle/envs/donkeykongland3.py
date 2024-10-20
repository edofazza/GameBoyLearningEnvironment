from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class DonkeyKongLand3(Env):
    BONUS_TIMER_ADDR = 0xC5EE
    CURRENT_CHARACTER_ADDR = 0xDF00
    #   00 = Kiddy Kong
    #   01 = Dixie Kong
    #   02 = Ellie
    #   03 = Squawks
    #   04 = Enguarde
    #   05 = Squitter
    #   06 = Rattly  (glitched, simply leftover from DKL2)
    #   07 = Toboggan
    INVINCIBILITY_TIMER_ADDR = 0xDF30
    KONG_X_POS = [0xDF05, 0xDF06]   # big endian
    KONG_Y_POS = [0xDF07, 0xDF08]  # big endian

    # STAGE DATA
    CURRENT_STAGE_TYPE_ADDR = 0xC5EC
    #   00 = Find the Token!
    #   01 = Collect the Stars!
    #   02 = Bash the Baddies!
    #   03 = Warp
    #   FE = Regular stage (after exiting bonus stage)
    #   FF = Regular stage (from world map screen)
    BONUS_STAGE_TIME_COUNTER_ADDR = 0xC5ED
    BONUS_STAGE_COUNTER_ADDR = 0xC5EE
    HORIZONTAL_VELOCITY_ADDR = 0xDF10
    VERTICAL_VELOCITY_ADDR = 0xDF11

    # HRAM
    CURRENT_LEVEL_ADDR = 0xFFA6
    #    00 - Seabed Shanty
    #    01 - Coral Quarrel
    #    02 - Deep Reef Grief
    #    03 - Total Rekoil
    #    04 - Liftshaft Lottery
    #    05 - Miller Instinct
    #    06 - Koco Channel
    #    07 - Riverbank Riot
    #    08 - Surface Tension
    #    09 - Black Ice Blitz
    #    0A - Polar Pitfalls
    #    0B - Tundra Blunda
    #    0C - Red Wharf
    #    0D - Ford Knocks
    #    0E - Jetty Jitters
    #    0F - Minky Mischief
    #    10 - Redwood Rampage
    #    11 - Simian Shimmy
    #    12 - Vertigo Verge
    #    13 - Rockface Chase
    #    14 - Clifftop Critters
    #    15 - Rocketeer Rally
    #    16 - Footloose Falls
    #    17 - Rickety Rapids
    #    18 - Stalagmite Frights
    #    19 - Haunted Hollows
    #    1A - Ghoulish Grotto
    #    1B - Jungle Jeopardy
    #    1C - Tropical Tightropes
    #    1D - Rainforest Rumble
    #    1E - Karbine Kaos
    #    1F - Bazuka Bombard
    #    20 - Kuchuka Karnage
    #    21 - Barrel Boulevard
    #    22 - Ugly Ducting
    #    23 - Whiplash Dash
    #    24 - Barbos Bastion
    #    25 - Arich Attack
    #    26 - Krazy Kaos
    #    27 - Bleak Magic
    #    28 - K. Rool Duel
    #    29 - K. Rool's Last Stand
    LIVES_ADDR = 0xFFAA
    BANANAS_ADDR = 0xFFAB
    CURRENT_KONG_ADDR = 0xFFAC  # 00=Diddy, 01=Dixie
    NUMBER_HITS_REMAINING_ADDR = 0xFFAD
    CURRENT_WORLD_IN_WORLD_MAP_ADDR = 0xFFAE
    #   00 = Cape Codswallop
    #   01 = Primate Plains
    #   02 = Blackforest Plateau
    #   03 = Great Ape Lakes
    #   04 = Tin Can Valley
    #   05 = The Lost World
    #   06 = Northern Kremisphere
    CURRENT_LEVEL_IN_WORLD_MAP_ADDR = 0xFFAF
    HOURS_ADDR = 0xFFB0
    MINUTES_ADDR = 0xFFB1
    SECONDS_ADDR = 0xFFB2
    BEAR_COINS_ADDR = 0xFFB3
    BONUS_COINS_ADDR = 0xFFB4
    DK_COINS_ADDR = 0xFFB5
    WATCHES_ADDR = 0xFFB6
    BOSS_STAGE_FLAG = 0xFFDF    # 00 regular stage  01 boss stage
    CURRENT_LEVEL_BITFLAG_ADDR = 0xFFE9
    #   0x80 = Cleared with Dixie for the first time
    #   0x40 = Both Bonus Coins have been found
    #   0x10 = Bonus Coin #2 collected
    #   0x08 = Bonus Coin #1 collected
    #   0x02 = DK Coin collected
    #   0x01 = Level cleared

    # SAVED DATA (1)
    LEVEL_DATA_ADDRS = [0xA01C, 0xA03F]     # same of level bitflag
    BOSS_LEVEL_DATA_ADDRS = [0xA040, 0xA045]  # bit 6: if bonus coin is collected, bit 1: cleared

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.prev_action_idx = None
        self.prev_distance = None
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        with importlib.resources.path('gle.roms', "Donkey Kong Land III (U) [S][!].gb") as rom_path:
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

        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True

        obs = self.render()
        info = self.get_info()

        if info['lives'] == 4:
            done = True

        return obs, self.get_reward(info['x_pos']), done, False, info     # TODO: reset reward to x_pose

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.prev_distance = None
        self.actions_taken = 0

        if self.load_path is None:
            self.skip_game_initial_video()

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.roms', "Donkey Kong Land III (U) [S][!].gb") as rom_path:
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
            for _ in range(12):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(2):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(5):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(4):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(3):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(3):
                self.take_action2(6)
            self.take_action2(0)
            for _ in range(4):
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
        return {**self.get_general_info(),
                **self.get_hram_info(),
                **self.get_saved_data1()
                }

    def get_reward(self, distance) -> int:
        if self.prev_distance is None:
            self.prev_distance = distance
            return 0
        else:
            reward = distance - self.prev_distance
            self.prev_distance = distance
            return reward
    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_current_character(self) -> str:
        names = ['Kiddy Kong', 'Dixie Kong', 'Ellie', 'Squawks', 'Enguarde', 'Squitter', 'Rattly', 'Toboggan']
        return names[self.pyboy.get_memory_value(self.CURRENT_CHARACTER_ADDR)]

    def get_general_info(self) -> dict:
        info = dict()
        info['character'] = self.get_current_character()
        info['x_pos'] = (self.pyboy.get_memory_value(self.KONG_X_POS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.KONG_X_POS[1]))
        info['y_pos'] = (self.pyboy.get_memory_value(self.KONG_Y_POS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.KONG_Y_POS[1]))
        info['invincibility_timer'] = self.pyboy.get_memory_value(self.INVINCIBILITY_TIMER_ADDR)
        info['stage_type'] = self.pyboy.get_memory_value(self.CURRENT_STAGE_TYPE_ADDR)
        info['bonus_timer'] = self.pyboy.get_memory_value(self.BONUS_TIMER_ADDR)
        info['bonus_timer_counter'] = self.pyboy.get_memory_value(self.BONUS_STAGE_COUNTER_ADDR)
        info['bonus_counter'] = self.pyboy.get_memory_value(self.BONUS_STAGE_COUNTER_ADDR)
        return info

    def get_hram_info(self) -> dict:
        info = dict()
        info['level'] = self.pyboy.get_memory_value(self.CURRENT_LEVEL_ADDR)
        info['lives'] = self.pyboy.get_memory_value(self.LIVES_ADDR)
        info['bananas'] = self.pyboy.get_memory_value(self.BANANAS_ADDR)
        info['current_kong'] = 'Diddy' if self.pyboy.get_memory_value(self.CURRENT_KONG_ADDR) == 0x00 else 'Dixie'
        info['hits_remaining'] = self.pyboy.get_memory_value(self.NUMBER_HITS_REMAINING_ADDR)
        info['world'] = self.pyboy.get_memory_value(self.CURRENT_WORLD_IN_WORLD_MAP_ADDR)
        info['level_world_map'] = self.pyboy.get_memory_value(self.CURRENT_LEVEL_IN_WORLD_MAP_ADDR)
        info['hours'] = self.pyboy.get_memory_value(self.HOURS_ADDR)
        info['minutes'] = self.pyboy.get_memory_value(self.MINUTES_ADDR)
        info['seconds'] = self.pyboy.get_memory_value(self.SECONDS_ADDR)
        info['bear_coins'] = self.pyboy.get_memory_value(self.BEAR_COINS_ADDR)
        info['dk_coins'] = self.pyboy.get_memory_value(self.DK_COINS_ADDR)
        info['watches'] = self.pyboy.get_memory_value(self.WATCHES_ADDR)
        info['boss_stage'] = self.pyboy.get_memory_value(self.BOSS_STAGE_FLAG) == 0x01
        info['level_bitflags'] = self.decode_level_bitflag(self.CURRENT_LEVEL_BITFLAG_ADDR)
        return info

    def get_saved_data1(self) -> dict:
        saved_info = dict()
        saved_info['levels_data'] = self.decode_level_data()
        saved_info['bosses_data'] = self.decode_boss_data()
        return saved_info

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def decode_boss_data(self) -> dict:
        boss_info = dict()
        for i, addr in enumerate(range(self.BOSS_LEVEL_DATA_ADDRS[0], self.BOSS_LEVEL_DATA_ADDRS[1] + 1)):
            tmp = dict()
            value = self.pyboy.get_memory_value(addr)
            tmp['cleared'] = value & (1 << 0) != 0
            tmp['bonus_coin'] = value & (1 << 6) != 0
            boss_info[f'boss{i}'] = tmp
        return boss_info

    def decode_level_data(self) -> dict:
        level_info = dict()
        for i, addr in enumerate(range(self.LEVEL_DATA_ADDRS[0], self.LEVEL_DATA_ADDRS[1] + 1)):
            level_info[f'level{i}'] = self.decode_level_bitflag(addr)
        return level_info

    def decode_level_bitflag(self, addr) -> dict:
        tmp = dict()
        value = self.pyboy.get_memory_value(addr)
        tmp['cleared'] = value & (1 << 0) != 0
        tmp['both_coins'] = value & (1 << 1) != 0
        tmp['bonus_coin1'] = value & (1 << 3) != 0
        tmp['bonus_coin2'] = value & (1 << 4) != 0
        tmp['all_kremkoins_found'] = value & (1 << 6) != 0
        tmp['dixie'] = value & (1 << 7) != 0
        return tmp
