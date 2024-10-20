from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class FinalFantasyAdventure(Env):
    BOSS_HP_ADDRS = [0xD3F4, 0xD3F5]
    POWER_EQUIPMENT_BAG_ADDRS = [0xD6B3, 0xD6BE]  # 12 addresses (power value for equipment bag, one byte per slot. They are not used for weapons!)
    ITEM_BAG_ADDRS = [0xD6C5, 0xD6D4]  # 16 addresses
    EQUIPMENT_BAG_ADDRS = [0xD6DD, 0xD6E8]
    HERO_NAME_ADDRS = [0xD79D, 0xD7A0]  # 4 addresses
    HEROIN_NAME_ADDRS = [0xD7A2, 0xD7A5]  # 4 addresses
    EXPERIENCE_NEEDED_LEVEL_UP_ADDRS = [0xD7A9, 0xD7AF]  # 7 addresses Binary coded decimal. Maximum value shown, to level 99, is 0x00090909040105. Seems the limit is at 0x01000408050705.
    CURRENT_HP_ADDRS = [0xD7B2, 0xD7B3]
    MAX_HP_ADDRS = [0xD7B4, 0xD7B5]  # Limited at 0xE703 (999) (little endian)
    CURRENT_MP_ADDRS = [0xD7B6, 0xD7B7]
    MAX_MP_ADDRS = [0xD7B8, 0xD7B9]  # Limited at 0x6200 (98)  (little endian)
    LEVEL_ADDR = 0xD7BA  # Limited at 0x63 (99), however the game lets you level up again!
    EXPERIENCE_ADDRS = [0xD7BB, 0xD7BD]  # 3 addresses Limited at 0xFFFF0F (1048575). It will lock the game in a constant level up loop, if you reach this value!
    LUCRE_ADDRS = [0xD7BE, 0xD7BF]  # limited at 0xFFFF (65535)
    STAMINA_ADDR = 0xD7C1  # Limited at 0x63 (99)
    POWER_ADDR = 0xD7C2  # Limited at 0x63 (99)
    WISDOM_ADDR = 0xD7C3  # Limited at 0x63 (99)
    WILL_ADDR = 0xD7C4  # Limited at 0x63 (99)
    ATTACK_POWER_ADDR = 0xD7DF  # Limited at 0xFF (255)
    DEFENSE_POWER_ADDR = 0xD7E0  # Limited at 0xFF (255)
    DEATHBLOW_GAUGE_ADDR = 0xD858  # Limited at 0x40, higher values are brought to this mark, except 0xFF that resets to 0.

    def __init__(self, start_button: bool = False, window_type: str = 'headless', save_path: str | None = None,
                 load_path: str | None = None, all_actions: bool = False, subtask: str | None = None, reach_level_subtask: int = 0):
        assert window_type == 'SDL2' or window_type == 'headless'
        assert subtask is None or subtask in ['initial_boss_battle', 'reach_level']
        super().__init__()

        self.window_type = window_type
        with importlib.resources.path('gle.rom', "Final Fantasy Adventure (USA).gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
                window_type=self.window_type
            )

        self.pyboy.set_emulation_speed(6)
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
                WindowEvent.PASS
            ]

            self.release_actions = [
                WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.RELEASE_ARROW_RIGHT,
                WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PASS
            ]

        if start_button and not all_actions:
            self.actions += [WindowEvent.PRESS_BUTTON_START]
            self.release_actions += [WindowEvent.RELEASE_BUTTON_START]

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

        self.screen = self.pyboy.botsupport_manager().screen()

        self.subtask = subtask
        if self.subtask == 'initial_boss_battle':
            self.prev_boss_hp = 30  # initial boss life
        elif self.subtask == 'reach_level':
            self.current_level = 1

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

        done = True if info['hp'] == 0 else False

        reward = 0.0
        if self.subtask == 'initial_boss_battle':
            if info['boss_hp'] == 16711935:  # boss defeated
                done = True
                reward = 20
            else:
                reward = self.prev_boss_hp - info['boss_hp']
                self.prev_boss_hp = info['boss_hp']
        elif self.subtask == 'reach_level':
            if info['level'] == self.current_level:
                done = True
                reward = 20
            else:
                reward = info['level'] - self.current_level
                self.current_level = info['level']

        return obs, reward, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        if self.subtask == 'initial_boss_battle':
            self.prev_boss_hp = 30
        elif self.subtask == 'reach_level':
            self.current_level = 1
        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.rom', "Final Fantasy Adventure (USA).gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
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

    #   ******************************************************
    #              UTILITY FUNCTIONS USED IN OVERRIDING
    #   ******************************************************
    def take_action(self, action_idx: int):
        self.pyboy.send_input(self.actions[action_idx])
        for i in range(15):
            if i == 4:
                self.pyboy.send_input(self.release_actions[action_idx])
            self.pyboy.tick()

    def get_info(self) -> dict:
        info = dict()
        info['hero_name'] = self.decode_name(self.HERO_NAME_ADDRS)
        info['heroin_name'] = self.decode_name(self.HEROIN_NAME_ADDRS)
        info['hp'] = int.from_bytes(self.pyboy.get_memory_value(self.CURRENT_HP_ADDRS[1]).to_bytes(2, byteorder='big')
                                    + self.pyboy.get_memory_value(self.CURRENT_HP_ADDRS[0]).to_bytes(2, byteorder='big'),
                                    byteorder='big')
        info['max_hp'] = int.from_bytes(self.pyboy.get_memory_value(self.MAX_HP_ADDRS[1]).to_bytes(2, byteorder='big')
                                        + self.pyboy.get_memory_value(self.MAX_HP_ADDRS[0]).to_bytes(2, byteorder='big'),
                                        byteorder='big')
        info['mp'] = int.from_bytes(self.pyboy.get_memory_value(self.CURRENT_MP_ADDRS[1]).to_bytes(2, byteorder='big')
                                    + self.pyboy.get_memory_value(self.CURRENT_MP_ADDRS[0]).to_bytes(2, byteorder='big'),
                                    byteorder='big')
        info['max_mp'] = int.from_bytes(self.pyboy.get_memory_value(self.MAX_MP_ADDRS[1]).to_bytes(2, byteorder='big')
                                        + self.pyboy.get_memory_value(self.MAX_MP_ADDRS[0]).to_bytes(2, byteorder='big'),
                                        byteorder='big')
        info['level'] = self.pyboy.get_memory_value(self.LEVEL_ADDR)
        info['stamina'] = self.pyboy.get_memory_value(self.STAMINA_ADDR)
        info['power'] = self.pyboy.get_memory_value(self.POWER_ADDR)
        info['wisdom'] = self.pyboy.get_memory_value(self.WISDOM_ADDR)
        info['will'] = self.pyboy.get_memory_value(self.WILL_ADDR)
        info['atk'] = self.pyboy.get_memory_value(self.ATTACK_POWER_ADDR)
        info['def'] = self.pyboy.get_memory_value(self.DEFENSE_POWER_ADDR)
        info['lucre'] = int.from_bytes(self.pyboy.get_memory_value(self.LUCRE_ADDRS[1]).to_bytes(2, byteorder='big')
                                       + self.pyboy.get_memory_value(self.LUCRE_ADDRS[0]).to_bytes(2, byteorder='big'),
                                       byteorder='big')
        info['deathblow_gauge'] = self.decode_deathblow()
        info['exp'] = self.decode_exp(self.EXPERIENCE_ADDRS)
        info['needed_exp'] = self.decode_needed_exp(self.EXPERIENCE_NEEDED_LEVEL_UP_ADDRS)
        info['boss_hp'] = int.from_bytes(self.pyboy.get_memory_value(self.BOSS_HP_ADDRS[1]).to_bytes(2, byteorder='big')
                                         + self.pyboy.get_memory_value(self.BOSS_HP_ADDRS[0]).to_bytes(2, byteorder='big'),
                                         byteorder='big')
        info['item'] = self.decode_item_bag()
        info['equipment'] = self.decode_equipment()
        return info

    def get_reward(self, boss_hp: int) -> int:
        return 0
    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def decode_name(self, addrs) -> str:
        name = str()
        for addr in range(addrs[0], addrs[1] + 1):
            value = self.pyboy.get_memory_value(addr)
            if value == 0x00:
                continue
            name += chr(value - 0x79)
        return name

    def decode_deathblow(self) -> int:
        value = self.pyboy.get_memory_value(self.DEATHBLOW_GAUGE_ADDR)
        if value == 0xFF:
            value = 0x00
        elif value > 0x40:
            value = 0x40
        return value

    def decode_exp(self, addrs: list) -> int:
        exp_bytes = bytes()
        for addr in range(addrs[0], addrs[1] + 1)[::-1]:
            exp_bytes += self.pyboy.get_memory_value(addr).to_bytes(2, byteorder='big')
        return int.from_bytes(exp_bytes, byteorder='big')

    def decode_needed_exp(self, addrs: list) -> int:
        exp_str = str()
        for addr in range(addrs[0], addrs[1] + 1)[::-1]:
            exp_str += str(self.read_bcd(self.pyboy.get_memory_value(addr)))
        return int(exp_str)

    def decode_item_bag(self) -> dict:
        item_info = dict()
        for i, addr in enumerate(range(self.ITEM_BAG_ADDRS[0], self.ITEM_BAG_ADDRS[1] + 1)):
            item_info[f'item{i}'] = self.pyboy.get_memory_value(addr)
        return item_info

    def decode_equipment(self) -> dict:
        equip_info = dict()
        for addr_id, addr_qt in zip(range(self.EQUIPMENT_BAG_ADDRS[0], self.EQUIPMENT_BAG_ADDRS[1] + 1),
                                    range(self.POWER_EQUIPMENT_BAG_ADDRS[0], self.POWER_EQUIPMENT_BAG_ADDRS[1] + 1)):
            id = self.pyboy.get_memory_value(addr_id)
            if id != 0:
                equip_info[f'equip_{id}'] = self.pyboy.get_memory_value(addr_qt)
        return equip_info

    def read_bcd(self, value):
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
