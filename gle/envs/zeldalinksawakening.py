from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class ZeldaLinksAwakening(Env):
    DESTINATION_DATA1_ADDR = 0xD401     # Destination data byte 1: 00 - overworld, 01 - dungeon, 02 - side view area
    DESTINATION_DATA2_ADDR = 0xD402     # Destination data byte 2: Values from 00 to 1F accepted. FF is Color Dungeon
    ROOM_NUMBER_ADDR = 0xD403           # Must appear on map or it will lead to an empty room
    X_POS_ADDR = 0xD404
    Y_POS_ADDR = 0xD405
    WORLD_MAP_STATUS_ADDRS = [0xD800, 0xD8FF]  # 255 addresses
    #  Each screen status is represented by a byte, which is a combination of the following mask
    #       00 : Unexplored
    #       10 : changed from initial status (for example sword taken on the beach or dungeon opened with key)
    #       20 : owl talked
    #       80 : visited
    CURRENT_HELD_ITEMS_ADDRS = [0xDB00, 0xDB01]
    INVENTORY_ADDRS = [0xDB02, 0xDB0B]  # 10 addresses
    ITEM_DICT = {
        0x01: 'sword',
        0x02: 'bombs',
        0x03: 'power bracelet',
        0x04: 'shield',
        0x05: 'bow',
        0x06: 'hookshot',
        0x07: 'fire rod',
        0x08: 'pegasus boots',
        0x09: 'ocarina',
        0x0A: 'feather',
        0x0B: 'shovel',
        0x0C: 'magic powder',
        0x0D: 'boomerang'
    }
    FLIPPER_ADDR = 0xDB0C   # 0x01 have
    POTION_ADDR = 0xDB0D    # 0x01 have
    CURRENT_ITEM_TRADING_GAME_ADDR = 0xDB0E  # 01=Yoshi, 0E=magnifier
    NUMBER_SECRET_SHELLS_ADDR = 0xDB0F
    DUNGEON_ENTRANCE_KEYS_ADDRS = [0xDB10, 0xDB14]   # 5 addresses 01=have
    NUMBER_GOLD_LEAVES_ADDR = 0xDB15
    DUNGEONS_ITEM_FLAGS_ADDRS = [0xDB16, 0xDB3D]    # 40 addresses. 5 bytes for each dungeon, 5th byte is quantity of keys for that dungeon
    POWER_BRACELET_LEVEL_ADDR = 0xDB43
    SHIELD_LEVEL_ADDR = 0xDB44
    NUMBER_ARROWS_ADDR = 0xDB45
    OCARINA_SONGS_POSSESSION_ADDR = 0xDB49  # 3 bits mask, 0=no songs, 7=all songs
    OCARINA_SELECTED_SONG_ADDR = 0xDB4A
    MAGIC_POWDER_QUANTITY_ADDR = 0xDB4C
    NUMBER_BOMBS_ADDR = 0xDB4D
    SWORD_LEVEL_ADDR = 0xDB4E
    NUMBER_OF_DEATHS_SLOT1_ADDR = 0xDB56
    NUMBER_OF_DEATHS_SLOT2_ADDR = 0xDB57
    NUMBER_OF_DEATHS_SLOT3_ADDR = 0xDB58
    CURRENT_HEALTH_ADDR = 0xDB5A    # Each increment of 08h is one full heart, each increment of 04h is one-half heart.
    MAXIMUM_HEALTH_ADDR = 0xDB5B    # Simply counts the number of hearts Link has in hex. Max recommended value is 0E (14 hearts)
    NUMBER_RUPEES_ADDRS = [0xDB5D, 0xDB5E]  # for 999 put 0999
    INSTRUMENT_FOR_EVERY_DUNGEON_ADDRS = [0xDB65, 0xDB6C]    # 8 addresses 00=no instrument, 03=have instrument
    MAX_MAGIC_POWDER_ADDR = 0xDB76
    MAX_BOMBS_ADDR = 0xDB77
    MAX_ARROWS_ADDR = 0xDB78
    POS_IN8X8_DUNGEON_GRID_ADDR = 0xDBAE
    NUMBER_POSSESSED_KEYS_ADDR = 0xDBD0

    TEXT_TABLE = {
        b'\xD6': 'D-pad',
        b'\xD7': 'Letter',
        b'\xD9': 'Flower',
        b'\xDA': 'Footprint',
        b'\xDC': 'Skull',
        b'\xDD': 'Link',
        b'\xDE': 'Marin',
        b'\xDF': 'Tarin',
        b'\xE0': 'Yoshi',
        b'\xE1': 'Ribbon',
        b'\xE2': 'Dog Food',
        b'\xE3': 'Bananas',
        b'\xE4': 'Stick',
        b'\xE5': 'Honeycomb',
        b'\xE6': 'Pineapple',
        b'\xE7': 'Flower',
        b'\xE8': 'Broom',
        b'\xE9': 'Fishhook',
        b'\xEA': 'Necklace or Bra',
        b'\xEB': 'Scale',
        b'\xEC': 'Magnifying Glass',
        b'\xF0': 'UP',
        b'\xF1': 'DOWN',
        b'\xF2': 'LEFT',
        b'\xF3': 'RIGHT'
    }

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 start_button: bool = False, max_actions: int | None = None, all_actions: bool = False,
                 subtask: str | None = None
                 ):
        assert window_type == 'SDL2' or window_type == 'headless'
        assert subtask is None or subtask in ['get_shield']
        super().__init__()
        self.original_n_death = 0
        # episode truncation
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        self.subtask = subtask

        with importlib.resources.path('gle.roms', "Legend of Zelda, The - Link's Awakening (U) (V1.2) [!].gb") as rom_path:
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

        if start_button:
            self.actions += [WindowEvent.PRESS_BUTTON_START]
            self.release_actions += [WindowEvent.RELEASE_BUTTON_START]

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

        if info['deaths_slot1'] > self.original_n_death or info['shield_level']:
            done = True

        reward = info['shield_level'] if self.subtask == 'get_shield' else 0

        return obs, reward, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.actions_taken = 0

        if self.load_path is None:
            self.skip_game_initial_video()

        self.original_n_death = self.get_player_info()['deaths_slot1']
        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.roms',
                                      "Legend of Zelda, The - Link's Awakening (U) (V1.2) [!].gb") as rom_path:
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
            if self.pyboy.frame_count == 180 or self.pyboy.frame_count == 230:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            if self.pyboy.frame_count == 185 or self.pyboy.frame_count == 235:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.pyboy.frame_count == 280:
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
        self.pyboy.send_input(self.actions[action_idx])
        for i in range(24):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action_idx])
            self.pyboy.tick()

    def get_info(self) -> dict:
        return {**self.get_player_info(),
                **self.get_items_info(),
                **self.get_dungeon_info(),
                **self.get_world_map_info()}

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_player_info(self) -> dict:
        info = dict()
        info['hearts'] = self.decode_hearts(self.CURRENT_HEALTH_ADDR)
        info['max_hearts'] = self.pyboy.get_memory_value(self.MAXIMUM_HEALTH_ADDR)
        info['sword_level'] = self.pyboy.get_memory_value(self.SWORD_LEVEL_ADDR)
        info['shield_level'] = self.pyboy.get_memory_value(self.SHIELD_LEVEL_ADDR)
        info['bracelet_level'] = self.pyboy.get_memory_value(self.POWER_BRACELET_LEVEL_ADDR)
        info['rupees'] = self.decode_rupees()
        info['deaths_slot1'] = self.pyboy.get_memory_value(self.NUMBER_OF_DEATHS_SLOT1_ADDR)
        info['deaths_slot2'] = self.pyboy.get_memory_value(self.NUMBER_OF_DEATHS_SLOT2_ADDR)
        info['deaths_slot3'] = self.pyboy.get_memory_value(self.NUMBER_OF_DEATHS_SLOT3_ADDR)
        info['x_pos'] = self.pyboy.get_memory_value(self.X_POS_ADDR)
        info['y_pos'] = self.pyboy.get_memory_value(self.Y_POS_ADDR)
        info['destination_data1'] = self.pyboy.get_memory_value(self.DESTINATION_DATA1_ADDR)
        info['destination_data2'] = self.pyboy.get_memory_value(self.DESTINATION_DATA2_ADDR)
        info['room'] = self.pyboy.get_memory_value(self.ROOM_NUMBER_ADDR)
        return info

    def get_items_info(self) -> dict:
        info = dict()
        info['held_item1'] = self.pyboy.get_memory_value(self.CURRENT_HELD_ITEMS_ADDRS[0])
        info['held_item2'] = self.pyboy.get_memory_value(self.CURRENT_HELD_ITEMS_ADDRS[1])
        info['inventory'] = self.decode_inventory()
        info['flipper'] = self.pyboy.get_memory_value(self.FLIPPER_ADDR) == 0x01
        info['potion'] = self.pyboy.get_memory_value(self.POTION_ADDR) == 0x01
        info['item_trading_game'] = self.decode_trading_game_item()
        info['secret_shells'] = self.pyboy.get_memory_value(self.NUMBER_SECRET_SHELLS_ADDR)
        info['gold_leaves'] = self.pyboy.get_memory_value(self.NUMBER_GOLD_LEAVES_ADDR)
        info['arrows'] = self.pyboy.get_memory_value(self.NUMBER_ARROWS_ADDR)
        info['max_arrows'] = self.pyboy.get_memory_value(self.MAX_ARROWS_ADDR)
        info['magic_powder'] = self.pyboy.get_memory_value(self.MAGIC_POWDER_QUANTITY_ADDR)
        info['max_magic_powder'] = self.pyboy.get_memory_value(self.MAX_MAGIC_POWDER_ADDR)
        info['bombs'] = self.pyboy.get_memory_value(self.NUMBER_BOMBS_ADDR)
        info['max_bombs'] = self.pyboy.get_memory_value(self.MAX_BOMBS_ADDR)
        info['ocarina_songs_possessed'] = self.pyboy.get_memory_value(self.OCARINA_SONGS_POSSESSION_ADDR)
        info['selected_ocarina_song'] = self.pyboy.get_memory_value(self.OCARINA_SELECTED_SONG_ADDR)
        info['keys'] = self.pyboy.get_memory_value(self.NUMBER_POSSESSED_KEYS_ADDR)
        return info

    def get_dungeon_info(self) -> dict:
        info = dict()
        entrance_keys = list()
        for i, addr in enumerate(range(self.DUNGEON_ENTRANCE_KEYS_ADDRS[0], self.DUNGEON_ENTRANCE_KEYS_ADDRS[1] + 1)):
            if self.pyboy.get_memory_value(addr) == 0x01:
                entrance_keys.append(i)
        info['dungeon_entrance_keys'] = entrance_keys

        dungeon_items = dict()
        for i, addr in enumerate(range(self.DUNGEONS_ITEM_FLAGS_ADDRS[0], self.DUNGEONS_ITEM_FLAGS_ADDRS[1] + 1, 5)):
            dungeon_items[f'dungeon{i}_keys'] = self.pyboy.get_memory_value(addr)
        info['dungeon_items'] = dungeon_items

        dungeon_instrument = dict()
        for i, addr in enumerate(range(self.INSTRUMENT_FOR_EVERY_DUNGEON_ADDRS[0], self.INSTRUMENT_FOR_EVERY_DUNGEON_ADDRS[1] + 1)):
            dungeon_instrument[f'dungeon{i}_instrument'] = (self.pyboy.get_memory_value(addr) == 0x03)
        info['dungeon_instruments'] = dungeon_instrument

        info['dungeon_pos_grid'] = self.pyboy.get_memory_value(self.POS_IN8X8_DUNGEON_GRID_ADDR)

        return info

    def get_world_map_info(self) -> dict:
        info = dict()
        for i, addr in enumerate(range(self.WORLD_MAP_STATUS_ADDRS[0], self.WORLD_MAP_STATUS_ADDRS[1] + 1)):
            value = self.pyboy.get_memory_value(addr)
            attributes = list()
            if value == 0x00:
                attributes.append('U')
            else:
                if value & 0x10:
                    attributes.append('C')
                if value & 0x20:
                    attributes.append('O')
                if value & 0x80:
                    attributes.append('V')
            info[f'{i}'] = attributes
        return {'world_map': info}

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def decode_string(self, starting_addr: int, final_addr: int) -> str:
        name = ''
        for addr in range(starting_addr, final_addr + 1):
            bytes_value = self.pyboy.get_memory_value(addr).to_bytes(2, byteorder='big')
            if bytes_value in self.TEXT_TABLE.keys():
                char = self.TEXT_TABLE[bytes_value]
            else:
                char = str(bytes_value[0])
            """if char == 'END_MARKER':
                break"""
            name += char
        return name

    def decode_hearts(self, addr: int) -> float:
        return (self.pyboy.get_memory_value(addr) / 0x04) * 0.5

    def decode_rupees(self) -> int:
        return int(
            str(self.read_bcd(self.pyboy.get_memory_value(self.NUMBER_RUPEES_ADDRS[0]))) +
            str(self.read_bcd(self.pyboy.get_memory_value(self.NUMBER_RUPEES_ADDRS[1])))
        )

    def decode_inventory(self) -> list[str]:
        items = list()
        for addr in range(self.INVENTORY_ADDRS[0], self.INVENTORY_ADDRS[0] + 1):
            value = self.pyboy.get_memory_value(addr)
            if value != 0x00:
                items.append(self.ITEM_DICT[value])
        return items

    def decode_trading_game_item(self):
        return 'yoshi' if self.pyboy.get_memory_value(self.CURRENT_ITEM_TRADING_GAME_ADDR) == 0x01 else 'magnifier'

    def read_bcd(self, value) -> int:
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
