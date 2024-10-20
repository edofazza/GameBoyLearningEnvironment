from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class SolomonsClub(Env):
    LEVEL_ADDR = 0xC100
    ROOM_ADDR = 0xC101
    LIVES_ADDR = 0xFFC8
    N_COLLECTED_FAIRIES_ADDR = 0xFFCB
    N_FIREBALLS_ADDR = 0xFFD9
    N_WATERGUNS_ADDR = 0xFFDA
    N_HAMMERS_ADDR = 0xFFDA
    N_HOURGLASSES_ADDR = 0xFFDB
    SHOES_ADDR = 0xFFDC
    HAT_ADDR = 0xFFDD
    TIME_REMAINING_ADDR = [0xFFC0, 0xFFC1]  # big endian
    MONEY_ADDR = [0xFFCC, 0xFFCD, 0xFFCE]   # big endian

    def __init__(self, level: int = 1, room: int = 1, window_type: str = 'headless',
                 save_path: str | None = None, load_path: str | None = None, all_actions: bool = False
                 ):
        assert 1 <= level <= 5, 'Level must be between 1 and 5'
        assert 1 <= room <= 10, 'Room must be between 1 and 10'
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.level = level
        self.room = room
        self.prev_time = None
        self.prev_lives = None
        self.prev_money = None
        self.window_type = window_type

        with importlib.resources.path('gle.rom', "Solomon's Club (UE) [!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title()}')
        assert "SOLOMON'S CLUB" == self.pyboy.cartridge_title(), "The cartridge's title should be 'SOLOMON'S CLUB"

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
                [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT],
                [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]
            ]

            self.release_actions = [
                WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.RELEASE_ARROW_RIGHT,
                WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PASS,
                [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],  # JUMP AND CREATE/DESTROY BLOCK
                [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_B],     # DOWN AND CREATE AND DESTROY BLOCK
                [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_RIGHT],  # JUMP AND CREATE/DESTROY BLOCK
                [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_ARROW_LEFT]  # JUMP AND CREATE/DESTROY BLOCK
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

        if info['remaining_time'] == 0 or info['lives'] == 0:
            done = True
        else:
            done = False

        return obs, self.reward(info['money'], info['remaining_time'], info['lives']), done, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.go_to_level_selection()
        self.wait_level_selection()

        self.from_level_selection_to_level(self.level)
        self.from_level_to_room(self.room)

        self.prev_money = None
        self.prev_lives = None
        self.prev_time = None

        # WAIT START OF THE GAME
        for _ in range(60 * 8):
            self.pyboy.tick()

        return self.render(), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()   # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        self.pyboy = PyBoy(
            "roms/Solomon's Club (UE) [!].gb",
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
    def take_action(self, action_idx: int) -> None:
        if action_idx > len(self.actions) - 4:     # DOWN AND CREATE AND DESTROY BLOCK
            self.pyboy.send_input(self.actions[action_idx][0])

            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.actions[action_idx][1])
                self.pyboy.tick()

            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.release_actions[action_idx][1])
                self.pyboy.tick()

            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.release_actions[action_idx][0])
                self.pyboy.tick()

        elif action_idx == len(self.actions) - 4:   # JUMP AND CREATE/DESTROY BLOCK
            self.pyboy.send_input(self.actions[action_idx][0])

            for i in range(15):
                if i == 16:
                    self.pyboy.send_input(self.actions[action_idx][1])
                self.pyboy.tick()

            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.release_actions[action_idx][1])
                self.pyboy.tick()

            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.release_actions[action_idx][0])
                self.pyboy.tick()
        else:
            self.pyboy.send_input(self.actions[action_idx])
            for i in range(15):
                if i == 8:
                    self.pyboy.send_input(self.release_actions[action_idx])
                self.pyboy.tick()

    def get_info(self) -> dict:
        return {'level': self.get_level(),
                'room': self.get_room(),
                'remaining_time': self.get_time_remaining(),
                'money': self.get_money(),
                'lives': self.get_lives(),
                'n_fairies': self.get_n_collected_fairies(),
                'n_fireballs': self.get_n_fireballs(),
                'n_hammer': self.get_n_hammers(),
                'n_waterguns': self.get_n_waterguns(),
                'n_hourglasses': self.get_n_hourglasses(),
                'shoes': self.has_shoes(),
                'hat': self.has_hat()
                }

    def reward(self, money: int, remaining_time: int, lives: int):
        if self.prev_lives is None:
            self.prev_lives = lives
            self.prev_time = remaining_time
            self.prev_money = 0

        reward = (money - self.prev_money) #+ 1000 * (lives - self.prev_lives)
        self.prev_time = remaining_time
        self.prev_lives = lives
        self.prev_money = money
        return reward

    #   ******************************************************
    #                FUNCTION FOR MOVING IN THE MENU
    #   ******************************************************
    def go_to_level_selection(self) -> None:
        while self.pyboy.frame_count != 250:
            self.pyboy.tick()
        if self.pyboy.frame_count == 250:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            for i in range(80):
                if i == 8:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                self.pyboy.tick()

    def wait_level_selection(self) -> None:
        while self.pyboy.frame_count != 335:
            self.pyboy.tick()

    def from_level_selection_to_level(self, level):
        while level != 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            self.pyboy.tick()
            level -= 1
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        for i in range(80):
            if i == 8:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick()

    def from_level_to_room(self, room: int = 1) -> None:
        if room > 5:
            room -= 5
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            self.pyboy.tick()
        while room != 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.tick()
            room -= 1
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        self.pyboy.tick()

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_time_remaining(self) -> int:
        return self.pyboy.get_memory_value(self.TIME_REMAINING_ADDR[0]) * (2 ** 8) + self.pyboy.get_memory_value(self.TIME_REMAINING_ADDR[1])
        #    int.from_bytes(self.pyboy.get_memory_value(self.TIME_REMAINING_ADDR[0]).to_bytes(2, byteorder='big')
        #                      + self.pyboy.get_memory_value(self.TIME_REMAINING_ADDR[1]).to_bytes(2, byteorder='big'),
        #                      byteorder='little')

    def get_level(self) -> int:
        return self.pyboy.get_memory_value(self.LEVEL_ADDR)

    def get_room(self) -> int:
        return self.pyboy.get_memory_value(self.ROOM_ADDR)

    def get_lives(self) -> int:
        return self.pyboy.get_memory_value(self.LIVES_ADDR)

    def get_n_collected_fairies(self) -> int:
        return self.pyboy.get_memory_value(self.N_COLLECTED_FAIRIES_ADDR)

    def get_n_fireballs(self) -> int:
        return self.pyboy.get_memory_value(self.N_FIREBALLS_ADDR)

    def get_n_hammers(self) -> int:
        return self.pyboy.get_memory_value(self.N_HAMMERS_ADDR)

    def get_n_hourglasses(self) -> int:
        return self.pyboy.get_memory_value(self.N_HOURGLASSES_ADDR)

    def get_n_waterguns(self) -> int:
        return self.pyboy.get_memory_value(self.N_WATERGUNS_ADDR)

    def has_shoes(self) -> bool:
        return self.pyboy.get_memory_value(self.SHOES_ADDR) == 1

    def has_hat(self) -> bool:
        return self.pyboy.get_memory_value(self.HAT_ADDR) == 1

    def get_money(self) -> int:
        return (self.pyboy.get_memory_value(self.MONEY_ADDR[0]) * (2 ** 16)
                + self.pyboy.get_memory_value(self.MONEY_ADDR[1]) * (2 ** 8)
                + self.pyboy.get_memory_value(self.MONEY_ADDR[2]))
