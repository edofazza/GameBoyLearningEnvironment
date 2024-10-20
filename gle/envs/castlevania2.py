from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class CastlevaniaIIBelmontsRevenge(Env):
    BELMONT_STATE_ADDR = 0xC001     # 0: standing. 1: walking. 2: crouching. 3: jumping.
    BELMONT_SUBSTATE_ADDR = 0xC002  # 0: none. 1: attacking, including both whip and subweapon. 2: knockback
    FLAG_BELMONT_HURT_ADDR = 0xC008 # Reset bits 0 and 1 to induce iframes; set bit 7 to cause knockback
    BELMONT_FACING_ADDR = 0xC009    # Bit 5 is Belmont's facing (20: left. 0: right.); bit 6 is Belmont's vertical flip, etc.
    BELMONT_SPRITE_INDEX_ADDR = 0xC00A
    BELMONT_Y_VELOCITY_ADDR = 0xC00F    # [s2] Negative is upward
    BELMONT_Y_SUBPIXEL_ADDR = 0xC011    # [u]
    BELMONT_Y_PIXEL_ADDR = 0xC012       # [u]
    BELMONT_X_VELOCITY_ADDR = 0xC014    # [s2] Negative is leftward
    BELMONT_X_SUBPIXEL_ADDR = 0xC016    # [u]
    BELMONT_X_PIXEL_ADDR = 0xC017       # [u]
    HITSTUN_ADDR = 0xC018   # [u]
    SUBWEAPON_PROJECTILE_ID_ADDR = 0xC300   # [u] 0 if no subweapon being thrown currently; 1 if cross/axe; 2 if holy water
    SUBWEAPON_MODE_ADDR = 0xC301        # [u] 0: belmont is still winding up to toss. 1: in the air. 2: become flame/return [cross]
    GAME_MODE_ADDR = 0xC880             # 0: konami logo. 1: title screen. 2: title fade-in. 3: stage select and title screen selected. 4: stage entry. 5: normal gameplay. 6: death. 7: game over. 9: Credits. D: password entry. E: intro reel.
    GLOBAL_GAME_TIMER_ADDR = 0xC882
    CURRENT_STAGE_ADDR = 0xC8C0         # 0: plant (glitched). 1: plant. 2: crystal. 3: cloud. 4: rock. 5: Drac 1. 6: Drac 2. 7: Drac 3.)
    CURRENT_SUBSTAGE_ADDR = 0xC8C1      # initially 0, increments every time belmont passes through a door.
    SCORE_ADDRS = [0xC8C2, 0xC8C4]      # [d3]
    LIVES_ADDR = 0xC8C5
    SUBWEAPON_ADDR = 0xC8D0     # 0: none. 1: axe (us) / cross (jp). 2: holy water
    WHIP_UPGRADE_ADDR = 0xC8D1  # 0: leather. 1: chain. 2: fireball
    SCREEN_SCROLL_SPEED_HORIZONTAL_ADDR = [0xCA82, 0xCA83]  # [u2]
    SCREEN_X_ADDR = 0xCA86      # [u] Every screen that Belmont moves right, this increments, and vice versa.
    SCREEN_SCROLL_SPEED_VERTICAL_ADDR = [0xCA8B, 0xCA8C]  # [u2]
    SCREEN_Y_ADDR = 0xCA8D      # [u] Every screen that Belmont moves downward, this increments, and vice versa
    SUBLEVEL_SCROLL_ADDR = 0xCA95   # bit 0 is 1 if sublevel scrolls vertically, otherwise horizontal.
    TIME_REMAINING_ADDRS = [0xCC80, 0xCC81]  # [d2]
    HEARTS_ADDR = 0xCC86    # [d]
    BELMONT_HITPOINTS_ADDR = 0xCC89  # [u]

    def __init__(self, level: str = 'crystal', hard: bool = False, window_type: str = 'headless',
                 save_path: str | None = None, load_path: str | None = None, max_actions: int | None = None,
                 all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        with importlib.resources.path('gle.rom', "Castlevania II - Belmont's Revenge (U) [!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()
        else:
            if level in ['crystal', 'cloud', 'plant', 'rock'] and hard:
                level += '_hard'
            self.load_path = f'states/castlevaniaiibelmont/{level}.state'

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
                [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
                [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_BUTTON_B],
                [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B],
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
            [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A],
            [WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_BUTTON_B],
            [WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B],
        ]

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

        self.screen = self.pyboy.botsupport_manager().screen()

        self.prev_action_idx = None
        self.prev_score = 0
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

        done = False
        if info['lives'] == 0 or info['time_remaining'] == 0 or info['game_mode'] != 'normal gameplay':
            done = True

        reward = info['score'] - self.prev_score
        self.prev_score = info['score']
        return obs, reward, done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.actions_taken = 0
        self.prev_score = 0

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.rom', "Castlevania II - Belmont's Revenge (U) [!].gb") as rom_path:
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
        info['hearts'] = self.read_bcd(self.pyboy.get_memory_value(self.HEARTS_ADDR))
        info['lives'] = self.pyboy.get_memory_value(self.LIVES_ADDR)
        info['score'] = self.decode_score()
        info['time_remaining'] = (100 * self.read_bcd(self.pyboy.get_memory_value(self.TIME_REMAINING_ADDRS[1]))
                                  + self.read_bcd(self.pyboy.get_memory_value(self.TIME_REMAINING_ADDRS[0])))
        info['hitpoints'] = self.pyboy.get_memory_value(self.BELMONT_HITPOINTS_ADDR)
        info['game_mode'] = self.decode_game_mode()
        info['state'] = self.decode_state()
        info['substate'] = self.decode_substate()
        info['stage'] = self.decode_stage()
        info['substage'] = self.pyboy.get_memory_value(self.CURRENT_SUBSTAGE_ADDR)
        info['hurt_flag'] = self.pyboy.get_memory_value(self.FLAG_BELMONT_HURT_ADDR)
        info['facing'] = 'left' if self.pyboy.get_memory_value(self.BELMONT_FACING_ADDR) & 0x20 else 'right'
        info['sprite_idx'] = self.pyboy.get_memory_value(self.BELMONT_SPRITE_INDEX_ADDR)
        info['x_pixel'] = self.pyboy.get_memory_value(self.BELMONT_X_PIXEL_ADDR)
        info['x_subpixel'] = self.pyboy.get_memory_value(self.BELMONT_X_SUBPIXEL_ADDR)
        info['y_pixel'] = self.pyboy.get_memory_value(self.BELMONT_Y_PIXEL_ADDR)
        info['y_subpixel'] = self.pyboy.get_memory_value(self.BELMONT_Y_SUBPIXEL_ADDR)
        info['hitstun'] = self.pyboy.get_memory_value(self.HITSTUN_ADDR)
        info['subweapon_projectile'] = self.decode_projectile()
        info['subweapon_mode'] = self.decode_subweapon_mode()
        info['global_game_timer'] = self.pyboy.get_memory_value(self.GLOBAL_GAME_TIMER_ADDR)
        info['subweapon'] = self.decode_subweapon()
        info['whip_upgrade'] = self.decode_whip_upgrade()
        info['screen_x'] = self.pyboy.get_memory_value(self.SCREEN_X_ADDR)
        info['screen_y'] = self.pyboy.get_memory_value(self.SCREEN_Y_ADDR)
        info['sublevel_scroll'] = 'vertical' if self.pyboy.get_memory_value(self.SUBLEVEL_SCROLL_ADDR) else 'horizontal'
        return info

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def decode_state(self) -> str:
        value = self.pyboy.get_memory_value(self.BELMONT_STATE_ADDR)
        if value == 0x00:
            return 'standing'
        elif value == 0x01:
            return 'walking'
        elif value == 0x02:
            return 'crouching'
        elif value == 0x03:
            return 'jumping'
        else:
            return 'UNK'

    def decode_substate(self) -> str:
        value = self.pyboy.get_memory_value(self.BELMONT_STATE_ADDR)
        if value == 0x00:
            return 'none'
        elif value == 0x01:
            return 'attacking'
        elif value == 0x02:
            return 'knockback'
        else:
            return 'UNK'

    def decode_projectile(self) -> str:
        value = self.pyboy.get_memory_value(self.SUBWEAPON_PROJECTILE_ID_ADDR)
        if value == 0x00:
            return 'none'
        elif value == 0x01:
            return 'axe'
        elif value == 0x02:
            return 'holy water'
        else:
            return 'UNK'

    def decode_subweapon_mode(self) -> str:
        value = self.pyboy.get_memory_value(self.SUBWEAPON_MODE_ADDR)
        if value == 0x00:
            return 'winding up to toss'
        elif value == 0x01:
            return 'in the air'
        elif value == 0x02:
            return 'become flame/return [cross]'
        else:
            return 'UNK'

    def decode_game_mode(self) -> str:
        try:
            modes = ['konami logo', 'title screen', 'title fade', 'stage\\title screen selected', 'stage entry',
                 'normal gameplay', 'death', 'game over', 'credits', 'password entry', 'intro reel']
            return modes[self.pyboy.get_memory_value(self.GAME_MODE_ADDR)]
        except:
            return str(self.pyboy.get_memory_value(self.GAME_MODE_ADDR))

    def decode_stage(self) -> str:
        stages = ['plant', 'plant', 'crystal', 'cloud', 'rock', 'drac1', 'drac2', 'drac3']
        return stages[self.pyboy.get_memory_value(self.CURRENT_STAGE_ADDR)]

    def decode_score(self) -> int:
        score = 0
        for i, addr in enumerate(range(self.SCORE_ADDRS[0], self.SCORE_ADDRS[1])):
            score += self.read_bcd(self.pyboy.get_memory_value(addr)) * (100 ** i)
        return score

    def decode_subweapon(self) -> str:
        value = self.pyboy.get_memory_value(self.SUBWEAPON_ADDR)
        if value == 0x00:
            return 'none'
        elif value == 0x01:
            return 'axe'
        elif value == 0x02:
            return 'holy water'
        else:
            return 'UNK'

    def decode_whip_upgrade(self) -> str:
        value = self.pyboy.get_memory_value(self.WHIP_UPGRADE_ADDR)
        if value == 0x00:
            return 'leather'
        elif value == 0x01:
            return 'chain'
        elif value == 0x02:
            return 'fireball'
        else:
            return 'UNK'

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_bcd(self, value) -> int:
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)

