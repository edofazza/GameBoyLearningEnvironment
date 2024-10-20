from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class SuperMarioLand(Env):
    # VRAM
    LIVES_DISPLAYED_ADDR = [0x9806, 0x9807]   # copy from 0xDA15 (which are lives left)
    SCORE_ADDRS = [0x9820, 0x9825]   # copy from 0xC0A0 (Binary-coded decimal)
    COINS_TENS_ADDR = 0x9829
    COINS_ONES_ADDR = 0x982A
    CURRENT_WORLD_ADDR = 0x982C
    CURRENT_STAGE_ADDR = 0x982E
    TIMER_HUNDREDS_ADDR = 0x9831
    TIMER_TENS_ADDR = 0x9832
    TIMER_ONES_ADDR = 0x9833
    # WRAM
    #SCORE_ADDRS = [0xC0A0, 0xC0A2] # not working
    LIVES_EARNED_LOST_ADDR = 0xC0A3
    GAME_OVER_ADDR = 0xC0A4     # 0x39 = Game Over  (also 0xFFB3 HRAM)
    SUPERBALL_TIME_LEFT_ADDR = 0xC0A9
    MARIO_DEAD_JUMP_TIMER_ADDR = 0xC0AC  # 0x00 (upward speed) to 0x26 (downward speed)
    MARIO_STARMAN_TIMER_ADDR = 0xC0D3

    """TITLE_SCREEN_TIMER_ADDR = 0xC0D7    # (switch to/from demo)
    #       DEMO
    DEMO_CONTROLLER_STATE_CHANGE_TIMER_ADDR = 0xC0D8
    DEMO_CONTROLLER_STATE_ID_ADDR = 0xC0D9
    DEMO_CONTROLLER_STATE_ADDR = 0xC0DA
    DEMO_ID_ADDR = 0xC0DC   # 0x00 = 1-1 demo, 0x01 = 1-2 demo, 0x02 = 3-3 demo"""
    #       MARIO
    MARIO_Y_POS_RELATIVE_SCREEN_ADDR = 0xC201
    MARIO_X_POS_RELATIVE_SCREEN_ADDR = 0xC202
    MARIO_POSE_ADDR = 0xC203
    MARIO_JUMPING_ROUTINE_ADDR = 0xC207     # (probably) 0x00 = Not jumping, 0x01 = Ascending, 0x02 = Descending
    MARIO_Y_SPEED_ADDR = 0xC208     # 0x00 (a lot of speed) to 0x19 (no speed, top of jump)) (unintentionally reaches 0x1a and 0xff
    MARIO_LOW_JUMP_ADDR = 0xC209    # how long the button was pressed (0x02? to 0x0d frames)
    MARIO_ON_GROUND_FLAG_ADDR = 0xC20A  # Mario is on the ground flag (0x01 = On the ground, 0x00 = In the air)
    MARIO_X_SPEED_ADDR = 0xC20C     # ? Absolute value
    MARIO_FACING_DIRECTION_ADDR = 0xC20D    # 0x20 = Left, 0x10 = Right
    #       SOUND
    MUSIC_TRACK_ADDR = 0xDFE9
    #   0x00 = no request, 0x01 = level clear, 0x02 = death, 0x03 = pyramid,
    #   0x04 = underground, 0x05 = shoot-'em-up, 0x06 = chai kingdom,
    #   0x07 = birabuto kingdom, 0x08 = muda kingdom, 0x09 = bonus game,
    #   0x0a = walk to prize, 0x0b = boss, 0x0c = starman, 0x0d = get prize,
    #   0x0e = failure?, 0x0f = daisy, 0x10 = game over, 0x11 = credits,
    #   0x12 = fake daisy, 0x13 = tatanga, 0x14 and above = invalid

    # HRAM
    POWERUP_STATUS_ADDR = 0xFF99    # 0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 0x03 = shrinking, 0x04 = invincibility blinking
    POWERUP_STATUS_TIMER_ADDR = 0xFFA6  # growing = set to 0x50, shrinking = set to 0x50, invincibility blinking = set to 0x40
                                        # Time until respawn from death (set to 0x90 when Mario falls to the bottom of the screen, whether it's in a pit or from dying to an enemy)
    MARIO_HAS_SUPERBALL_ADDR = 0xFFB5  # 0x00 = no, 0x02 = yes
    #COINS_BINARY_DECIMAL_ADDR = 0xFFFA

    # LEVEL BLOCK INFO
    LEVEL_BLOCK_ADDR = 0xC0AB

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False, world: int = 1, level: int = 1,):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type
        self.prev_score = None

        with importlib.resources.path('gle.rom', "Super Mario Land (JUE) (V1.1) [!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
                window_type=self.window_type
            )

        self.set_world_level(world, level)

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

        self.prev_action_idx = None
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
        if info['game_over']:
            done = True

        return obs, self.get_reward_distance(info['level_progress']), done, False, info # obs, self.get_reward_(info['score']), done, False, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.close()
        self.prev_action_idx = None
        self.actions_taken = 0
        self.prev_score = None

        if self.load_path is None:
            self.skip_game_initial_video()

        return self.render(), self.get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.rom', "Super Mario Land (JUE) (V1.1) [!].gb") as rom_path:
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
            if self.pyboy.frame_count == 180:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            if self.pyboy.frame_count == 185:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.pyboy.frame_count == 200:
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
    def set_world_level(self, world: int, level: int) -> None:
        """
        Patches the handler for pressing start in the menu. It hardcodes a world and level to always "continue" from.

        Args:
            world (int): The world to select a level from, 0-3
            level (int): The level to start from, 0-2
        """

        for i in range(0x450, 0x461):
            self.pyboy.override_memory_value(0, i, 0x00)

        patch1 = [
            0x3E,   # LD A, d8
            (world << 4) | (level & 0x0F), # d8
        ]

        for i, byte in enumerate(patch1):
            self.pyboy.override_memory_value(0, 0x451 + i, byte)

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

    def get_info(self) -> dict:
        info = dict()

        info['lives'] = int(str(self.pyboy.get_memory_value(self.LIVES_DISPLAYED_ADDR[0])) + str(self.pyboy.get_memory_value(self.LIVES_DISPLAYED_ADDR[1])))
        info['lives_earned_lost'] = self.pyboy.get_memory_value(self.LIVES_EARNED_LOST_ADDR)
        info['level_progress'] = (self.pyboy.get_memory_value(self.LEVEL_BLOCK_ADDR) * 16 +
                                  (self.screen.tilemap_position_list()[16][0] - 7) % 16 +
                                  self.pyboy.get_memory_value(self.MARIO_X_POS_RELATIVE_SCREEN_ADDR))
        info['world'] = self.pyboy.get_memory_value(self.CURRENT_WORLD_ADDR)
        info['stage'] = self.pyboy.get_memory_value(self.CURRENT_STAGE_ADDR)
        info['time'] = (100 * self.read_bcd(self.pyboy.get_memory_value(self.TIMER_HUNDREDS_ADDR))
                        + 10 * self.read_bcd(self.pyboy.get_memory_value(self.TIMER_TENS_ADDR))
                        + self.read_bcd(self.pyboy.get_memory_value(self.TIMER_ONES_ADDR)))
        info['x_pos_screen'] = self.pyboy.get_memory_value(self.MARIO_X_POS_RELATIVE_SCREEN_ADDR)
        info['y_pos_screen'] = self.pyboy.get_memory_value(self.MARIO_Y_POS_RELATIVE_SCREEN_ADDR)
        info['score'] = self.decode_score()
        info['coins'] = (10 * self.read_bcd(self.pyboy.get_memory_value(self.COINS_TENS_ADDR))
                         + self.read_bcd(self.pyboy.get_memory_value(self.COINS_ONES_ADDR)))
        info['pose'] = self.pyboy.get_memory_value(self.MARIO_POSE_ADDR)
        info['facing'] = 'left' if self.pyboy.get_memory_value(self.MARIO_FACING_DIRECTION_ADDR) == 0x20 else 'right'
        info['on_ground'] = self.pyboy.get_memory_value(self.MARIO_ON_GROUND_FLAG_ADDR) == 0x01
        info['game_over'] = self.pyboy.get_memory_value(self.GAME_OVER_ADDR) == 0x39
        info['superball_time_left'] = self.pyboy.get_memory_value(self.SUPERBALL_TIME_LEFT_ADDR)
        info['has_superball'] = self.pyboy.get_memory_value(self.MARIO_HAS_SUPERBALL_ADDR) == 0x02
        info['dead_jump_timer'] = self.pyboy.get_memory_value(self.MARIO_DEAD_JUMP_TIMER_ADDR)
        info['starman_timer'] = self.pyboy.get_memory_value(self.MARIO_STARMAN_TIMER_ADDR)
        info['jumping_routing'] = self.decode_jumping_routine()
        info['x_speed'] = self.pyboy.get_memory_value(self.MARIO_X_SPEED_ADDR)
        info['y_speed'] = self.pyboy.get_memory_value(self.MARIO_Y_SPEED_ADDR)
        info['low_jump'] = self.pyboy.get_memory_value(self.MARIO_LOW_JUMP_ADDR)
        info['music_track'] = self.pyboy.get_memory_value(self.MUSIC_TRACK_ADDR)
        info['powerup'] = self.pyboy.get_memory_value(self.POWERUP_STATUS_ADDR)
        info['powerup_timer'] = self.pyboy.get_memory_value(self.POWERUP_STATUS_TIMER_ADDR)
        return info

    def get_reward(self, score) -> int:
        if self.prev_score is None:
            self.prev_score = score
            return 0
        else:
            reward = score - self.prev_score
            self.prev_score = score
            return reward

    def get_reward_distance(self, score) -> int:
        if self.prev_score is None or self.prev_score > score + 20:
            self.prev_score = score
            return 0
        else:
            reward = score - self.prev_score
            self.prev_score = score
            return reward
    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_bcd(self, value) -> int:
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)

    def decode_jumping_routine(self) -> str:
        value = self.pyboy.get_memory_value(self.MARIO_JUMPING_ROUTINE_ADDR)
        if value == 0x00:
            return 'not jumping'
        elif value == 0x01:
            return 'ascending'
        elif value == 0x02:
            return 'descending'
        else:
            return 'unknown'

    def decode_score(self) -> int:
        score = 0
        for i, addr in enumerate(range(self.SCORE_ADDRS[0], self.SCORE_ADDRS[1] + 1)):
            value = self.pyboy.get_memory_value(addr)
            if value > 9:
                continue
            else:
                value = self.read_bcd(value)
                score += (10 ** (self.SCORE_ADDRS[1] - self.SCORE_ADDRS[0] - i)) * value
        return score
