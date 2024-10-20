from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class CastlevaniaTheAdventure(Env):
    # 0xC00C  Input held (1: right. 2: left. 4: up. 8: down. 10: a [jump]. 20: b [whip]. 40: select. 80: start.) Copied to C41A.
    CURRENT_STAGE_ADDR = 0xC02B
    HI_SCORE_ADDRS = [0xC030, 0xC032]   # C030 = ones and tens, C031 = hundreds and thousands, C032 = tens of thousands
    CURRENT_SCORE_ADDR = [0xC034, 0xC036]
    CURRENT_LIVES_ADDR = 0xC040
    CURRENT_SUBLEVEL_X_POSITION_METATILES_ADDR = 0xC402  # 4 tiles
    CURRENT_SUBLEVEL_ADDR = 0xC412
    LEVEL_TIMER_SECONDS_ADDR = 0xC436
    LEVEL_TIMER_MINUTES_ADDR = 0xC437
    # CHRISTOPHER
    CHRISTOPHER_ACTION_STATE_ADDR = 0xC502  # $00 = standing/walking.  $01 = jumping. $02 = crouching. $03 = rope. $04 = whipping/whip-jumping/whip-crouching.
    CHRISTOPHER_TIMER_WHIPPING_ADDR = 0xC505
    CHRISTOPHER_X_POS_ADDRS = [0xC50A, 0xC50C]  # pixels and sub-pixels
    CHRISTOPHER_JUMP_STATE_ADDR = 0xC50D    # 00 = grounded, $01 = jumpsquat, $0F = rising, $0A = falling
    CHRISTOPHER_GRAVITY_Y_ADDR = 0xC50E
    CHRISTOPHER_Y_VELOCITY = [0xC50F, 0xC510]   # pixels and sub-pixels
    CHRISTOPHER_Y_POS_CAMERA_ADDRS = [0xC511, 0xC512]
    CHRISTOPHER_X_POS_CAMERA_ADDR = 0xC513
    CHRISTOPHER_POSE_ADDR = 0xC514  # $01 = standing/walking, $02 = crouching/jumping, $03 = whipping, $05 = crouch-whipping, $07 = knockback, $09 = climbing rope...
    FACING_ADDR = 0xC515    # Bit 5 of this is facing ($20 = right; 0 = left)
    CHRISTOPHER_CURRENT_HP = 0xC519
    CHRISTOPHER_CURRENT_WHIP_ATTACK_TYPE_ADDR = 0xC51B  # $00 = grounded, $01 = jumping, $02 = crouching
    CHRISTOPHER_WHIP_STATUS_ADDR = 0xC51C   # $00 = regular / $01 = chain whip / $02 = fireball whip
    # ENEMY
    ENEMY_SLOT_TAKEN_ADDR = 0xC600  # x80 if slot taken
    ENEMY_ID_ADDR = 0xC601
    ENEMY_STATUS_ADDR = 0xC602  # for eyeball, 01 = midair / 02 = rolling
    ENEMY_X_POS_ADDR = 0xC607
    ENEMY_Y_POS_ADDR = 0xC612
    ENEMY_RESPAWN_ADDR = 0xC60C # 0 = yes / 1 = no

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 max_actions: int | None = None, all_actions: bool = False):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        with importlib.resources.path('gle.rom', "Castlevania Adventure, The (E) [!].gb") as rom_path:
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
        if info['lives'] == 0 or (info['level_timer_minutes'] == 0 and info['level_timer_seconds'] == 0):
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
        with importlib.resources.path('gle.rom', "Castlevania Adventure, The (E) [!].gb") as rom_path:
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
            for _ in range(11):
                self.take_action2(6)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            for _ in range(4):
                self.take_action2(6)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            for _ in range(120):
                self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            for _ in range(5):
                self.pyboy.tick()
            for _ in range(18):
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
        else: # different action
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
                **self.get_christopher_info(),
                **self.get_enemy_info()
                }

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_general_info(self) -> dict:
        info = dict()
        info['stage'] = self.pyboy.get_memory_value(self.CURRENT_STAGE_ADDR)
        info['hi_score'] = self.decode_score(self.HI_SCORE_ADDRS)
        info['score'] = self.decode_score(self.CURRENT_SCORE_ADDR)
        info['lives'] = self.pyboy.get_memory_value(self.CURRENT_LIVES_ADDR)
        info['sublevel_x_pos_metatiles'] = self.pyboy.get_memory_value(self.CURRENT_SUBLEVEL_X_POSITION_METATILES_ADDR)
        info['sublevel'] = self.pyboy.get_memory_value(self.CURRENT_SUBLEVEL_ADDR)
        info['level_timer_minutes'] = self.read_bcd(self.pyboy.get_memory_value(self.LEVEL_TIMER_MINUTES_ADDR))
        info['level_timer_seconds'] = self.read_bcd(self.pyboy.get_memory_value(self.LEVEL_TIMER_SECONDS_ADDR))
        return info

    def get_christopher_info(self) -> dict:
        info = dict()
        info['hp'] = self.pyboy.get_memory_value(self.CHRISTOPHER_CURRENT_HP)
        info['action_state'] = self.decode_action_state()
        info['whipping_timer'] = self.pyboy.get_memory_value(self.CHRISTOPHER_TIMER_WHIPPING_ADDR)
        info['x_pos'] = [self.pyboy.get_memory_value(addr) for addr in
                         range(self.CHRISTOPHER_X_POS_ADDRS[0], self.CHRISTOPHER_X_POS_ADDRS[1] + 1)]
        info['jump_state'] = self.decode_jump_state()
        info['y_gravity'] = self.pyboy.get_memory_value(self.CHRISTOPHER_GRAVITY_Y_ADDR)
        info['y_velocity_pixel'] = self.pyboy.get_memory_value(self.CHRISTOPHER_Y_VELOCITY[0])
        info['y_velocity_subpixel'] = self.pyboy.get_memory_value(self.CHRISTOPHER_Y_VELOCITY[1])
        info['y_pos_camera_pixel'] = self.pyboy.get_memory_value(self.CHRISTOPHER_Y_POS_CAMERA_ADDRS[0])
        info['y_pos_camera_subpixel'] = self.pyboy.get_memory_value(self.CHRISTOPHER_Y_POS_CAMERA_ADDRS[1])
        info['x_pos_camera'] = self.pyboy.get_memory_value(self.CHRISTOPHER_X_POS_CAMERA_ADDR)
        info['pose'] = self.decode_pose()
        info['facing'] = 'right' if self.pyboy.get_memory_value(self.FACING_ADDR) == 0x20 else 'left'
        info['whip_atk_type'] = self.decode_whip_atk()
        info['whip_status'] = self.decode_whip_status()
        return {'christopher': info}

    def get_enemy_info(self) -> dict:
        info = dict()
        info['slot_taken'] = self.pyboy.get_memory_value(self.ENEMY_SLOT_TAKEN_ADDR) == 0x80
        info['id'] = self.pyboy.get_memory_value(self.ENEMY_ID_ADDR)
        info['status'] = self.decode_enemy_status()
        info['x_pos'] = self.pyboy.get_memory_value(self.ENEMY_X_POS_ADDR)
        info['y_pos'] = self.pyboy.get_memory_value(self.ENEMY_Y_POS_ADDR)
        info['respawn'] = self.pyboy.get_memory_value(self.ENEMY_RESPAWN_ADDR) == 0x00
        return {'enemy': info}

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_bcd(self, value) -> int:
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)

    def decode_score(self, addr_list) -> int:
        score = 0
        for i, addr in enumerate(range(addr_list[0], addr_list[1] + 1)):
            score += (100 ** i) * self.read_bcd(self.pyboy.get_memory_value(addr))
        return score

    def decode_action_state(self) -> str:
        action_states = ['jumping', 'crouching', 'rope', 'whipping', 'UNK', 'UNK', 'UNK', 'UNK']
        return action_states[self.pyboy.get_memory_value(self.CHRISTOPHER_ACTION_STATE_ADDR)]

    def decode_jump_state(self) -> str:
        jump = self.pyboy.get_memory_value(self.CHRISTOPHER_JUMP_STATE_ADDR)
        if jump == 0x00:
            return 'grounded'
        elif jump == 0x01:
            return 'jumpsquat'
        elif jump == 0x0F:
            return 'rising'
        elif jump == 0x0A:
            return 'falling'
        else:
            return 'UNK'

    def decode_pose(self) -> str:
        pose = self.pyboy.get_memory_value(self.CHRISTOPHER_POSE_ADDR)
        if pose == 0x01:
            return 'standing/walking'
        elif pose == 0x02:
            return 'crouching/jumping'
        elif pose == 0x03:
            return 'whipping'
        elif pose == 0x05:
            return 'crouch-whipping'
        elif pose == 0x07:
            return 'knockback'
        elif pose == 0x09:
            return 'climbing rope'
        else:
            return 'UNK'

    def decode_whip_atk(self):
        value = self.pyboy.get_memory_value(self.CHRISTOPHER_CURRENT_WHIP_ATTACK_TYPE_ADDR)
        if value == 0x00:
            return 'grounded'
        elif value == 0x01:
            return 'jumping'
        elif value == 0x02:
            return 'crouching'
        else:
            return 'UNK'

    def decode_whip_status(self):
        value = self.pyboy.get_memory_value(self.CHRISTOPHER_WHIP_STATUS_ADDR)
        if value == 0x00:
            return 'regular'
        elif value == 0x01:
            return 'chain'
        elif value == 0x02:
            return 'fireball'
        else:
            return 'UNK'

    def decode_enemy_status(self):
        value = self.pyboy.get_memory_value(self.CHRISTOPHER_WHIP_STATUS_ADDR)
        if value == 0x00:
            return 'regular'
        elif value == 0x01:
            return 'midair'
        elif value == 0x02:
            return 'rolling'
        else:
            return 'UNK'
