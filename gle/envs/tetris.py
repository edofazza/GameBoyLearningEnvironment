from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class Tetris(Env):
    # WRAM
    SCORE_BCD_ADDR = 0xC0A0
    PTRS_TO_CLEARED_LINES_ADDR = 0xC0A3
    SINGLES_ADDR = 0xC0AC
    DOUBLES_ADDR = 0xC0B1
    TRIPLES_ADDR = 0xC0B6
    TETRISES_ADDR = 0xC0BB
    FASTDROP_BONUS_SUM_ADDR = 0xC0C0
    DROPS_ADDR = 0xC0C2
    SLOW_DROP_ADDR = 0xC0C7
    FASTDROP_BONUS_ADDED_ADDR = 0xC0CE
    NEXT_TETROMINO_ADDR = 0xC213
    CURRENT_STEP_ADDR = 0xDF7F
    BG_MUSIC_ADDR = 0xDFBF
    # HRAM
    INPUT_ADDR = 0xFF80
    INPUT_DELTA_ADDR = 0xFF81
    #   Sprite info
    SPRITE_TMP_VALID_ADDR = 0xFF86
    SPRITE_TMP_Y_ADDR = 0xFF87
    SPRITE_TMP_X_ADDR = 0xFF88
    SPRITE_TMP_ID_ADDR = 0xFF89
    SPRITE_TMP_PRIO_ADDR = 0xFF8A
    SPRITE_TMP_FLIP_ADDR = 0xFF8B
    SPRITE_TMP_PAL_ADDR = 0xFF8C
    SPRITE_COORD_Y_ADDR = 0xFF90
    SPRITE_FINAL_X_ADDR = 0xFF91
    SPRITE_FINAL_Y_ADDR = 0xFF90
    SPRITE_HIDDEN_ADDR = 0xFF95
    SPRITE_ADDR_HI_ADDR = 0xFF96
    SPRITE_ADDR_LO_ADDR = 0xFF97
    #   Object info
    OBJ_ATTR_DEST_HI_ADDR = 0xFF8D
    OBJ_ATTR_DEST_LOW_ADDR = 0xFF8E
    OBJ_ATTR_DEST_NUM_ADDR = 0xFF8F
    #   Drop info
    DROP_STATE_ADDR = 0xFF98
    DROP_DELAY_COUNTER_ADDR = 0xFF99
    DROP_DELAY_ADDR = 0xFF9A
    #
    TETRIMINO_COLLIDED_ADDR = 0xFF99
    LEVEL_ADDR = 0xFFA9
    PAUSE_ADDR = 0xFFAB
    GAME_MODE_ADDR = 0xFFC0
    SONG_SELECTION_ADDR = 0xFFC1
    GAME_STATE_ADDR = 0xFFE1
    EXPERT_MODE_ADDR = 0xFFF4

    def __init__(self, level: int, window_type: str = 'headless', save_path: str | None = None,
                 load_path: str | None = None, max_actions: int | None = None, all_actions: bool = False,
                 return_sound: bool = False, start_button: bool = False,):
        assert window_type == 'SDL2' or window_type == 'headless'
        super().__init__()
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type
        # Sound
        self.return_sound = return_sound
        # Level
        self.level = level

        with importlib.resources.path('gle.roms', "Tetris (JUE) (V1.1) [!].gb") as rom_path:
            self.pyboy = PyBoy(
                str(rom_path),
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None and load_path.endswith('level_selection.state') and level != -1:
            self.load()
            self.from_level_selection_to_level(level)
        elif load_path is not None:
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
            ]

            self.release_actions = [
                WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.RELEASE_ARROW_RIGHT,
                WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PASS,
            ]

        if start_button and not all_actions:
            self.actions += [WindowEvent.PRESS_BUTTON_START]
            self.release_actions += [WindowEvent.RELEASE_BUTTON_START]

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))

        self.screen = self.pyboy.botsupport_manager().screen()

        self.prev_action_idx = None
        self.prev_score = 0
        self.reset()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]] | tuple[ObsType, np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.take_action(action)
        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True

        obs = self.render()
        info = self.get_info()

        if info['game_state'] != 0: #== 13: 13 game over
            done = True

        reward = info['score'] - self.prev_score
        self.prev_score = info['score']
        if self.return_sound:
            return obs, self.screen.sound, reward, done, False, info
        else:
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
        with importlib.resources.path('gle.roms', "Tetris (JUE) (V1.1) [!].gb") as rom_path:
            self.pyboy = PyBoy(
                str(rom_path),
                window_type=self.window_type
            )
        if self.load_path is not None and self.load_path.endswith('level_selection.state') and self.level != -1:
            self.load()
            self.from_level_selection_to_level(self.level)
        elif self.load_path is not None:
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
        for i in range(24):
            if i == 16:
                self.pyboy.send_input(self.release_actions[action_idx])
            self.pyboy.tick()

    def from_level_selection_to_level(self, level):
        if level > 4:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            self.pyboy.tick()
            level -= 5
        while level != 0:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.tick()
            level -= 1
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        for i in range(10):
            self.pyboy.tick()

    def get_info(self) -> dict:
        info = dict()
        get_memory_value = self.pyboy.get_memory_value

        # Core stats
        info['score'] = self.read_bcd(get_memory_value(self.SCORE_BCD_ADDR))
        info['ptrs_to_cleared_lines'] = self.read_bcd(get_memory_value(self.PTRS_TO_CLEARED_LINES_ADDR))
        info['singles'] = get_memory_value(self.SINGLES_ADDR)
        info['doubles'] = get_memory_value(self.DOUBLES_ADDR)
        info['triples'] = get_memory_value(self.TRIPLES_ADDR)
        info['tetrises'] = get_memory_value(self.TETRISES_ADDR)
        info['fastdrop_bonus_sum'] = get_memory_value(self.FASTDROP_BONUS_SUM_ADDR)
        info['drops'] = get_memory_value(self.DROPS_ADDR)
        info['slow_drop'] = get_memory_value(self.SLOW_DROP_ADDR)
        info['fastdrop_bonus_added'] = get_memory_value(self.FASTDROP_BONUS_ADDED_ADDR)
        info['next_tetromino'] = get_memory_value(self.NEXT_TETROMINO_ADDR)
        info['current_step'] = get_memory_value(self.CURRENT_STEP_ADDR)
        info['bg_music'] = get_memory_value(self.BG_MUSIC_ADDR)
        info['input'] = get_memory_value(self.INPUT_ADDR)
        info['input_delta'] = get_memory_value(self.INPUT_DELTA_ADDR)
        info['level'] = get_memory_value(self.LEVEL_ADDR)
        info['pause'] = get_memory_value(self.PAUSE_ADDR)
        info['game_mode'] = get_memory_value(self.GAME_MODE_ADDR)
        info['song_selection'] = get_memory_value(self.SONG_SELECTION_ADDR)
        info['game_state'] = get_memory_value(self.GAME_STATE_ADDR)
        info['expert_mode'] = get_memory_value(self.EXPERT_MODE_ADDR)

        # Sprite info
        info['sprite'] = {
            'tmp_valid': get_memory_value(self.SPRITE_TMP_VALID_ADDR),
            'tmp_y': get_memory_value(self.SPRITE_TMP_Y_ADDR),
            'tmp_x': get_memory_value(self.SPRITE_TMP_X_ADDR),
            'tmp_id': get_memory_value(self.SPRITE_TMP_ID_ADDR),
            'tmp_prio': get_memory_value(self.SPRITE_TMP_PRIO_ADDR),
            'tmp_flip': get_memory_value(self.SPRITE_TMP_FLIP_ADDR),
            'tmp_pal': get_memory_value(self.SPRITE_TMP_PAL_ADDR),
            'coord_y': get_memory_value(self.SPRITE_COORD_Y_ADDR),
            'final_x': get_memory_value(self.SPRITE_FINAL_X_ADDR),
            'final_y': get_memory_value(self.SPRITE_FINAL_Y_ADDR),
            'hidden': get_memory_value(self.SPRITE_HIDDEN_ADDR),
            'addr_hi': get_memory_value(self.SPRITE_ADDR_HI_ADDR),
            'addr_lo': get_memory_value(self.SPRITE_ADDR_LO_ADDR),
        }

        # Object info
        info['object'] = {
            'attr_dest_hi': get_memory_value(self.OBJ_ATTR_DEST_HI_ADDR),
            'attr_dest_low': get_memory_value(self.OBJ_ATTR_DEST_LOW_ADDR),
            'attr_dest_num': get_memory_value(self.OBJ_ATTR_DEST_NUM_ADDR),
        }

        # Drop info
        info['drop'] = {
            'state': get_memory_value(self.DROP_STATE_ADDR),
            'delay_counter': get_memory_value(self.DROP_DELAY_COUNTER_ADDR),
            'delay': get_memory_value(self.DROP_DELAY_ADDR),
            'collided': get_memory_value(self.TETRIMINO_COLLIDED_ADDR),
        }

        return info

    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_bcd(self, value) -> int:
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
