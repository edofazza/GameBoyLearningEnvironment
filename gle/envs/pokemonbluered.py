from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces
import importlib.resources

from gle.envs.general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class PokemonBlueRed(Env):
    # BATTLE
    WHOSE_TURN_ADDR = 0xFFF3
    # BATTLE ENEMY
    ENEMY_MOVE_ADDR = 0xCFCC
    ENEMY_MOVE_EFFECT_ADDR = 0xCFCD
    ENEMY_MOVE_POWER_ADDR = 0xCFCE
    ENEMY_MOVE_TYPE_ADDR = 0xCFCF
    ENEMY_MOVE_ACCURACY_ADDR = 0xCFD0
    ENEMY_MOVE_MAX_PP_ADDR = 0xCFD1
    ENEMY_POKEMON_ID_ADDR = 0xCFD8
    ENEMY_NAME_ADDRS = [0xCFDA, 0xCFE4]  # CFE4 – CFDA = A = 10 (dec)
    ENEMY_HP_ADDR = [0xCFE6, 0xCFE7]
    ENEMY_LEVEL_ADDR = 0xCFF3  # also CFF3  0xCFE8 (giving 0 during the battle with rival)
    #   ENEMY STATUS
    #       bit 6: paralyzed
    #       bit 5: frozen
    #       bit 4: burned
    #       bit 3: poisoned
    #       bit 0-2: sleep counter
    ENEMY_STATUS_ADDR = 0xCFE9
    ENEMY_TYPE1_ADDR = 0xCFEA
    ENEMY_TYPE2_ADDR = 0xCFEB
    ENEMY_MOVE1_ADDR = 0xCFED
    ENEMY_MOVE2_ADDR = 0xCFEE
    ENEMY_MOVE3_ADDR = 0xCFEF
    ENEMY_MOVE4_ADDR = 0xCFF0
    ENEMY_ATTACK_DEFENSE_DVS_ADDR = 0xCFF1
    ENEMY_SPEED_SPECIAL_DVS_ADDR = 0xCFF2
    ENEMY_MAX_HP_ADDRS = [0xCFF4, 0xCFF5]
    ENEMY_ATTACK_ADDRS = [0xCFF6, 0xCFF7]
    ENEMY_DEFENSE_ADDRS = [0xCFF8, 0xCFF9]
    ENEMY_SPEED_ADDRS = [0xCFFA, 0xCFFB]
    ENEMY_SPECIAL_ADDRS = [0xCFFC, 0xCFFD]
    ENEMY_PP_SLOT1_ADDR = 0xCFFE
    ENEMY_PP_SLOT2_ADDR = 0xCFFF
    ENEMY_PP_SLOT3_ADDR = 0xD000
    ENEMY_PP_SLOT4_ADDR = 0xD001
    ENEMY_BASE_STATS_ADDRS = [0xD002, 0xD006]
    ENEMY_CATCH_RATE_ADDR = 0xD007
    ENEMY_BASE_EXPERIENCE_ADDR = 0xD008
    # BATTLE PLAYER
    PLAYER_MOVE_ID_ADDR = 0xCFD2
    PLAYER_MOVE_EFFECT_ADDR = 0xCFD3
    PLAYER_MOVE_POWER_ADDR = 0xCFD4
    PLAYER_MOVE_TYPE_ADDR = 0xCFD5
    PLAYER_MOVE_ACCURACY_ADDR = 0xCFD6
    PLAYER_MOVE_MAX_PP_ADDR = 0xCFD7
    PLAYER_POKEMON_ID_ADDR = 0xCFD9
    #       Pokemon 1st slot (in-battle)
    IN_BATTLE_POKEMON_NAME_ADDRS = [0xD009, 0xD013]  # 10 entries, each is a char
    IN_BATTLE_POKEMON_NUMBER_ADDR = 0xD014
    IN_BATTLE_POKEMON_HP_ADDRS = [0xD015, 0xD016]
    IN_BATTLE_POKEMON_STATUS_ADDR = 0xD018
    IN_BATTLE_POKEMON_TYPE1_ADDR = 0xD019
    IN_BATTLE_POKEMON_TYPE2_ADDR = 0xD01A
    IN_BATTLE_POKEMON_MOVE1_ADDR = 0xD01C
    IN_BATTLE_POKEMON_MOVE2_ADDR = 0xD01D
    IN_BATTLE_POKEMON_MOVE3_ADDR = 0xD01E
    IN_BATTLE_POKEMON_MOVE4_ADDR = 0xD01F
    IN_BATTLE_POKEMON_ATTACK_DEFENSE_DVS_ADDR = 0xD020
    IN_BATTLE_POKEMON_SPEED_SPECIAL_DVS_ADDR = 0xD021
    IN_BATTLE_POKEMON_LEVEL_ADDR = 0xD022
    IN_BATTLE_POKEMON_MAX_HP_ADDRS = [0xD023, 0xD024]
    IN_BATTLE_POKEMON_ATTACK_ADDRS = [0xD025, 0xD026]
    IN_BATTLE_POKEMON_DEFENSE_ADDRS = [0xD027, 0xD028]
    IN_BATTLE_POKEMON_SPEED_ADDRS = [0xD029, 0xD02A]
    IN_BATTLE_POKEMON_SPECIAL_ADDRS = [0xD02B, 0xD02C]
    IN_BATTLE_POKEMON_PP_SLOT1_ADDR = 0xD02D
    IN_BATTLE_POKEMON_PP_SLOT2_ADDR = 0xD02E
    IN_BATTLE_POKEMON_PP_SLOT3_ADDR = 0xD02F
    IN_BATTLE_POKEMON_PP_SLOT4_ADDR = 0xD030

    # BATTLE GENERAL
    CRITICAL_OHKO_FLAG_ADDR = 0xD05E  # 01 - Critical Hit! 02 - One-hit KO!
    HOOKED_POKEMON_FLAG_ADDR = 0xD5F
    AMOUNT_DAMAGE_ADDR = 0xD0D8  # Amount of damage attack is about to do. Max possible damage may appear one frame before actual damage

    # PLAYER
    PLAYER_NAME_ADDRS = [0xD158, 0xD162]
    N_POKEMON_IN_PARTY_ADDR = 0xD163
    ON_BIKE_ADDR = 0xD700
    #   Pokemon 1
    PARTY_POKEMON1_ADDR = 0xD164  # also 0xD16B
    PARTY_POKEMON1_HP_ADDR = [0xD16C, 0xD16D]
    PARTY_POKEMON1_STATUS_ADDR = 0xD16F
    PARTY_POKEMON1_TYPE1_ADDR = 0xD170
    PARTY_POKEMON1_TYPE2_ADDR = 0xD171
    PARTY_POKEMON1_MOVE1_ADDR = 0xD173
    PARTY_POKEMON1_MOVE2_ADDR = 0xD174
    PARTY_POKEMON1_MOVE3_ADDR = 0xD175
    PARTY_POKEMON1_MOVE4_ADDR = 0xD176
    PARTY_POKEMON1_EXP_ADDR = [0xD179, 0xD17B]
    PARTY_POKEMON1_HP_EV_ADDR = [0xD17C, 0xD17D]
    PARTY_POKEMON1_ATTACK_EV_ADDR = [0xD17E, 0xD17F]
    PARTY_POKEMON1_DEFENSE_EV_ADDR = [0xD180, 0xD181]
    PARTY_POKEMON1_SPEED_EV_ADDR = [0xD182, 0xD183]
    PARTY_POKEMON1_SPECIAL_EV_ADDR = [0xD184, 0xD185]
    PARTY_POKEMON1_ATTACK_DEFENSE_IV_ADDR = 0xD186
    PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR = 0xD187
    PARTY_POKEMON1_PP_SLOT1_ADDR = 0xD188
    PARTY_POKEMON1_PP_SLOT2_ADDR = 0xD189
    PARTY_POKEMON1_PP_SLOT3_ADDR = 0xD18A
    PARTY_POKEMON1_PP_SLOT4_ADDR = 0xD18B
    PARTY_POKEMON1_LEVEL_ADDR = 0xD18C
    PARTY_POKEMON1_MAX_HP_ADDR = [0xD18D, 0xD18E]
    PARTY_POKEMON1_ATTACK_ADDR = [0xD18F, 0xD190]
    PARTY_POKEMON1_DEFENSE_ADDR = [0xD191, 0xD192]
    PARTY_POKEMON1_SPEED_ADDR = [0xD193, 0xD194]
    PARTY_POKEMON1_SPECIAL_ADDR = [0xD195, 0xD196]
    #   Pokemon 2
    PARTY_POKEMON2_ADDR = 0xD165  # also 0xD197
    PARTY_POKEMON2_HP_ADDR = [0xD198, 0xD199]
    PARTY_POKEMON2_STATUS_ADDR = 0xD19B
    PARTY_POKEMON2_TYPE1_ADDR = 0xD19C
    PARTY_POKEMON2_TYPE2_ADDR = 0xD19D
    PARTY_POKEMON2_MOVE1_ADDR = 0xD19F
    PARTY_POKEMON2_MOVE2_ADDR = 0xD1A0
    PARTY_POKEMON2_MOVE3_ADDR = 0xD1A1
    PARTY_POKEMON2_MOVE4_ADDR = 0xD1A2
    PARTY_POKEMON2_EXP_ADDR = [0xD1A5, 0xD1A7]
    PARTY_POKEMON2_HP_EV_ADDR = [0xD1A8, 0xD1A9]
    PARTY_POKEMON2_ATTACK_EV_ADDR = [0xD1AA, 0xD1AB]
    PARTY_POKEMON2_DEFENSE_EV_ADDR = [0xD1AC, 0xD1AD]
    PARTY_POKEMON2_SPEED_EV_ADDR = [0xD1AE, 0xD1AF]
    PARTY_POKEMON2_SPECIAL_EV_ADDR = [0xD1B0, 0xD1B1]
    PARTY_POKEMON2_ATTACK_DEFENSE_IV_ADDR = 0xD1B2
    PARTY_POKEMON2_SPEED_SPECIAL_IV_ADDR = 0xD1B3
    PARTY_POKEMON2_PP_SLOT1_ADDR = 0xD1B4
    PARTY_POKEMON2_PP_SLOT2_ADDR = 0xD1B5
    PARTY_POKEMON2_PP_SLOT3_ADDR = 0xD1B6
    PARTY_POKEMON2_PP_SLOT4_ADDR = 0xD1B7
    PARTY_POKEMON2_LEVEL_ADDR = 0xD1B8
    PARTY_POKEMON2_MAX_HP_ADDR = [0xD1B9, 0xD1BA]
    PARTY_POKEMON2_ATTACK_ADDR = [0xD1BB, 0xD1BC]
    PARTY_POKEMON2_DEFENSE_ADDR = [0xD1BD, 0xD1BE]
    PARTY_POKEMON2_SPEED_ADDR = [0xD1BF, 0xD1C0]
    PARTY_POKEMON2_SPECIAL_ADDR = [0xD1C1, 0xD1C2]
    #   Pokemon 3
    PARTY_POKEMON3_ADDR = 0xD166  # also 0xD1C3
    PARTY_POKEMON3_HP_ADDR = [0xD1C4, 0xD1C5]
    PARTY_POKEMON3_STATUS_ADDR = 0xD1C7
    PARTY_POKEMON3_TYPE1_ADDR = 0xD1C8
    PARTY_POKEMON3_TYPE2_ADDR = 0xD1C9
    PARTY_POKEMON3_MOVE1_ADDR = 0xD1CB
    PARTY_POKEMON3_MOVE2_ADDR = 0xD1CC
    PARTY_POKEMON3_MOVE3_ADDR = 0xD1CD
    PARTY_POKEMON3_MOVE4_ADDR = 0xD1CE
    PARTY_POKEMON3_EXP_ADDR = [0xD1D1, 0xD1D3]
    PARTY_POKEMON3_HP_EV_ADDR = [0xD1D4, 0xD1D5]
    PARTY_POKEMON3_ATTACK_EV_ADDR = [0xD1D6, 0xD1D7]
    PARTY_POKEMON3_DEFENSE_EV_ADDR = [0xD1D8, 0xD1D9]
    PARTY_POKEMON3_SPEED_EV_ADDR = [0xD1DA, 0xD1DB]
    PARTY_POKEMON3_SPECIAL_EV_ADDR = [0xD1DC, 0xD1DD]
    PARTY_POKEMON3_ATTACK_DEFENSE_IV_ADDR = 0xD1DE
    PARTY_POKEMON3_SPEED_SPECIAL_IV_ADDR = 0xD1DF
    PARTY_POKEMON3_PP_SLOT1_ADDR = 0xD1E0
    PARTY_POKEMON3_PP_SLOT2_ADDR = 0xD1E1
    PARTY_POKEMON3_PP_SLOT3_ADDR = 0xD1E2
    PARTY_POKEMON3_PP_SLOT4_ADDR = 0xD1E3
    PARTY_POKEMON3_LEVEL_ADDR = 0xD1E4
    PARTY_POKEMON3_MAX_HP_ADDR = [0xD1E5, 0xD1E6]
    PARTY_POKEMON3_ATTACK_ADDR = [0xD1E7, 0xD1E8]
    PARTY_POKEMON3_DEFENSE_ADDR = [0xD1E9, 0xD1EA]
    PARTY_POKEMON3_SPEED_ADDR = [0xD1EB, 0xD1EC]
    PARTY_POKEMON3_SPECIAL_ADDR = [0xD1ED, 0xD1EE]
    #   Pokemon 4
    PARTY_POKEMON4_ADDR = 0xD167  # also 0xD1EF
    PARTY_POKEMON4_HP_ADDR = [0xD1F0, 0xD1F1]
    PARTY_POKEMON4_STATUS_ADDR = 0xD1F3
    PARTY_POKEMON4_TYPE1_ADDR = 0xD1F4
    PARTY_POKEMON4_TYPE2_ADDR = 0xD1F5
    PARTY_POKEMON4_MOVE1_ADDR = 0xD1F7
    PARTY_POKEMON4_MOVE2_ADDR = 0xD1F8
    PARTY_POKEMON4_MOVE3_ADDR = 0xD1F9
    PARTY_POKEMON4_MOVE4_ADDR = 0xD1FA
    PARTY_POKEMON4_EXP_ADDR = [0xD1FB, 0xD1FC]
    PARTY_POKEMON4_HP_EV_ADDR = [0xD200, 0xD201]
    PARTY_POKEMON4_ATTACK_EV_ADDR = [0xD202, 0xD203]
    PARTY_POKEMON4_DEFENSE_EV_ADDR = [0xD204, 0xD205]
    PARTY_POKEMON4_SPEED_EV_ADDR = [0xD206, 0xD207]
    PARTY_POKEMON4_SPECIAL_EV_ADDR = [0xD208, 0xD209]
    PARTY_POKEMON4_ATTACK_DEFENSE_IV_ADDR = 0xD20A
    PARTY_POKEMON4_SPEED_SPECIAL_IV_ADDR = 0xD20B
    PARTY_POKEMON4_PP_SLOT1_ADDR = 0xD20C
    PARTY_POKEMON4_PP_SLOT2_ADDR = 0xD20D
    PARTY_POKEMON4_PP_SLOT3_ADDR = 0xD20E
    PARTY_POKEMON4_PP_SLOT4_ADDR = 0xD20F
    PARTY_POKEMON4_LEVEL_ADDR = 0xD210
    PARTY_POKEMON4_MAX_HP_ADDR = [0xD211, 0xD212]
    PARTY_POKEMON4_ATTACK_ADDR = [0xD213, 0xD214]
    PARTY_POKEMON4_DEFENSE_ADDR = [0xD215, 0xD216]
    PARTY_POKEMON4_SPEED_ADDR = [0xD217, 0xD218]
    PARTY_POKEMON4_SPECIAL_ADDR = [0xD219, 0xD21A]
    #   Pokemon 5
    PARTY_POKEMON5_ADDR = 0xD168  # also 0xD21B
    PARTY_POKEMON5_HP_ADDR = [0xD21C, 0xD21D]
    PARTY_POKEMON5_STATUS_ADDR = 0xD21F
    PARTY_POKEMON5_TYPE1_ADDR = 0xD220
    PARTY_POKEMON5_TYPE2_ADDR = 0xD221
    PARTY_POKEMON5_MOVE1_ADDR = 0xD223
    PARTY_POKEMON5_MOVE2_ADDR = 0xD224
    PARTY_POKEMON5_MOVE3_ADDR = 0xD225
    PARTY_POKEMON5_MOVE4_ADDR = 0xD226
    PARTY_POKEMON5_EXP_ADDR = [0xD229, 0xD22B]
    PARTY_POKEMON5_HP_EV_ADDR = [0xD22C, 0xD22D]
    PARTY_POKEMON5_ATTACK_EV_ADDR = [0xD22E, 0xD22F]
    PARTY_POKEMON5_DEFENSE_EV_ADDR = [0xD230, 0xD231]
    PARTY_POKEMON5_SPEED_EV_ADDR = [0xD232, 0xD233]
    PARTY_POKEMON5_SPECIAL_EV_ADDR = [0xD234, 0xD235]
    PARTY_POKEMON5_ATTACK_DEFENSE_IV_ADDR = 0xD236
    PARTY_POKEMON5_SPEED_SPECIAL_IV_ADDR = 0xD237
    PARTY_POKEMON5_PP_SLOT1_ADDR = 0xD238
    PARTY_POKEMON5_PP_SLOT2_ADDR = 0xD239
    PARTY_POKEMON5_PP_SLOT3_ADDR = 0xD23A
    PARTY_POKEMON5_PP_SLOT4_ADDR = 0xD23B
    PARTY_POKEMON5_LEVEL_ADDR = 0xD23C
    PARTY_POKEMON5_MAX_HP_ADDR = [0xD23D, 0xD23E]
    PARTY_POKEMON5_ATTACK_ADDR = [0xD23F, 0xD240]
    PARTY_POKEMON5_DEFENSE_ADDR = [0xD241, 0xD242]
    PARTY_POKEMON5_SPEED_ADDR = [0xD243, 0xD244]
    PARTY_POKEMON5_SPECIAL_ADDR = [0xD245, 0xD246]
    #   Pokemon 6
    PARTY_POKEMON6_ADDR = 0xD169  # also 0xD247
    PARTY_POKEMON6_HP_ADDR = [0xD248, 0xD249]
    PARTY_POKEMON6_STATUS_ADDR = 0xD24B
    PARTY_POKEMON6_TYPE1_ADDR = 0xD24C
    PARTY_POKEMON6_TYPE2_ADDR = 0xD24D
    PARTY_POKEMON6_MOVE1_ADDR = 0xD24F
    PARTY_POKEMON6_MOVE2_ADDR = 0xD250
    PARTY_POKEMON6_MOVE3_ADDR = 0xD251
    PARTY_POKEMON6_MOVE4_ADDR = 0xD252
    PARTY_POKEMON6_EXP_ADDR = [0xD255, 0xD257]
    PARTY_POKEMON6_HP_EV_ADDR = [0xD258, 0xD259]
    PARTY_POKEMON6_ATTACK_EV_ADDR = [0xD25A, 0xD25B]
    PARTY_POKEMON6_DEFENSE_EV_ADDR = [0xD25C, 0xD25D]
    PARTY_POKEMON6_SPEED_EV_ADDR = [0xD25E, 0xD25F]
    PARTY_POKEMON6_SPECIAL_EV_ADDR = [0xD260, 0xD261]
    PARTY_POKEMON6_ATTACK_DEFENSE_IV_ADDR = 0xD262
    PARTY_POKEMON6_SPEED_SPECIAL_IV_ADDR = 0xD263
    PARTY_POKEMON6_PP_SLOT1_ADDR = 0xD264
    PARTY_POKEMON6_PP_SLOT2_ADDR = 0xD265
    PARTY_POKEMON6_PP_SLOT3_ADDR = 0xD266
    PARTY_POKEMON6_PP_SLOT4_ADDR = 0xD267
    PARTY_POKEMON6_LEVEL_ADDR = 0xD268
    PARTY_POKEMON6_MAX_HP_ADDR = [0xD269, 0xD26A]
    PARTY_POKEMON6_ATTACK_ADDR = [0xD26B, 0xD26C]
    PARTY_POKEMON6_DEFENSE_ADDR = [0xD26D, 0xD26E]
    PARTY_POKEMON6_SPEED_ADDR = [0xD26F, 0xD270]
    PARTY_POKEMON6_SPECIAL_ADDR = [0xD271, 0xD272]

    # POKEDEX
    POKEDEX_1_152_ADDRS = [0xD2F7, 0xD309]
    #   seen
    POKEDEX_SEEN_1_152_ADDRS = [0xD30A, 0xD31C]

    # ITEMS
    TOTAL_ITEM_COUNT_ADDR = 0xD31D
    ITEMS_ADDRS = [
        0xD31E, 0xD320, 0xD322, 0xD324, 0xD326,
        0xD328, 0xD32A, 0xD32C, 0xD32E, 0xD330,
        0xD332, 0xD334, 0xD336, 0xD338, 0xD33A,
        0xD33C, 0xD33E, 0xD340, 0xD342, 0xD344
    ]
    ITEM_QUANTITIES_ADDRS = [
        0xD31F, 0xD321, 0xD323, 0xD325, 0xD327,
        0xD329, 0xD32B, 0xD32D, 0xD32F, 0xD331,
        0xD333, 0xD335, 0xD337, 0xD339, 0xD33B,
        0xD33D, 0xD33F, 0xD341, 0xD343, 0xD345
    ]

    # MONEY
    MONEY_ADDRS = [0xD347, 0xD348, 0xD349]

    # RIVAL NAME
    RIVAL_NAME_ADDRS = [0xD34A, 0xD351]

    # MISCELLANEOUS
    BADGES_ADDR = 0xD356
    CURRENT_MAP_NUMBER_ADDR = 0xD35E
    CURRENT_Y_POS_ADDR = 0xD361
    CURRENT_X_POS_ADDR = 0xD362
    JOYPAD_SIMULATION_ADDR = 0xCD38

    # EVENT FLAGS
    STARTERS_BACK_FLAG_ADDR = 0xD5AB
    MEWTWO_APPEARS_FLAG_ADDR = 0xD5C0
    HAVE_TOWN_MAP_FLAG_ADDR = 0xD5F3
    HAVE_OAK_PARCEL_ADDR = 0xD60D
    FOSSILIZED_POKEMON_FLAG_ADDR = 0xD710
    G0T_LAPRAS_ADDR = 0xD72E
    FOUGHT_GIOVANNI_ADDR = 0xD751
    FOUGHT_BROOK_ADDR = 0xD755
    FOUGHT_MISTY_ADDR = 0xD75E
    FOUGHT_LT_SURGE_ADDR = 0xD773
    FOUGHT_ERIKA_ADDR = 0xD77C
    FOUGHT_ARTICUNO_ADDR = 0xD782
    FOUGHT_KOGA_ADDR = 0xD792
    FOUGHT_BLAINE_ADDR = 0xD79A
    FOUGHT_SABRINA_ADDR = 0xD7B3
    FOUGHT_ZAPDOS_ADDR = 0xD7D4
    FOUGHT_SNORLAX_VERMILION_ADDR = 0xD7D8
    FOUGHT_SNORLAX_CELADON_ADDR = 0xD7E0
    FOUGHT_MOLTRES_ADDR = 0xD7EE
    IS_SS_ANNE_HERE_ADDR = 0xD803
    MEWTWO_CAN_BE_CAUGHT = 0xD85F  # Mewtwo can be caught if bit 2 clear - Needs D5C0 bit 1 clear, too

    # OPPONENTS POKEMON
    N_POKEMON_OPPONENT_ADDR = 0xD89C
    #   Pokemon 1
    OPPONENT_POKEMON1_ADDR = 0xD89D  # also 0xD8A4
    OPPONENT_POKEMON1_HP_ADDR = [0xD8A5, 0xD8A6]
    OPPONENT_POKEMON1_STATUS_ADDR = 0xD8A8
    OPPONENT_POKEMON1_TYPE1_ADDR = 0xD8A9
    OPPONENT_POKEMON1_TYPE2_ADDR = 0xD8AA
    OPPONENT_POKEMON1_MOVE1_ADDR = 0xD8AC
    OPPONENT_POKEMON1_MOVE2_ADDR = 0xD8AD
    OPPONENT_POKEMON1_MOVE3_ADDR = 0xD8AE
    OPPONENT_POKEMON1_MOVE4_ADDR = 0xD8AF
    OPPONENT_POKEMON1_EXP_ADDR = [0xD8B2, 0xD8B4]
    OPPONENT_POKEMON1_HP_EV_ADDR = [0xD8B5, 0xD8B6]
    OPPONENT_POKEMON1_ATTACK_EV_ADDR = [0xD8B7, 0xD8B8]
    OPPONENT_POKEMON1_DEFENSE_EV_ADDR = [0xD8B9, 0xD8BA]
    OPPONENT_POKEMON1_SPEED_EV_ADDR = [0xD8BB, 0xD8BC]
    OPPONENT_POKEMON1_SPECIAL_EV_ADDR = [0xD8BD, 0xD8BE]
    OPPONENT_POKEMON1_ATTACK_DEFENSE_IV_ADDR = 0xD8BF
    OPPONENT_POKEMON1_SPEED_SPECIAL_IV_ADDR = 0xD8C0
    OPPONENT_POKEMON1_PP_SLOT1_ADDR = 0xD8C1
    OPPONENT_POKEMON1_PP_SLOT2_ADDR = 0xD8C2
    OPPONENT_POKEMON1_PP_SLOT3_ADDR = 0xD8C3
    OPPONENT_POKEMON1_PP_SLOT4_ADDR = 0xD8C4
    OPPONENT_POKEMON1_LEVEL_ADDR = 0xD8C5
    OPPONENT_POKEMON1_MAX_HP_ADDR = [0xD8C6, 0xD8C7]
    OPPONENT_POKEMON1_ATTACK_ADDR = [0xD8C8, 0xD8C9]
    OPPONENT_POKEMON1_DEFENSE_ADDR = [0xD8CA, 0xD8CB]
    OPPONENT_POKEMON1_SPEED_ADDR = [0xD8C, 0xD8CD]
    OPPONENT_POKEMON1_SPECIAL_ADDR = [0xD8CE, 0xD8CF]
    #   Pokemon 2
    OPPONENT_POKEMON2_ADDR = 0xD89E  # also 0xD8D0
    OPPONENT_POKEMON2_HP_ADDR = [0xD8D1, 0xD8D2]
    OPPONENT_POKEMON2_STATUS_ADDR = 0xD8D4
    OPPONENT_POKEMON2_TYPE1_ADDR = 0xD8D5
    OPPONENT_POKEMON2_TYPE2_ADDR = 0xD8D6
    OPPONENT_POKEMON2_MOVE1_ADDR = 0xD8D8
    OPPONENT_POKEMON2_MOVE2_ADDR = 0xD8D9
    OPPONENT_POKEMON2_MOVE3_ADDR = 0xD8DA
    OPPONENT_POKEMON2_MOVE4_ADDR = 0xD8DB
    OPPONENT_POKEMON2_EXP_ADDR = [0xD8DE, 0xD8E0]
    OPPONENT_POKEMON2_HP_EV_ADDR = [0xD8E1, 0xD8E2]
    OPPONENT_POKEMON2_ATTACK_EV_ADDR = [0xD8E3, 0xD8E4]
    OPPONENT_POKEMON2_DEFENSE_EV_ADDR = [0xD8E5, 0xD8E6]
    OPPONENT_POKEMON2_SPEED_EV_ADDR = [0xD8E7, 0xD8E8]
    OPPONENT_POKEMON2_SPECIAL_EV_ADDR = [0xD8E9, 0xD8EA]
    OPPONENT_POKEMON2_ATTACK_DEFENSE_IV_ADDR = 0xD8EB
    OPPONENT_POKEMON2_SPEED_SPECIAL_IV_ADDR = 0xD8EC
    OPPONENT_POKEMON2_PP_SLOT1_ADDR = 0xD8ED
    OPPONENT_POKEMON2_PP_SLOT2_ADDR = 0xD8EE
    OPPONENT_POKEMON2_PP_SLOT3_ADDR = 0xD8EF
    OPPONENT_POKEMON2_PP_SLOT4_ADDR = 0xD8F0
    OPPONENT_POKEMON2_LEVEL_ADDR = 0xD8F1
    OPPONENT_POKEMON2_MAX_HP_ADDR = [0xD8F2, 0xD8F3]
    OPPONENT_POKEMON2_ATTACK_ADDR = [0xD8F4, 0xD8F5]
    OPPONENT_POKEMON2_DEFENSE_ADDR = [0xD8F6, 0xD8F7]
    OPPONENT_POKEMON2_SPEED_ADDR = [0xD8F8, 0xD8F9]
    OPPONENT_POKEMON2_SPECIAL_ADDR = [0xD8FA, 0xD8FB]
    #   Pokemon 3
    OPPONENT_POKEMON3_ADDR = 0xD89F  # also 0xD8FC
    OPPONENT_POKEMON3_HP_ADDR = [0xD8FD, 0xD8FE]
    OPPONENT_POKEMON3_STATUS_ADDR = 0xD900
    OPPONENT_POKEMON3_TYPE1_ADDR = 0xD901
    OPPONENT_POKEMON3_TYPE2_ADDR = 0xD902
    OPPONENT_POKEMON3_MOVE1_ADDR = 0xD904
    OPPONENT_POKEMON3_MOVE2_ADDR = 0xD905
    OPPONENT_POKEMON3_MOVE3_ADDR = 0xD906
    OPPONENT_POKEMON3_MOVE4_ADDR = 0xD907
    OPPONENT_POKEMON3_EXP_ADDR = [0xD90A, 0xD90C]
    OPPONENT_POKEMON3_HP_EV_ADDR = [0xD90D, 0xD90E]
    OPPONENT_POKEMON3_ATTACK_EV_ADDR = [0xD90F, 0xD910]
    OPPONENT_POKEMON3_DEFENSE_EV_ADDR = [0xD911, 0xD912]
    OPPONENT_POKEMON3_SPEED_EV_ADDR = [0xD913, 0xD914]
    OPPONENT_POKEMON3_SPECIAL_EV_ADDR = [0xD915, 0xD916]
    OPPONENT_POKEMON3_ATTACK_DEFENSE_IV_ADDR = 0xD917
    OPPONENT_POKEMON3_SPEED_SPECIAL_IV_ADDR = 0xD918
    OPPONENT_POKEMON3_PP_SLOT1_ADDR = 0xD919
    OPPONENT_POKEMON3_PP_SLOT2_ADDR = 0xD91A
    OPPONENT_POKEMON3_PP_SLOT3_ADDR = 0xD91B
    OPPONENT_POKEMON3_PP_SLOT4_ADDR = 0xD91C
    OPPONENT_POKEMON3_LEVEL_ADDR = 0xD91D
    OPPONENT_POKEMON3_MAX_HP_ADDR = [0xD91E, 0xD91F]
    OPPONENT_POKEMON3_ATTACK_ADDR = [0xD920, 0xD921]
    OPPONENT_POKEMON3_DEFENSE_ADDR = [0xD922, 0xD923]
    OPPONENT_POKEMON3_SPEED_ADDR = [0xD924, 0xD925]
    OPPONENT_POKEMON3_SPECIAL_ADDR = [0xD926, 0xD927]
    #   Pokemon 4
    OPPONENT_POKEMON4_ADDR = 0xD8A0  # also 0xD928
    OPPONENT_POKEMON4_HP_ADDR = [0xD929, 0xD92A]
    OPPONENT_POKEMON4_STATUS_ADDR = 0xD92C
    OPPONENT_POKEMON4_TYPE1_ADDR = 0xD92D
    OPPONENT_POKEMON4_TYPE2_ADDR = 0xD92E
    OPPONENT_POKEMON4_MOVE1_ADDR = 0xD930
    OPPONENT_POKEMON4_MOVE2_ADDR = 0xD931
    OPPONENT_POKEMON4_MOVE3_ADDR = 0xD932
    OPPONENT_POKEMON4_MOVE4_ADDR = 0xD933
    OPPONENT_POKEMON4_EXP_ADDR = [0xD936, 0xD938]
    OPPONENT_POKEMON4_HP_EV_ADDR = [0xD939, 0xD93A]
    OPPONENT_POKEMON4_ATTACK_EV_ADDR = [0xD93B, 0xD93C]
    OPPONENT_POKEMON4_DEFENSE_EV_ADDR = [0xD93D, 0xD93E]
    OPPONENT_POKEMON4_SPEED_EV_ADDR = [0xD93F, 0xD940]
    OPPONENT_POKEMON4_SPECIAL_EV_ADDR = [0xD941, 0xD942]
    OPPONENT_POKEMON4_ATTACK_DEFENSE_IV_ADDR = 0xD943
    OPPONENT_POKEMON4_SPEED_SPECIAL_IV_ADDR = 0xD944
    OPPONENT_POKEMON4_PP_SLOT1_ADDR = 0xD945
    OPPONENT_POKEMON4_PP_SLOT2_ADDR = 0xD946
    OPPONENT_POKEMON4_PP_SLOT3_ADDR = 0xD947
    OPPONENT_POKEMON4_PP_SLOT4_ADDR = 0xD948
    OPPONENT_POKEMON4_LEVEL_ADDR = 0xD949
    OPPONENT_POKEMON4_MAX_HP_ADDR = [0xD94A, 0xD94B]
    OPPONENT_POKEMON4_ATTACK_ADDR = [0xD94C, 0xD94D]
    OPPONENT_POKEMON4_DEFENSE_ADDR = [0xD94E, 0xD94F]
    OPPONENT_POKEMON4_SPEED_ADDR = [0xD950, 0xD951]
    OPPONENT_POKEMON4_SPECIAL_ADDR = [0xD952, 0xD953]
    #   Pokemon 5
    OPPONENT_POKEMON5_ADDR = 0xD8A1  # also 0xD954
    OPPONENT_POKEMON5_HP_ADDR = [0xD955, 0xD956]
    OPPONENT_POKEMON5_STATUS_ADDR = 0xD958
    OPPONENT_POKEMON5_TYPE1_ADDR = 0xD959
    OPPONENT_POKEMON5_TYPE2_ADDR = 0xD95A
    OPPONENT_POKEMON5_MOVE1_ADDR = 0xD95C
    OPPONENT_POKEMON5_MOVE2_ADDR = 0xD95D
    OPPONENT_POKEMON5_MOVE3_ADDR = 0xD95E
    OPPONENT_POKEMON5_MOVE4_ADDR = 0xD95F
    OPPONENT_POKEMON5_EXP_ADDR = [0xD960, 0xD961]
    OPPONENT_POKEMON5_HP_EV_ADDR = [0xD962, 0xD964]
    OPPONENT_POKEMON5_ATTACK_EV_ADDR = [0xD965, 0xD966]
    OPPONENT_POKEMON5_DEFENSE_EV_ADDR = [0xD967, 0xD968]
    OPPONENT_POKEMON5_SPEED_EV_ADDR = [0xD96B, 0xD96C]
    OPPONENT_POKEMON5_SPECIAL_EV_ADDR = [0xD96D, 0xD96E]
    OPPONENT_POKEMON5_ATTACK_DEFENSE_IV_ADDR = 0xD96F
    OPPONENT_POKEMON5_SPEED_SPECIAL_IV_ADDR = 0xD970
    OPPONENT_POKEMON5_PP_SLOT1_ADDR = 0xD971
    OPPONENT_POKEMON5_PP_SLOT2_ADDR = 0xD972
    OPPONENT_POKEMON5_PP_SLOT3_ADDR = 0xD973
    OPPONENT_POKEMON5_PP_SLOT4_ADDR = 0xD974
    OPPONENT_POKEMON5_LEVEL_ADDR = 0xD975
    OPPONENT_POKEMON5_MAX_HP_ADDR = [0xD976, 0xD977]
    OPPONENT_POKEMON5_ATTACK_ADDR = [0xD978, 0xD979]
    OPPONENT_POKEMON5_DEFENSE_ADDR = [0xD97A, 0xD97B]
    OPPONENT_POKEMON5_SPEED_ADDR = [0xD97C, 0xD97D]
    OPPONENT_POKEMON5_SPECIAL_ADDR = [0xD97E, 0xD97F]
    #   Pokemon 6
    OPPONENT_POKEMON6_ADDR = 0xD8A2  # also 0xD980
    OPPONENT_POKEMON6_HP_ADDR = [0xD981, 0xD982]
    OPPONENT_POKEMON6_STATUS_ADDR = 0xD984
    OPPONENT_POKEMON6_TYPE1_ADDR = 0xD985
    OPPONENT_POKEMON6_TYPE2_ADDR = 0xD986
    OPPONENT_POKEMON6_MOVE1_ADDR = 0xD988
    OPPONENT_POKEMON6_MOVE2_ADDR = 0xD989
    OPPONENT_POKEMON6_MOVE3_ADDR = 0xD98A
    OPPONENT_POKEMON6_MOVE4_ADDR = 0xD98B
    OPPONENT_POKEMON6_EXP_ADDR = [0xD98E, 0xD990]
    OPPONENT_POKEMON6_HP_EV_ADDR = [0xD991, 0xD992]
    OPPONENT_POKEMON6_ATTACK_EV_ADDR = [0xD993, 0xD994]
    OPPONENT_POKEMON6_DEFENSE_EV_ADDR = [0xD995, 0xD996]
    OPPONENT_POKEMON6_SPEED_EV_ADDR = [0xD997, 0xD998]
    OPPONENT_POKEMON6_SPECIAL_EV_ADDR = [0xD999, 0xD99A]
    OPPONENT_POKEMON6_ATTACK_DEFENSE_IV_ADDR = 0xD99B
    OPPONENT_POKEMON6_SPEED_SPECIAL_IV_ADDR = 0xD99C
    OPPONENT_POKEMON6_PP_SLOT1_ADDR = 0xD99D
    OPPONENT_POKEMON6_PP_SLOT2_ADDR = 0xD99E
    OPPONENT_POKEMON6_PP_SLOT3_ADDR = 0xD99F
    OPPONENT_POKEMON6_PP_SLOT4_ADDR = 0xD9A0
    OPPONENT_POKEMON6_LEVEL_ADDR = 0xD9A1
    OPPONENT_POKEMON6_MAX_HP_ADDR = [0xD9A2, 0xD9A3]
    OPPONENT_POKEMON6_ATTACK_ADDR = [0xD9A4, 0xD9A5]
    OPPONENT_POKEMON6_DEFENSE_ADDR = [0xD9A6, 0xD9A7]
    OPPONENT_POKEMON6_SPEED_ADDR = [0xD9A8, 0xD9A9]
    OPPONENT_POKEMON6_SPECIAL_ADDR = [0xD9AA, 0xD9AB]

    # TIME
    GAME_TIME_HOURS_FLAG_ADDR = 0xDA40
    GAME_TIME_HOURS_ADDR = 0xDA41
    GAME_TIME_MINUTES_FLAG_ADDR = 0xDA42
    GAME_TIME_MINUTES_ADDR = 0xDA43
    GAME_TIME_SECONDS_ADDR = 0xDA44
    GAME_TIME_FRAMES_ADDR = 0xDA45

    # TEXT TABLE
    TEXT_TABLE = {b'\x50': 'END_MARKER',
                  b'\x4F': '=',
                  b'\x57': '#',
                  b'\x51': '*',
                  b'\x52': 'A1',
                  b'\x53': 'A2',
                  b'\x54': 'POKé',
                  b'\x55': '+',
                  b'\x58': '$',
                  b'\x75': '...',
                  b'\x7F': ' ',
                  b'\x80': 'A',
                  b'\x81': 'B',
                  b'\x82': 'C',
                  b'\x83': 'D',
                  b'\x84': 'E',
                  b'\x85': 'F',
                  b'\x86': 'G',
                  b'\x87': 'H',
                  b'\x88': 'I',
                  b'\x89': 'J',
                  b'\x8A': 'K',
                  b'\x8B': 'L',
                  b'\x8C': 'M',
                  b'\x8D': 'N',
                  b'\x8E': 'O',
                  b'\x8F': 'P',
                  b'\x90': 'Q',
                  b'\x91': 'R',
                  b'\x92': 'S',
                  b'\x93': 'T',
                  b'\x94': 'U',
                  b'\x95': 'V',
                  b'\x96': 'W',
                  b'\x97': 'X',
                  b'\x98': 'Y',
                  b'\x99': 'Z',
                  b'\x9A': '(',
                  b'\x9B': ')',
                  b'\x9C': ':',
                  b'\x9D': ';',
                  b'\x9E': '[',
                  b'\x9F': ']',
                  b'\xA0': 'a',
                  b'\xA1': 'b',
                  b'\xA2': 'c',
                  b'\xA3': 'd',
                  b'\xA4': 'e',
                  b'\xA5': 'f',
                  b'\xA6': 'g',
                  b'\xA7': 'h',
                  b'\xA8': 'i',
                  b'\xA9': 'j',
                  b'\xAA': 'k',
                  b'\xAB': 'l',
                  b'\xAC': 'm',
                  b'\xAD': 'n',
                  b'\xAE': 'o',
                  b'\xAF': 'p',
                  b'\xB0': 'q',
                  b'\xB1': 'r',
                  b'\xB2': 's',
                  b'\xB3': 't',
                  b'\xB4': 'u',
                  b'\xB5': 'v',
                  b'\xB6': 'w',
                  b'\xB7': 'x',
                  b'\xB8': 'y',
                  b'\xB9': 'z',
                  b'\xBA': 'é',
                  b'\xBB': "'d",
                  b'\xBC': "'l",
                  b'\xBD': "'s",
                  b'\xBE': "'t",
                  b'\xBF': "'v",
                  b'\xE0': "'",
                  b'\xE1': 'PK',
                  b'\xE2': 'MN',
                  b'\xE3': '-',
                  b'\xE4': "'r",
                  b'\xE5': "'m",
                  b'\xE6': '?',
                  b'\xE7': '!',
                  b'\xE8': '.',
                  b'\xED': '→',
                  b'\xEE': '↓',
                  b'\xEF': '♂',
                  b'\xF0': '¥',
                  b'\xF1': '×',
                  b'\xF3': '/',
                  b'\xF4': ',',
                  b'\xF5': '♀',
                  b'\xF6': '0',
                  b'\xF7': '1',
                  b'\xF8': '2',
                  b'\xF9': '3',
                  b'\xFA': '4',
                  b'\xFB': '5',
                  b'\xFC': '6',
                  b'\xFD': '7',
                  b'\xFE': '8',
                  b'\xFF': '9',
                  b'\x00': ''}

    def __init__(self, window_type: str = 'headless', save_path: str | None = None, load_path: str | None = None,
                 start_button: bool = False, max_actions: int | None = None, all_actions: bool = False,
                 subtask: str | None = None,
                 ):
        assert window_type == 'SDL2' or window_type == 'headless'
        assert subtask is None or subtask in ['first_pokemon', 'enter_first_route', 'beat_first_gym',
                                              'get_six_pokemon', 'evolve_pokemon']
        super().__init__()
        # episode truncation
        self.max_actions = max_actions
        self.actions_taken = 0
        self.window_type = window_type

        self.subtask = subtask
        if self.subtask == 'enter_first_route':
            self.visited = list()

        with importlib.resources.path('gle.rom', "Pokemon Red (UE) [S][!].gb") as rom_path:
            self.pyboy = PyBoy(
                rom_path,
                window_type=self.window_type
            )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title()}')
        assert self.pyboy.cartridge_title() == 'POKEMON RED' or self.pyboy.cartridge_title() == 'POKEMON BLUE', 'The cartridge title should be POKEMON BLUE or POKEMON RED, international edition not Japan edition (RAM addresses are different in the latter)!'

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
        _, info = self.reset()

        if self.subtask == 'evolve_pokemon':
            self.pokemon_ids = [info['player'][f'pokemon_{i}']['id'] for i in range(6)]

    #   ******************************************************
    #               GYMNASIUM OVERRIDING FUNCTION
    #   ******************************************************
    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        while self.get_joypad_simulation_info()['joypad_simulation']:
            self.pyboy.tick()

        self.take_action(action)
        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True

        obs = self.render()
        info = self.get_info()

        # ULTIMATE TASK
        if info['player']['location_id'] == 0x76:   # Pokemon League: Hall of Fame
            done = True
        # SUBTASKS
        if self.subtask == 'enter_first_route':
            if info['player']['location_id'] == 0 and 0 not in self.visited:
                reward = 1
                self.visited.append(0x00)
            elif info['player']['location_id'] == 12:
                reward = 10
                done = True
            else:
                reward = 0
        elif self.subtask == 'first_pokemon':
            reward = 1 if info['player']['pokemon_1']['id'] != 255 else 0
        elif self.subtask == 'get_six_pokemon':
            reward = 0
            for i in range(6):
                reward += 1 if info['player'][f'pokemon_{i}']['id'] != 255 else 0
            reward = 1 if reward == 6 else 0
        elif self.subtask == 'beat_first_gym':
            reward = 1 if 0 in info['player']['badges'] else 0
        elif self.subtask == 'evolve_pokemon':
            current_pokemon_ids = [info['player'][f'pokemon_{i}']['id'] for i in range(6)]
            evolved = False
            for i in range(6):
                evolved = True if current_pokemon_ids[i] - self.pokemon_ids[i] == 1 else evolved
            reward = 1 if evolved else 0
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
        self.actions_taken = 0
        self.visited = list()

        if self.load_path is None:
            self.skip_game_initial_video()

        info = self.get_info()

        if self.subtask == 'evolve_pokemon':
            self.pokemon_ids = [info['player'][f'pokemon_{i}']['id'] for i in range(6)]
        return self.render(), info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        screen_obs = self.screen.screen_ndarray()  # (144, 160, 3)
        return screen_obs.reshape((screen_obs.shape[2], screen_obs.shape[0], screen_obs.shape[1]))  # (3, 144, 160)

    def close(self):
        self.pyboy.stop(save=False)
        with importlib.resources.path('gle.rom', "Pokemon Red (UE) [S][!].gb") as rom_path:
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
            if self.pyboy.frame_count == 300 or self.pyboy.frame_count == 700 or self.pyboy.frame_count == 900:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            if self.pyboy.frame_count == 305 or self.pyboy.frame_count == 705 or self.pyboy.frame_count == 905:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.pyboy.frame_count == 1100:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            if self.pyboy.frame_count == 1105:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
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
                **self.get_opponent_info(),
                **self.get_time_info(),
                **self.get_item_info(),
                **self.get_flag_info(),
                **self.get_pokedex_info(),
                **self.get_battle_info(),
                **self.get_joypad_simulation_info()}

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_player_info(self) -> dict:
        player_info = dict()

        player_info['name'] = self.decode_name(self.PLAYER_NAME_ADDRS[0], self.PLAYER_NAME_ADDRS[1])
        player_info['money'] = self.read_money()
        player_info['rival_name'] = self.decode_name(self.RIVAL_NAME_ADDRS[0], self.RIVAL_NAME_ADDRS[1])
        player_info['x_pos'] = self.pyboy.get_memory_value(self.CURRENT_X_POS_ADDR)
        player_info['y_pos'] = self.pyboy.get_memory_value(self.CURRENT_Y_POS_ADDR)
        player_info['location_id'] = self.pyboy.get_memory_value(self.CURRENT_MAP_NUMBER_ADDR)
        player_info['badges'] = self.decode_badges()
        player_info['on_bike'] = self.pyboy.get_memory_value(self.ON_BIKE_ADDR) == 0x01

        player_info['pokemon_in_party'] = self.pyboy.get_memory_value(self.N_POKEMON_IN_PARTY_ADDR)

        # Pokemon 1 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON1_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON1_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_ADDR[1]))
        player_info['pokemon_1'] = pokemon.copy()
        # Pokemon 2 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_ADDR[0]) + (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON2_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON2_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_ADDR[1]))
        player_info['pokemon_2'] = pokemon.copy()
        # Pokemon 3 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON3_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON3_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_ADDR[1]))
        player_info['pokemon_3'] = pokemon.copy()
        # Pokemon 4 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON4_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON4_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_ADDR[1]))
        player_info['pokemon_4'] = pokemon.copy()
        # Pokemon 5 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON5_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON5_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_ADDR[1]))
        player_info['pokemon_5'] = pokemon.copy()
        # Pokemon 6 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.PARTY_POKEMON6_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.PARTY_POKEMON6_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_EV_ADDR[1]))
        pokemon['atk_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_DEFENSE_IV_ADDR) >> 4) & 0b1111
        pokemon['def_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_DEFENSE_IV_ADDR) & 0b1111
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_ADDR[1]))
        player_info['pokemon_6'] = pokemon.copy()
        return {'player': player_info}

    def read_money(self) -> int:
        return (100 * 100 * self.read_bcd(self.pyboy.get_memory_value(self.MONEY_ADDRS[0])) +
                100 * self.read_bcd(self.pyboy.get_memory_value(self.MONEY_ADDRS[1])) +
                self.read_bcd(self.pyboy.get_memory_value(self.MONEY_ADDRS[2])))

    def get_opponent_info(self) -> dict:
        opponent_info = dict()

        opponent_info['pokemon_in_party'] = self.pyboy.get_memory_value(self.N_POKEMON_IN_PARTY_ADDR)

        # Pokemon 1 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON1_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] =(self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPEED_EV_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_SPECIAL_ADDR[1]))
        opponent_info['pokemon_1'] = pokemon.copy()
        # Pokemon 2 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON2_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_SPECIAL_ADDR[1]))
        opponent_info['pokemon_2'] = pokemon.copy()
        # Pokemon 3 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON3_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_SPECIAL_ADDR[1]))
        opponent_info['pokemon_3'] = pokemon.copy()
        # Pokemon 4 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON4_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_SPECIAL_ADDR[1]))
        opponent_info['pokemon_4'] = pokemon.copy()
        # Pokemon 5 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON5_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_SPECIAL_ADDR[1]))
        opponent_info['pokemon_5'] = pokemon.copy()
        # Pokemon 6 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_HP_ADDR[1]))
        pokemon['status'] = self.read_status(self.OPPONENT_POKEMON6_STATUS_ADDR)
        pokemon['type1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_TYPE1_ADDR)
        pokemon['type2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_TYPE2_ADDR)
        pokemon['move1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MOVE1_ADDR)
        pokemon['move2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MOVE2_ADDR)
        pokemon['move3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MOVE3_ADDR)
        pokemon['move4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MOVE4_ADDR)
        pokemon['exp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_EXP_ADDR[0]) * (2 ** 8)
                          + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_EXP_ADDR[1]))
        pokemon['hp_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_HP_EV_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_HP_EV_ADDR[1]))
        pokemon['atk_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ATTACK_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ATTACK_EV_ADDR[1]))
        pokemon['def_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_DEFENSE_EV_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_DEFENSE_EV_ADDR[1]))
        pokemon['speed_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPEED_EV_ADDR[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPEED_EV_ADDR[1]))
        pokemon['special_ev'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPECIAL_EV_ADDR[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPECIAL_EV_ADDR[1]))
        pokemon['atk_def_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ATTACK_DEFENSE_IV_ADDR)
        pokemon['speed_special_iv'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPEED_SPECIAL_IV_ADDR)
        pokemon['pp1'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_PP_SLOT4_ADDR)
        pokemon['level'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_LEVEL_ADDR)
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPEED_ADDR[1]))
        pokemon['special'] = (self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPECIAL_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_SPECIAL_ADDR[1]))
        opponent_info['pokemon_6'] = pokemon.copy()
        return {'opponent': opponent_info}

    def get_time_info(self) -> dict:
        return {'time': {'hours': self.pyboy.get_memory_value(self.GAME_TIME_HOURS_ADDR),
                         'minutes': self.pyboy.get_memory_value(self.GAME_TIME_MINUTES_ADDR),
                         'seconds': self.pyboy.get_memory_value(self.GAME_TIME_SECONDS_ADDR)}}

    def get_item_info(self) -> dict:
        item_info = dict()
        item_info['total_items'] = self.pyboy.get_memory_value(self.TOTAL_ITEM_COUNT_ADDR)
        for i, addr in enumerate(self.ITEMS_ADDRS):
            item_info[f'item{i}'] = {'id': self.pyboy.get_memory_value(addr),
                                     'quantity': self.pyboy.get_memory_value(self.ITEM_QUANTITIES_ADDRS[i])}
        return {'items': item_info}

    def get_flag_info(self) -> dict:
        event_flags_info = dict()
        event_flags_info['starters_back'] = self.pyboy.get_memory_value(self.STARTERS_BACK_FLAG_ADDR) == 1
        event_flags_info['mewtwo_appears'] = self.pyboy.get_memory_value(self.MEWTWO_APPEARS_FLAG_ADDR) == 1
        event_flags_info['have_town_map'] = self.pyboy.get_memory_value(self.HAVE_TOWN_MAP_FLAG_ADDR) == 1
        event_flags_info['have_oak_parcel'] = self.pyboy.get_memory_value(self.HAVE_OAK_PARCEL_ADDR) == 1
        event_flags_info['fossilized_pokemon'] = self.pyboy.get_memory_value(self.FOSSILIZED_POKEMON_FLAG_ADDR) == 1
        event_flags_info['got_lapras'] = self.pyboy.get_memory_value(self.G0T_LAPRAS_ADDR) == 1
        event_flags_info['fought_giovanni'] = self.pyboy.get_memory_value(self.FOUGHT_GIOVANNI_ADDR) == 1
        event_flags_info['fought_misty'] = self.pyboy.get_memory_value(self.FOUGHT_MISTY_ADDR) == 1
        event_flags_info['fought_lt_surge'] = self.pyboy.get_memory_value(self.FOUGHT_LT_SURGE_ADDR) == 1
        event_flags_info['fought_erika'] = self.pyboy.get_memory_value(self.FOUGHT_ERIKA_ADDR) == 1
        event_flags_info['fought_articuno'] = self.pyboy.get_memory_value(self.FOUGHT_ARTICUNO_ADDR) == 1
        event_flags_info['fought_koga'] = self.pyboy.get_memory_value(self.FOUGHT_KOGA_ADDR) == 1
        event_flags_info['fought_blaine'] = self.pyboy.get_memory_value(self.FOUGHT_BLAINE_ADDR) == 1
        event_flags_info['fought_sabrina'] = self.pyboy.get_memory_value(self.FOUGHT_SABRINA_ADDR) == 1
        event_flags_info['fought_zapdos'] = self.pyboy.get_memory_value(self.FOUGHT_ZAPDOS_ADDR) == 1
        event_flags_info['fought_snorlax_vermilion'] = self.pyboy.get_memory_value(
            self.FOUGHT_SNORLAX_VERMILION_ADDR) == 1
        event_flags_info['fought_snorlax_celadon'] = self.pyboy.get_memory_value(self.FOUGHT_SNORLAX_CELADON_ADDR) == 1
        event_flags_info['fought_moltres'] = self.pyboy.get_memory_value(self.FOUGHT_MOLTRES_ADDR) == 1
        event_flags_info['is_ss_anne_here'] = self.pyboy.get_memory_value(self.IS_SS_ANNE_HERE_ADDR) == 1
        event_flags_info['mewtwo_can_be_caught'] = self.pyboy.get_memory_value(self.MEWTWO_CAN_BE_CAUGHT) == 1
        return {'event_flags': event_flags_info}

    def get_pokedex_info(self) -> dict:
        pokedex_caught = list()
        for i, addr in enumerate(range(self.POKEDEX_1_152_ADDRS[0], self.POKEDEX_1_152_ADDRS[1] + 1)):
            pokedex_caught += [i * 8 + ones for ones in self.list_one_bits_locations(addr)]
        pokedex_seen = list()
        for i, addr in enumerate(range(self.POKEDEX_SEEN_1_152_ADDRS[0], self.POKEDEX_SEEN_1_152_ADDRS[1] + 1)):
            pokedex_seen += [i * 8 + ones for ones in self.list_one_bits_locations(addr)]
        return {'pokedex': {'caught': pokedex_caught, 'seen': pokedex_seen}}

    def get_battle_info(self) -> dict:
        enemy_info = dict()
        enemy_info['id'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_ID_ADDR)
        enemy_info['name'] = self.decode_name(self.ENEMY_NAME_ADDRS[0], self.ENEMY_NAME_ADDRS[1])
        enemy_info['level'] = self.pyboy.get_memory_value(self.ENEMY_LEVEL_ADDR)
        enemy_info['move'] = self.pyboy.get_memory_value(self.ENEMY_MOVE_ADDR)
        enemy_info['move_effect'] = self.pyboy.get_memory_value(self.ENEMY_MOVE_EFFECT_ADDR)
        enemy_info['move_type'] = self.pyboy.get_memory_value(self.ENEMY_MOVE_TYPE_ADDR)
        enemy_info['move_accuracy'] = self.pyboy.get_memory_value(self.ENEMY_MOVE_ACCURACY_ADDR)
        enemy_info['move_max_pp'] = self.pyboy.get_memory_value(self.ENEMY_MOVE_MAX_PP_ADDR)
        enemy_info['hp'] = (self.pyboy.get_memory_value(self.ENEMY_HP_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.ENEMY_HP_ADDR[1]))
        enemy_info['status'] = self.read_status(self.ENEMY_STATUS_ADDR)
        enemy_info['type1'] = self.pyboy.get_memory_value(self.ENEMY_TYPE1_ADDR)
        enemy_info['type2'] = self.pyboy.get_memory_value(self.ENEMY_TYPE2_ADDR)
        enemy_info['move1'] = self.pyboy.get_memory_value(self.ENEMY_MOVE1_ADDR)
        enemy_info['move2'] = self.pyboy.get_memory_value(self.ENEMY_MOVE2_ADDR)
        enemy_info['move3'] = self.pyboy.get_memory_value(self.ENEMY_MOVE3_ADDR)
        enemy_info['move4'] = self.pyboy.get_memory_value(self.ENEMY_MOVE4_ADDR)
        enemy_info['attack_dv'] = self.pyboy.get_memory_value(self.ENEMY_ATTACK_DEFENSE_DVS_ADDR) & 0b1111
        enemy_info['defense_dv'] = (self.pyboy.get_memory_value(self.ENEMY_ATTACK_DEFENSE_DVS_ADDR) >> 4) & 0b1111
        enemy_info['speed_dv'] = self.pyboy.get_memory_value(self.ENEMY_SPEED_SPECIAL_DVS_ADDR) & 0b1111
        enemy_info['special_dv'] = (self.pyboy.get_memory_value(self.ENEMY_SPEED_SPECIAL_DVS_ADDR) >> 4) & 0b1111
        enemy_info['max_hp'] = (self.pyboy.get_memory_value(self.ENEMY_MAX_HP_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.ENEMY_MAX_HP_ADDRS[1]))
        enemy_info['attack'] = (self.pyboy.get_memory_value(self.ENEMY_ATTACK_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.ENEMY_ATTACK_ADDRS[1]))
        enemy_info['defense'] = (self.pyboy.get_memory_value(self.ENEMY_DEFENSE_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.ENEMY_DEFENSE_ADDRS[1]))
        enemy_info['speed'] = (self.pyboy.get_memory_value(self.ENEMY_SPEED_ADDRS[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.ENEMY_SPEED_ADDRS[1]))
        enemy_info['special'] = (self.pyboy.get_memory_value(self.ENEMY_SPECIAL_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.ENEMY_SPECIAL_ADDRS[1]))
        enemy_info['pp1'] = self.pyboy.get_memory_value(self.ENEMY_PP_SLOT1_ADDR)
        enemy_info['pp2'] = self.pyboy.get_memory_value(self.ENEMY_PP_SLOT2_ADDR)
        enemy_info['pp3'] = self.pyboy.get_memory_value(self.ENEMY_PP_SLOT3_ADDR)
        enemy_info['pp4'] = self.pyboy.get_memory_value(self.ENEMY_PP_SLOT4_ADDR)
        enemy_info['base_hp'] = self.pyboy.get_memory_value(self.ENEMY_BASE_STATS_ADDRS[0])
        enemy_info['base_attack'] = self.pyboy.get_memory_value(self.ENEMY_BASE_STATS_ADDRS[0] + 1)
        enemy_info['base_defense'] = self.pyboy.get_memory_value(self.ENEMY_BASE_STATS_ADDRS[0] + 2)
        enemy_info['base_speed'] = self.pyboy.get_memory_value(self.ENEMY_BASE_STATS_ADDRS[0] + 3)
        enemy_info['base_special'] = self.pyboy.get_memory_value(self.ENEMY_BASE_STATS_ADDRS[0] + 4)
        enemy_info['catch_rate'] = self.pyboy.get_memory_value(self.ENEMY_CATCH_RATE_ADDR)
        enemy_info['base_experience'] = self.pyboy.get_memory_value(self.ENEMY_BASE_EXPERIENCE_ADDR)

        player_info = dict()
        player_info['id'] = self.pyboy.get_memory_value(self.PLAYER_POKEMON_ID_ADDR)
        player_info['number'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_NUMBER_ADDR)
        player_info['name'] = self.decode_name(self.IN_BATTLE_POKEMON_NAME_ADDRS[0], self.IN_BATTLE_POKEMON_NAME_ADDRS[1])
        player_info['level'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_LEVEL_ADDR)
        player_info['move'] = self.pyboy.get_memory_value(self.PLAYER_MOVE_ID_ADDR)
        player_info['move_effect'] = self.pyboy.get_memory_value(self.PLAYER_MOVE_EFFECT_ADDR)
        player_info['move_type'] = self.pyboy.get_memory_value(self.PLAYER_MOVE_TYPE_ADDR)
        player_info['move_accuracy'] = self.pyboy.get_memory_value(self.PLAYER_MOVE_ACCURACY_ADDR)
        player_info['move_max_pp'] = self.pyboy.get_memory_value(self.PLAYER_MOVE_MAX_PP_ADDR)
        player_info['hp'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_HP_ADDRS[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_HP_ADDRS[1]))
        player_info['status'] = self.read_status(self.IN_BATTLE_POKEMON_STATUS_ADDR)
        player_info['type1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_TYPE1_ADDR)
        player_info['type2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_TYPE2_ADDR)
        player_info['move1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE1_ADDR)
        player_info['move2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE2_ADDR)
        player_info['move3'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE3_ADDR)
        player_info['move4'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE4_ADDR)
        player_info['attack_dv'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_ATTACK_DEFENSE_DVS_ADDR) & 0b1111
        player_info['defense_dv'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_ATTACK_DEFENSE_DVS_ADDR) >> 4) & 0b1111
        player_info['speed_dv'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPEED_SPECIAL_DVS_ADDR) & 0b1111
        player_info['special_dv'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPEED_SPECIAL_DVS_ADDR) >> 4) & 0b1111
        player_info['max_hp'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MAX_HP_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MAX_HP_ADDRS[1]))
        player_info['attack'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_ATTACK_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_ATTACK_ADDRS[1]))
        player_info['defense'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_DEFENSE_ADDRS[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_DEFENSE_ADDRS[1]))
        player_info['speed'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPEED_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPEED_ADDRS[1]))
        player_info['special'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPECIAL_ADDRS[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SPECIAL_ADDRS[1]))
        player_info['pp1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP_SLOT1_ADDR)
        player_info['pp2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP_SLOT2_ADDR)
        player_info['pp3'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP_SLOT3_ADDR)
        player_info['pp4'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP_SLOT4_ADDR)

        return {'battle':
                    {
                        'player_turn': self.pyboy.get_memory_value(self.WHOSE_TURN_ADDR) == 0,
                        'enemy': enemy_info,
                        'player': player_info,
                        'critical_ohko_flag': self.pyboy.get_memory_value(self.CRITICAL_OHKO_FLAG_ADDR),
                        'hooked': self.pyboy.get_memory_value(self.HOOKED_POKEMON_FLAG_ADDR),
                        'damage_amount': self.pyboy.get_memory_value(self.AMOUNT_DAMAGE_ADDR)
                    }
                }

    def get_joypad_simulation_info(self) -> dict:
        return {'joypad_simulation': self.pyboy.get_memory_value(self.JOYPAD_SIMULATION_ADDR) != 0x00}
    
    #   ******************************************************
    #                        UTILITIES
    #   ******************************************************
    def read_status(self, addr: int) -> dict:
        status_info = self.pyboy.get_memory_value(addr)
        return {'paralyzed': (0b01000000 & status_info) == 0b01000000,
                'frozen': (0b00100000 & status_info) == 0b00100000,
                'burned': (0b00010000 & status_info) == 0b00010000,
                'poisoned': (0b00001000 & status_info) == 0b00001000,
                'sleep_counter': 0b00000111 & status_info}

    def decode_name(self, starting_addr: int, final_addr: int) -> str:
        name = ''
        for addr in range(starting_addr, final_addr + 1):
            char = self.TEXT_TABLE[self.pyboy.get_memory_value(addr).to_bytes(1, byteorder='big')]
            if char == 'END_MARKER':
                break
            name += char
        return name

    def decode_badges(self) -> list:
        return self.list_one_bits_locations(self.BADGES_ADDR)

    def list_one_bits_locations(self, addr: int) -> list:
        value = self.pyboy.get_memory_value(addr)
        ones = list()
        for i in range(8):
            if value & (1 << i):
                ones.append(i)
        return ones

    def read_bcd(self, value):
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
