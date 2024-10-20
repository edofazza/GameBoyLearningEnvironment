from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from pyboy import PyBoy, WindowEvent
from gymnasium import Env, spaces

from general import ALL_ACTIONS, ALL_RELEASE_ACTIONS


class PokemonGoldSilver(Env):
    # IN BATTLE
    #       ID    Battle Type
    #       --    -----------
    #       00    Normal
    #       02    No Pokémon (guaranteed escape)
    #       03    Tutorial battle
    #       04    Hooked Pokémon
    #       05    Suicune
    #       06    Bug catching contest
    #       07    Shiny Gyarados (14 ATK / 10 DEF / 10 SPEED / 10 SPC)
    #       08    Headbutt
    #       0C    Entei, Raikou
    BATTLE_TYPE_ADDR = 0xD116
    #   Your Pokemon
    IN_BATTLE_POKEMON_ITEM_ADDR = 0xCB0D
    IN_BATTLE_POKEMON_MOVE1_ADDR = 0xCB0E
    IN_BATTLE_POKEMON_MOVE2_ADDR = 0xCB0F
    IN_BATTLE_POKEMON_MOVE3_ADDR = 0xCB10
    IN_BATTLE_POKEMON_MOVE4_ADDR = 0xCB11
    IN_BATTLE_POKEMON_PP1_ADDR = 0xCB14
    IN_BATTLE_POKEMON_PP2_ADDR = 0xCB15
    IN_BATTLE_POKEMON_PP3_ADDR = 0xCB16
    IN_BATTLE_POKEMON_PP4_ADDR = 0xCB17
    IN_BATTLE_POKEMON_STATUS_ADDR = 0xCB1A
    IN_BATTLE_POKEMON_HP_ADDRS = [0xCB1C, 0xCB1D]
    IN_BATTLE_POKEMON_TYPE1_ADDR = 0xCB2A
    IN_BATTLE_POKEMON_TYPE2_ADDR = 0xCB2B
    IN_BATTLE_POKEMON_SUBSTITUTE_ADDR = 0xCB49
    IN_BATTLE_MONEY_EARNED_ADDRS = [0xCB65, 0xCB66]
    IN_BATTLE_EXP_GIVEN_ADDRS = [0xCF7E, 0xCF7F]
    IN_BATTLE_CURRENT_ATTACK_ADDR = 0xCBC1
    # Opposing Pokemon
    WHAT_MOVES_WILL_HAVE1_ADDR = 0xCC13
    WHAT_MOVES_WILL_HAVE2_ADDR = 0xCC14
    WHAT_MOVES_WILL_HAVE3_ADDR = 0xCC15
    WHAT_MOVES_WILL_HAVE4_ADDR = 0xCC16
    ENEMY_POKEMON_ITEM_ADDR = 0xD0F0
    ENEMY_POKEMON_MOVE1_ADDR = 0xD0F1  # also 0xD149
    ENEMY_POKEMON_MOVE2_ADDR = 0xD0F2  # also 0xD14A
    ENEMY_POKEMON_MOVE3_ADDR = 0xD0F3  # also 0xD14B
    ENEMY_POKEMON_MOVE4_ADDR = 0xD0F4  # also 0xD14C
    ENEMY_POKEMON_ATTACK_DEFENSE_DV_ADDR = 0xD0F5
    ENEMY_POKEMON_SPEED_SPECIAL_DV_ADDR = 0xD0F6
    ENEMY_POKEMON_LEVEL_ADDR = 0xD0FC
    ENEMY_POKEMON_STATUS_ADDR = 0xD0FD
    ENEMY_POKEMON_CURRENT_HP_ADDRS = [0xD0FF, 0xD100]  # two-byte big-endian
    ENEMY_POKEMON_TOTAL_HP_ADDRS = [0xD101, 0xD102]
    ENEMY_POKEMON_ATTACK_ADDRS = [0xD103, 0xD104]
    ENEMY_POKEMON_DEFENSE_ADDRS = [0xD105, 0xD106]
    ENEMY_POKEMON_SPEED_ADDRS = [0xD107, 0xD108]
    ENEMY_POKEMON_SPECIAL_ATTACK_ADDRS = [0xD109, 0xD10A]
    ENEMY_POKEMON_SPECIAL_DEFENSE_ADDRS = [0xD10B, 0xD10C]
    ENEMY_POKEMON_SEX_ADDR = 0xD119
    ENEMY_POKEMON_TYPE1_ADDR = 0xD127
    ENEMY_POKEMON_TYPE2_ADDR = 0xD128
    ENEMY_POKEMON_DAMAGE_ADDRS = [0xD141,
                                  0xD142]  # Enemy Damage (big-endian) that attack is about to do. Max damage is shown one frame before actual damage.
    ENEMY_POKEMON_MAGNITUDE_ADDR = 0xD151  # Enemy Magnitude number (if using Magnitude); otherwise unknown (https://bulbapedia.bulbagarden.net/wiki/Magnitude_(move))

    # OPPONENT
    N_POKEMON_OPPONENT_ADDR = 0xDD55
    OPPONENT_POKEMON1_ADDR = 0xDD56
    OPPONENT_POKEMON2_ADDR = 0xDD57
    OPPONENT_POKEMON3_ADDR = 0xDD58
    OPPONENT_POKEMON4_ADDR = 0xDD59
    OPPONENT_POKEMON5_ADDR = 0xDD5A
    OPPONENT_POKEMON6_ADDR = 0xDD5B
    OPPONENT_POKEMON1_NAME_ADDR = [0xDEBF, 0xDEC8]
    OPPONENT_POKEMON2_NAME_ADDR = [0xEC9, 0xDED3]
    OPPONENT_POKEMON3_NAME_ADDR = [0xDED4, 0xDEDE]
    OPPONENT_POKEMON4_NAME_ADDR = [0xDEDF, 0xDEE9]
    OPPONENT_POKEMON5_NAME_ADDR = [0xDEEA, 0xDEF4]
    OPPONENT_POKEMON6_NAME_ADDR = [0xDEF5, 0xDEFF]

    # WILD POKEMON
    WILD_POKEMON_NUMBER_ADDR = 0xD0ED
    WILD_POKEMON_LEVEL_ADDR = 0xD0FC

    # LOCATION
    X_POS_ADDR = 0xD20D
    Y_POS_ADDR = 0xD20E
    WHAT_MAP_BANK_ADDR = 0xDA00
    WHAT_MAP_NUMBER_ADDR = 0xDA01
    ON_BIKE_ADDR = 0xD682
    REPEL_STEP_LEFT_ADDR = 0xD9EB
    PARK_TIME_ADDR = 0xD193

    # BUG CONTEST
    BUG_CONTEST_POKEMON_ID_ADDR = 0xDCE7
    BUG_CONTEST_POKEMON_LEVEL_ADDR = 0xDCE6
    BUG_CONTEST_POKEMON_CURRENT_HP_ADDRS = [0xDD09, 0xDD0A]
    BUG_CONTEST_POKEMON_TOTAL_HP_ADDRS = [0xDD0B, 0xDD0C]
    BUG_CONTEST_POKEMON_ATTACK_ADDRS = [0xDD0D, 0xDD0E]
    BUG_CONTEST_POKEMON_DEFENSE_ADDRS = [0xDD0F, 0xDD10]
    BUG_CONTEST_POKEMON_SPEED_ADDRS = [0xDD11, 0xDD12]
    BUG_CONTEST_POKEMON_SPECIAL_ATTACK_ADDRS = [0xDD13, 0xDD14]
    BUG_CONTEST_POKEMON_SPECIAL_DEFENSE_ADDRS = [0xDD15, 0xDD16]

    # GAME SETTINGS
    PLAYER_NAME_ADDRS = [0xD1A3, 0xD1AC]
    RIVAL_NAME_ADDRS = [0xD1BC, 0xD1B5]
    MONEY_ADDRS = [0xD573, 0xD574, 0xD575]
    MOTHER_HELD_MONEY_ADDRS = [0xD576, 0xD577, 0xD578]
    CASINO_COINS_ADDRS = [0xD57A, 0xD57B]
    JOHTO_BADGES_ADDR = 0xD57C
    #   $01 = Falkner
    #   $02 = Bugsy
    #   $04 = Whitney
    #   $08 = Morty
    #   $10 = Jasmine
    #   $20 = Chuck
    #   $40 = Pryce
    #   $80 = Clair
    # Above values are in hex. To get multiple badges e.g. Whitney and Morty,
    # add their values in hex (04h+08h = 0Ch) to get both badges. Adding all
    # the values will add up to FFh, therefore, that's the value you use to
    # get all the badges.
    KANTO_BADGES_ADDR = 0xD57D

    # ITEMS
    TM_ADDRS = [0xD57E, 0xD5AF]  # 50 addresses https://pokemondb.net/gold-silver/tms
    HM_ADDRS = [0xD5B0, 0xD5B6]  # 7 addresses https://pokemondb.net/gold-silver/hms
    #   Normal items
    TOTAL_ITEM_COUNT_ADDR = 0xD5B7
    ITEM_ADDRS = [
        0xD5B8, 0xD5BA, 0xD5BC, 0xD5BE, 0xD5C0,
        0xD5C2, 0xD5C4, 0xD5C6, 0xD5C8, 0xD5CA,
        0xD5CC, 0xD5CE, 0xD5D0, 0xD5D2, 0xD5D4,
        0xD5D6, 0xD5D8, 0xD5DA, 0xD5DC, 0xD5DE
    ]
    ITEM_QUANTITIES_ADDRS = [
        0xD5B9, 0xD5BB, 0xD5BD, 0xD5BF, 0xD5C1,
        0xD5C3, 0xD5C5, 0xD5C7, 0xD5C9, 0xD5CB,
        0xD5CD, 0xD5CF, 0xD5D1, 0xD5D3, 0xD5D5,
        0xD5D7, 0xD5D9, 0xD5DB, 0xD5DD, 0xD5DF
    ]

    #   Key items
    TOTAL_KEY_ITEMS_COUNT_ADDR = 0xD5E1
    KEY_ITEMS_ADDRS = [0xD5E2,
                       0xD5FA]  # 18 elements https://bulbapedia.bulbagarden.net/wiki/List_of_Key_Items_(Generation_II)
    #   Balls
    TOTAL_BALLS_COUNT_ADDR = 0xD5FC
    BALL_ADDRS = [
        0xD5FD, 0xD5FF, 0xD601, 0xD603, 0xD605, 0xD607,
        0xD609, 0xD60B, 0xD60D, 0xD60F, 0xD611, 0xD613
    ]
    BALL_QUANTITIES_ADDRS = [
        0xD5FE, 0xD600, 0xD602, 0xD604, 0xD606, 0xD608,
        0xD60A, 0xD60C, 0xD60E, 0xD610, 0xD612, 0xD614
    ]

    # PARTY POKEMON
    N_POKEMON_IN_PARTY_ADDR = 0xDA22
    #   Pokemon 1
    PARTY_POKEMON1_ADDR = 0xDA23  # also 0xDA2A
    PARTY_POKEMON1_NAME = [0xDB8C, 0xDB96]
    PARTY_POKEMON1_ITEM_ADDR = 0xDA2B
    PARTY_POKEMON1_MOVE1_ADDR = 0xDA2C
    PARTY_POKEMON1_MOVE2_ADDR = 0xDA2D
    PARTY_POKEMON1_MOVE3_ADDR = 0xDA2E
    PARTY_POKEMON1_MOVE4_ADDR = 0xDA2F
    PARTY_POKEMON1_ID_ADDRS = [0xDA30, 0xDA31]
    PARTY_POKEMON1_EXP_ADDR = [0xDA32, 0xDA34]
    PARTY_POKEMON1_HP_EV_ADDR = [0xDA35, 0xDA36]
    PARTY_POKEMON1_ATTACK_EV_ADDR = [0xDA37, 0xDA38]
    PARTY_POKEMON1_DEFENSE_EV_ADDR = [0xDA39, 0xDA3A]
    PARTY_POKEMON1_SPEED_EV_ADDR = [0xDA3B, 0xDA3C]
    PARTY_POKEMON1_SPECIAL_EV_ADDR = [0xDA3D, 0xDA3E]
    PARTY_POKEMON1_ATTACK_DEFENSE_IV_ADDR = 0xDA3F
    PARTY_POKEMON1_SPEED_SPECIAL_IV_ADDR = 0xDA40
    PARTY_POKEMON1_PP_SLOT1_ADDR = 0xDA41
    PARTY_POKEMON1_PP_SLOT2_ADDR = 0xDA42
    PARTY_POKEMON1_PP_SLOT3_ADDR = 0xDA43
    PARTY_POKEMON1_PP_SLOT4_ADDR = 0xDA44
    PARTY_POKEMON1_HAPPINESS_TIME_HATCHING_ADDR = 0xDA45
    PARTY_POKEMON1_POKERUS_ADDR = 0xDA46
    PARTY_POKEMON1_CATCH_DATA_ADDRS = [0xDA47, 0xDA48]
    PARTY_POKEMON1_LEVEL_ADDR = 0xDA49
    PARTY_POKEMON1_STATUS_ADDR = [0xDA4A, 0xDA4B]
    PARTY_POKEMON1_HP_ADDR = [0xDA4C, 0xDA4D]
    PARTY_POKEMON1_MAX_HP_ADDR = [0xDA4E, 0xDA4F]
    PARTY_POKEMON1_ATTACK_ADDR = [0xDA50, 0xDA51]
    PARTY_POKEMON1_DEFENSE_ADDR = [0xDA52, 0xDA53]
    PARTY_POKEMON1_SPEED_ADDR = [0xDA54, 0xDA55]
    PARTY_POKEMON1_SPECIAL_DEFENSE_ADDR = [0xDA56, 0xDA57]
    PARTY_POKEMON1_SPECIAL_ATTACK_ADDR = [0xDA58, 0xDA59]
    #   Pokemon 2
    PARTY_POKEMON2_ADDR = 0xDA24  # also 0xDA5A
    PARTY_POKEMON2_NAME = [0xDB97, 0xDBA1]
    PARTY_POKEMON2_ITEM_ADDR = 0xDA5B
    PARTY_POKEMON2_MOVE1_ADDR = 0xDA5C
    PARTY_POKEMON2_MOVE2_ADDR = 0xDA5D
    PARTY_POKEMON2_MOVE3_ADDR = 0xDA5E
    PARTY_POKEMON2_MOVE4_ADDR = 0xDA5F
    PARTY_POKEMON2_ID_ADDRS = [0xDA60, 0xDA61]
    PARTY_POKEMON2_EXP_ADDR = [0xDA62, 0xDA64]
    PARTY_POKEMON2_HP_EV_ADDR = [0xDA65, 0xDA66]
    PARTY_POKEMON2_ATTACK_EV_ADDR = [0xDA67, 0xDA68]
    PARTY_POKEMON2_DEFENSE_EV_ADDR = [0xDA69, 0xDA6A]
    PARTY_POKEMON2_SPEED_EV_ADDR = [0xDA6B, 0xDA6C]
    PARTY_POKEMON2_SPECIAL_EV_ADDR = [0xDA6D, 0xDA6E]
    PARTY_POKEMON2_ATTACK_DEFENSE_IV_ADDR = 0xDA6F
    PARTY_POKEMON2_SPEED_SPECIAL_IV_ADDR = 0xDA70
    PARTY_POKEMON2_PP_SLOT1_ADDR = 0xDA71
    PARTY_POKEMON2_PP_SLOT2_ADDR = 0xDA72
    PARTY_POKEMON2_PP_SLOT3_ADDR = 0xDA73
    PARTY_POKEMON2_PP_SLOT4_ADDR = 0xDA74
    PARTY_POKEMON2_HAPPINESS_TIME_HATCHING_ADDR = 0xDA75
    PARTY_POKEMON2_POKERUS_ADDR = 0xDA76
    PARTY_POKEMON2_CATCH_DATA_ADDRS = [0xDA77, 0xDA78]
    PARTY_POKEMON2_LEVEL_ADDR = 0xDA79
    PARTY_POKEMON2_STATUS_ADDR = [0xDA7A, 0xDA7B]
    PARTY_POKEMON2_HP_ADDR = [0xDA7C, 0xDA7D]
    PARTY_POKEMON2_MAX_HP_ADDR = [0xDA7E, 0xDA7F]
    PARTY_POKEMON2_ATTACK_ADDR = [0xDA80, 0xDA81]
    PARTY_POKEMON2_DEFENSE_ADDR = [0xDA82, 0xDA83]
    PARTY_POKEMON2_SPEED_ADDR = [0xDA84, 0xDA85]
    PARTY_POKEMON2_SPECIAL_DEFENSE_ADDR = [0xDA86, 0xDA87]
    PARTY_POKEMON2_SPECIAL_ATTACK_ADDR = [0xDA88, 0xDA89]
    #   Pokemon 3
    PARTY_POKEMON3_ADDR = 0xDA25  # also 0xDA8A
    PARTY_POKEMON3_NAME = [0xDBA2, 0xDBAC]
    PARTY_POKEMON3_ITEM_ADDR = 0xDA8B
    PARTY_POKEMON3_MOVE1_ADDR = 0xDA8C
    PARTY_POKEMON3_MOVE2_ADDR = 0xDA8D
    PARTY_POKEMON3_MOVE3_ADDR = 0xDA8E
    PARTY_POKEMON3_MOVE4_ADDR = 0xDA8F
    PARTY_POKEMON3_ID_ADDRS = [0xDA90, 0xDA91]
    PARTY_POKEMON3_EXP_ADDR = [0xDA92, 0xDA94]
    PARTY_POKEMON3_HP_EV_ADDR = [0xDA95, 0xDA96]
    PARTY_POKEMON3_ATTACK_EV_ADDR = [0xDA97, 0xDA98]
    PARTY_POKEMON3_DEFENSE_EV_ADDR = [0xDA99, 0xDA9A]
    PARTY_POKEMON3_SPEED_EV_ADDR = [0xDA9B, 0xDA9C]
    PARTY_POKEMON3_SPECIAL_EV_ADDR = [0xDA9D, 0xDA9E]
    PARTY_POKEMON3_ATTACK_DEFENSE_IV_ADDR = 0xDA9F
    PARTY_POKEMON3_SPEED_SPECIAL_IV_ADDR = 0xDAA0
    PARTY_POKEMON3_PP_SLOT1_ADDR = 0xDAA1
    PARTY_POKEMON3_PP_SLOT2_ADDR = 0xDAA2
    PARTY_POKEMON3_PP_SLOT3_ADDR = 0xDAA3
    PARTY_POKEMON3_PP_SLOT4_ADDR = 0xDAA4
    PARTY_POKEMON3_HAPPINESS_TIME_HATCHING_ADDR = 0xDAA5
    PARTY_POKEMON3_POKERUS_ADDR = 0xDAA6
    PARTY_POKEMON3_CATCH_DATA_ADDRS = [0xDAA7, 0xDAA8]
    PARTY_POKEMON3_LEVEL_ADDR = 0xDAA9
    PARTY_POKEMON3_STATUS_ADDR = [0xDAAA, 0xDAAB]
    PARTY_POKEMON3_HP_ADDR = [0xDAAC, 0xDAAD]
    PARTY_POKEMON3_MAX_HP_ADDR = [0xDAAE, 0xDAAF]
    PARTY_POKEMON3_ATTACK_ADDR = [0xDAB0, 0xDAB1]
    PARTY_POKEMON3_DEFENSE_ADDR = [0xDAB2, 0xDAB3]
    PARTY_POKEMON3_SPEED_ADDR = [0xDAB4, 0xDAB5]
    PARTY_POKEMON3_SPECIAL_DEFENSE_ADDR = [0xDAB6, 0xDAB7]
    PARTY_POKEMON3_SPECIAL_ATTACK_ADDR = [0xDAB8, 0xDAB8]
    #   Pokemon 4
    PARTY_POKEMON4_ADDR = 0xDA26  # also 0xDABA
    PARTY_POKEMON4_NAME = [0xDBAD, 0xDBB7]
    PARTY_POKEMON4_ITEM_ADDR = 0xDABB
    PARTY_POKEMON4_MOVE1_ADDR = 0xDABC
    PARTY_POKEMON4_MOVE2_ADDR = 0xDABD
    PARTY_POKEMON4_MOVE3_ADDR = 0xDABE
    PARTY_POKEMON4_MOVE4_ADDR = 0xDABF
    PARTY_POKEMON4_ID_ADDRS = [0xDAC0, 0xDAC1]
    PARTY_POKEMON4_EXP_ADDR = [0xDAC2, 0xDAC4]
    PARTY_POKEMON4_HP_EV_ADDR = [0xDAC5, 0xDAC6]
    PARTY_POKEMON4_ATTACK_EV_ADDR = [0xDAC7, 0xDAC8]
    PARTY_POKEMON4_DEFENSE_EV_ADDR = [0xDAC9, 0xDACA]
    PARTY_POKEMON4_SPEED_EV_ADDR = [0xDACB, 0xDACC]
    PARTY_POKEMON4_SPECIAL_EV_ADDR = [0xDACD, 0xDACE]
    PARTY_POKEMON4_ATTACK_DEFENSE_IV_ADDR = 0xDACF
    PARTY_POKEMON4_SPEED_SPECIAL_IV_ADDR = 0xDAD0
    PARTY_POKEMON4_PP_SLOT1_ADDR = 0xDAD1
    PARTY_POKEMON4_PP_SLOT2_ADDR = 0xDAD2
    PARTY_POKEMON4_PP_SLOT3_ADDR = 0xDAD3
    PARTY_POKEMON4_PP_SLOT4_ADDR = 0xDAD4
    PARTY_POKEMON4_HAPPINESS_TIME_HATCHING_ADDR = 0xDAD5
    PARTY_POKEMON4_POKERUS_ADDR = 0xDAD6
    PARTY_POKEMON4_CATCH_DATA_ADDRS = [0xDAD7, 0xDAD8]
    PARTY_POKEMON4_LEVEL_ADDR = 0xDAD9
    PARTY_POKEMON4_STATUS_ADDR = [0xDADA, 0xDADB]
    PARTY_POKEMON4_HP_ADDR = [0xDADC, 0xDADD]
    PARTY_POKEMON4_MAX_HP_ADDR = [0xDADE, 0xDADF]
    PARTY_POKEMON4_ATTACK_ADDR = [0xDAE0, 0xDAE1]
    PARTY_POKEMON4_DEFENSE_ADDR = [0xDAE2, 0xDAE3]
    PARTY_POKEMON4_SPEED_ADDR = [0xDAE4, 0xDAE5]
    PARTY_POKEMON4_SPECIAL_DEFENSE_ADDR = [0xDAE6, 0xDAE7]
    PARTY_POKEMON4_SPECIAL_ATTACK_ADDR = [0xDAE8, 0xDAE9]
    #   Pokemon 5
    PARTY_POKEMON5_ADDR = 0xDA27  # also 0xDAEA
    PARTY_POKEMON5_NAME = [0xDBB8, 0xDBC2]
    PARTY_POKEMON5_ITEM_ADDR = 0xDAEB
    PARTY_POKEMON5_MOVE1_ADDR = 0xDAEC
    PARTY_POKEMON5_MOVE2_ADDR = 0xDAED
    PARTY_POKEMON5_MOVE3_ADDR = 0xDAEE
    PARTY_POKEMON5_MOVE4_ADDR = 0xDAEF
    PARTY_POKEMON5_ID_ADDRS = [0XDAF0, 0XDAF1]
    PARTY_POKEMON5_EXP_ADDR = [0XDAF2, 0XDAF4]
    PARTY_POKEMON5_HP_EV_ADDR = [0XDAF5, 0XDAF6]
    PARTY_POKEMON5_ATTACK_EV_ADDR = [0XDAF7, 0XDAF8]
    PARTY_POKEMON5_DEFENSE_EV_ADDR = [0XDAF9, 0XDAFA]
    PARTY_POKEMON5_SPEED_EV_ADDR = [0XDAFB, 0XDAFC]
    PARTY_POKEMON5_SPECIAL_EV_ADDR = [0XDAFD, 0XDAFE]
    PARTY_POKEMON5_ATTACK_DEFENSE_IV_ADDR = 0XDAFF
    PARTY_POKEMON5_SPEED_SPECIAL_IV_ADDR = 0xDB00
    PARTY_POKEMON5_PP_SLOT1_ADDR = 0xDB01
    PARTY_POKEMON5_PP_SLOT2_ADDR = 0xDB02
    PARTY_POKEMON5_PP_SLOT3_ADDR = 0xDB03
    PARTY_POKEMON5_PP_SLOT4_ADDR = 0xDB04
    PARTY_POKEMON5_HAPPINESS_TIME_HATCHING_ADDR = 0xDB05
    PARTY_POKEMON5_POKERUS_ADDR = 0xDB06
    PARTY_POKEMON5_CATCH_DATA_ADDRS = [0xDB07, 0xDB08]
    PARTY_POKEMON5_LEVEL_ADDR = 0xDB09
    PARTY_POKEMON5_STATUS_ADDR = [0xDB0A, 0xDB0B]
    PARTY_POKEMON5_HP_ADDR = [0xDB0C, 0xDB0D]
    PARTY_POKEMON5_MAX_HP_ADDR = [0xDB0E, 0xDB0F]
    PARTY_POKEMON5_ATTACK_ADDR = [0xDB10, 0xDB11]
    PARTY_POKEMON5_DEFENSE_ADDR = [0xDB12, 0xDB13]
    PARTY_POKEMON5_SPEED_ADDR = [0xDB14, 0xDB15]
    PARTY_POKEMON5_SPECIAL_DEFENSE_ADDR = [0xDB16, 0xDB17]
    PARTY_POKEMON5_SPECIAL_ATTACK_ADDR = [0xDB18, 0xDB19]
    #   Pokemon 6
    PARTY_POKEMON6_ADDR = 0xDA28  # also 0xDB1A
    PARTY_POKEMON6_NAME = [0xDBC3, 0xDBCD]
    PARTY_POKEMON6_ITEM_ADDR = 0xDB1B
    PARTY_POKEMON6_MOVE1_ADDR = 0xDB1C
    PARTY_POKEMON6_MOVE2_ADDR = 0xDB1D
    PARTY_POKEMON6_MOVE3_ADDR = 0xDB1E
    PARTY_POKEMON6_MOVE4_ADDR = 0xDB1F
    PARTY_POKEMON6_ID_ADDRS = [0xDB20, 0xDB21]
    PARTY_POKEMON6_EXP_ADDR = [0xDB22, 0xDB24]
    PARTY_POKEMON6_HP_EV_ADDR = [0xDB25, 0xDB26]
    PARTY_POKEMON6_ATTACK_EV_ADDR = [0xDB27, 0xDB28]
    PARTY_POKEMON6_DEFENSE_EV_ADDR = [0xDB29, 0xDB2A]
    PARTY_POKEMON6_SPEED_EV_ADDR = [0xDB2B, 0xDB2C]
    PARTY_POKEMON6_SPECIAL_EV_ADDR = [0xDB2D, 0xDB2E]
    PARTY_POKEMON6_ATTACK_DEFENSE_IV_ADDR = 0xDB2F
    PARTY_POKEMON6_SPEED_SPECIAL_IV_ADDR = 0xDB30
    PARTY_POKEMON6_PP_SLOT1_ADDR = 0xDB31
    PARTY_POKEMON6_PP_SLOT2_ADDR = 0xDB32
    PARTY_POKEMON6_PP_SLOT3_ADDR = 0xDB33
    PARTY_POKEMON6_PP_SLOT4_ADDR = 0xDB34
    PARTY_POKEMON6_HAPPINESS_TIME_HATCHING_ADDR = 0xDB35
    PARTY_POKEMON6_POKERUS_ADDR = 0xDB36
    PARTY_POKEMON6_CATCH_DATA_ADDRS = [0xDB37, 0xDB38]
    PARTY_POKEMON6_LEVEL_ADDR = 0xDB39
    PARTY_POKEMON6_STATUS_ADDR = [0xDB3A, 0xDB3B]
    PARTY_POKEMON6_HP_ADDR = [0xDB3C, 0xDB3D]
    PARTY_POKEMON6_MAX_HP_ADDR = [0xDB3E, 0xDB3F]
    PARTY_POKEMON6_ATTACK_ADDR = [0xDB40, 0xDB41]
    PARTY_POKEMON6_DEFENSE_ADDR = [0xDB42, 0xDB43]
    PARTY_POKEMON6_SPEED_ADDR = [0xDB44, 0xDB45]
    PARTY_POKEMON6_SPECIAL_DEFENSE_ADDR = [0xDB46, 0xDB47]
    PARTY_POKEMON6_SPECIAL_ATTACK_ADDR = [0xDB48, 0xDB49]

    # POKEDEX
    POKEDEX_1_256_ADDRS = [0xDBE4, 0xDC03]
    #   seen
    POKEDEX_SEEN_1_256_ADDRS = [0xDC04, 0xDC23]

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

        self.pyboy = PyBoy(
            "roms/Pokemon - Silver Version (UE) [C][!].gbc",
            window_type=self.window_type
        )

        self.save_path = save_path
        self.load_path = load_path
        if load_path is not None:
            self.load()

        print(f'CARTRIDGE: {self.pyboy.cartridge_title()}')
        #assert self.pyboy.cartridge_title() == 'POKEMON_SLVAAX' or self.pyboy.cartridge_title() == 'POKEMON GOLD', 'The cartridge title should be POKEMON SILVER or POKEMON GOLD, international edition not Japan edition (RAM addresses are different in the latter)!'

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
        self.take_action(action)
        self.actions_taken += 1
        done = False
        if self.max_actions == self.actions_taken:
            done = True

        obs = self.render()
        info = self.get_info()

        # SUBTASKS
        if self.subtask == 'enter_first_route':
            if info['player']['what_map_number'] == 4 and 4 not in self.visited:
                reward = 1
                self.visited.append(4)
            elif info['player']['what_map_number'] == 3:
                reward = 10
                done = True
            else:
                reward = 0
        elif self.subtask == 'first_pokemon':
            reward = 1 if info['player']['pokemon_1']['id'] != 512 else 0 # TODO: check
        elif self.subtask == 'get_six_pokemon':
            reward = 0
            for i in range(6):
                reward += 1 if info['player'][f'pokemon_{i}']['id'] != 512 else 0 # TODO: check
            reward = 1 if reward == 6 else 0
        elif self.subtask == 'beat_first_gym':
            reward = 1 if 0 in info['player']['johto_badges'] else 0
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
        self.pyboy = PyBoy(
            "roms/Pokemon - Silver Version (UE) [C][!].gbc",
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
            if self.pyboy.frame_count == 300 or self.pyboy.frame_count == 700:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            if self.pyboy.frame_count == 305 or self.pyboy.frame_count == 705:
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            if self.pyboy.frame_count == 800:
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
                **self.get_item_info(),
                **self.get_pokedex_info(),
                **self.get_battle_info(),
                **self.get_wild_info(),
                **self.get_bug_contest_info()}

    #   ******************************************************
    #                FUNCTION FOR READING RAM
    #   ******************************************************
    def get_player_info(self) -> dict:
        player_info = dict()

        player_info['name'] = self.decode_name(self.PLAYER_NAME_ADDRS[0], self.PLAYER_NAME_ADDRS[1])
        player_info['money'] = self.read_money(self.MONEY_ADDRS)
        player_info['mother_money'] = self.read_money(self.MOTHER_HELD_MONEY_ADDRS)
        player_info['rival_name'] = self.decode_name(self.RIVAL_NAME_ADDRS[0], self.RIVAL_NAME_ADDRS[1])
        player_info['x_pos'] = self.pyboy.get_memory_value(self.X_POS_ADDR)
        player_info['y_pos'] = self.pyboy.get_memory_value(self.Y_POS_ADDR)
        player_info['what_map_bank'] = self.pyboy.get_memory_value(self.WHAT_MAP_BANK_ADDR)
        player_info['what_map_number'] = self.pyboy.get_memory_value(self.WHAT_MAP_NUMBER_ADDR)
        player_info['johto_badges'] = self.decode_badges(self.JOHTO_BADGES_ADDR)
        player_info['kanto_badges'] = self.decode_badges(self.KANTO_BADGES_ADDR)
        player_info['on_bike'] = self.pyboy.get_memory_value(self.ON_BIKE_ADDR) == 0x01
        player_info['repel_time_left'] = self.pyboy.get_memory_value(self.REPEL_STEP_LEFT_ADDR)
        player_info['park_time'] = self.pyboy.get_memory_value(self.PARK_TIME_ADDR)
        player_info['casino_coins'] = (self.pyboy.get_memory_value(self.CASINO_COINS_ADDRS[0]) * (2 ** 8)
                                       + self.pyboy.get_memory_value(self.CASINO_COINS_ADDRS[1]))
        player_info['pokemon_in_party'] = self.pyboy.get_memory_value(self.N_POKEMON_IN_PARTY_ADDR)

        # Pokemon 1 info
        pokemon = dict()
        pokemon['pokemon'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_ADDR)
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON1_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON1_NAME[0], self.PARTY_POKEMON1_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_ITEM_ADDR)
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
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON1_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON1_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON1_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON1_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON1_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON1_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON1_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_1'] = pokemon.copy()
        # Pokemon 2 info
        pokemon = dict()
        pokemon['pokemon'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_ADDR)
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON2_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON2_NAME[0], self.PARTY_POKEMON2_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_ITEM_ADDR)
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
        pokemon['speed_iv'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_SPECIAL_IV_ADDR) >> 4) & 0b1111
        pokemon['special_iv'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_SPECIAL_IV_ADDR) & 0b1111
        pokemon['pp1'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT1_ADDR)
        pokemon['pp2'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT2_ADDR)
        pokemon['pp3'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT3_ADDR)
        pokemon['pp4'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_PP_SLOT4_ADDR)
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON2_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON2_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON2_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON2_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON2_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON2_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON2_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_2'] = pokemon.copy()
        # Pokemon 3 info
        pokemon = dict()
        pokemon['pokemon'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_ADDR)
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON3_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON3_NAME[0], self.PARTY_POKEMON3_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_ITEM_ADDR)
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
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON3_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON3_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON3_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON3_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON3_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON3_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON3_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_3'] = pokemon.copy()
        # Pokemon 4 info
        pokemon = dict()
        pokemon['pokemon'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_ADDR)
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON4_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON4_NAME[0], self.PARTY_POKEMON4_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_ITEM_ADDR)
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
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON4_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON4_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON4_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON4_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON4_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON4_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON4_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_4'] = pokemon.copy()
        # Pokemon 5 info
        pokemon = dict()
        pokemon['pokemon'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_ADDR)
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON5_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON5_NAME[0], self.PARTY_POKEMON5_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_ITEM_ADDR)
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
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON5_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON5_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON5_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON5_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON5_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON5_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON5_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_5'] = pokemon.copy()
        # Pokemon 6 info
        pokemon = dict()
        pokemon['id'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON1_ID_ADDRS[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON6_ID_ADDRS[1]))
        pokemon['name'] = self.decode_name(self.PARTY_POKEMON6_NAME[0], self.PARTY_POKEMON6_NAME[1])
        pokemon['item'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_ITEM_ADDR)
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
        pokemon['happiness_time_hatching'] = self.pyboy.get_memory_value(
            self.PARTY_POKEMON6_HAPPINESS_TIME_HATCHING_ADDR)
        pokemon['pokerus'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_POKERUS_ADDR)
        pokemon['catch_data'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_CATCH_DATA_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.PARTY_POKEMON6_CATCH_DATA_ADDRS[1]))
        pokemon['level'] = self.pyboy.get_memory_value(self.PARTY_POKEMON6_LEVEL_ADDR)
        pokemon['hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_ADDR[0]) * (2 ** 8)
                         + self.pyboy.get_memory_value(self.PARTY_POKEMON6_HP_ADDR[1]))
        pokemon['max_hp'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_MAX_HP_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_MAX_HP_ADDR[1]))
        pokemon['attack'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_ADDR[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.PARTY_POKEMON6_ATTACK_ADDR[1]))
        pokemon['defense'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_ADDR[0]) * (2 ** 8)
                              + self.pyboy.get_memory_value(self.PARTY_POKEMON6_DEFENSE_ADDR[1]))
        pokemon['speed'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_ADDR[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPEED_ADDR[1]))
        pokemon['special_def'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_DEFENSE_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_DEFENSE_ADDR[1]))
        pokemon['special_atk'] = (self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_ATTACK_ADDR[0]) * (2 ** 8)
                                  + self.pyboy.get_memory_value(self.PARTY_POKEMON6_SPECIAL_ATTACK_ADDR[1]))
        player_info['pokemon_6'] = pokemon.copy()
        return {'player': player_info}

    def get_opponent_info(self) -> dict:
        opponent_info = dict()
        opponent_info['pokemon_in_party'] = self.pyboy.get_memory_value(self.N_POKEMON_OPPONENT_ADDR)
        # Pokemon 1 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON1_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON1_NAME_ADDR[0], self.OPPONENT_POKEMON1_NAME_ADDR[0])
        opponent_info['pokemon_1'] = pokemon.copy()
        # Pokemon 2 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON2_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON2_NAME_ADDR[0], self.OPPONENT_POKEMON2_NAME_ADDR[0])
        opponent_info['pokemon_2'] = pokemon.copy()
        # Pokemon 3 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON3_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON3_NAME_ADDR[0], self.OPPONENT_POKEMON3_NAME_ADDR[0])
        opponent_info['pokemon_3'] = pokemon.copy()
        # Pokemon 4 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON4_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON4_NAME_ADDR[0], self.OPPONENT_POKEMON4_NAME_ADDR[0])
        opponent_info['pokemon_4'] = pokemon.copy()
        # Pokemon 5 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON5_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON5_NAME_ADDR[0], self.OPPONENT_POKEMON5_NAME_ADDR[0])
        opponent_info['pokemon_5'] = pokemon.copy()
        # Pokemon 6 info
        pokemon = dict()
        pokemon['id'] = self.pyboy.get_memory_value(self.OPPONENT_POKEMON6_ADDR)
        pokemon['name'] = self.decode_name(self.OPPONENT_POKEMON6_NAME_ADDR[0], self.OPPONENT_POKEMON6_NAME_ADDR[0])
        opponent_info['pokemon_6'] = pokemon.copy()

        return {'opponent': opponent_info}

    def get_item_info(self) -> dict:
        item_info = dict()
        item_info['total_items'] = self.pyboy.get_memory_value(self.TOTAL_ITEM_COUNT_ADDR)
        for i, addr in enumerate(self.ITEM_ADDRS):
            item_info[f'item{i}'] = {'id': self.pyboy.get_memory_value(addr),
                                     'quantity': self.pyboy.get_memory_value(self.ITEM_QUANTITIES_ADDRS[i])}

        item_info['total_balls'] = self.pyboy.get_memory_value(self.TOTAL_BALLS_COUNT_ADDR)
        for i, addr in enumerate(self.BALL_ADDRS):
            item_info[f'ball{i}'] = {'id': self.pyboy.get_memory_value(addr),
                                     'quantity': self.pyboy.get_memory_value(self.BALL_QUANTITIES_ADDRS[i])}

        tm_list = list()
        for i, addr in enumerate(range(self.TM_ADDRS[0], self.TM_ADDRS[1] + 1)):
            if self.pyboy.get_memory_value(addr) == 1:
                tm_list += [i]
        item_info['tms_held'] = tm_list

        hm_list = list()
        for i, addr in enumerate(range(self.HM_ADDRS[0], self.HM_ADDRS[1] + 1)):
            hm_list += [i]
        item_info['hms_held'] = hm_list

        item_info['total_key_items'] = self.pyboy.get_memory_value(self.TOTAL_KEY_ITEMS_COUNT_ADDR)
        key_list = list()
        for i, addr in enumerate(range(self.KEY_ITEMS_ADDRS[0], self.KEY_ITEMS_ADDRS[1] + 1)):
            hm_list += [i]
        item_info['key_items_held'] = key_list

        return {'items': item_info}

    def get_pokedex_info(self) -> dict:
        pokedex_caught = list()
        for i, addr in enumerate(range(self.POKEDEX_1_256_ADDRS[0], self.POKEDEX_1_256_ADDRS[1] + 1)):
            pokedex_caught += [i * 8 + ones for ones in self.list_one_bits_locations(addr)]
        pokedex_seen = list()
        for i, addr in enumerate(range(self.POKEDEX_SEEN_1_256_ADDRS[0], self.POKEDEX_SEEN_1_256_ADDRS[1] + 1)):
            pokedex_seen += [i * 8 + ones for ones in self.list_one_bits_locations(addr)]
        return {'pokedex': {'caught': pokedex_caught, 'seen': pokedex_seen}}

    def get_battle_info(self) -> dict:
        enemy_info = dict()
        enemy_info['future_move1'] = self.pyboy.get_memory_value(self.WHAT_MOVES_WILL_HAVE1_ADDR)
        enemy_info['future_move2'] = self.pyboy.get_memory_value(self.WHAT_MOVES_WILL_HAVE2_ADDR)
        enemy_info['future_move3'] = self.pyboy.get_memory_value(self.WHAT_MOVES_WILL_HAVE3_ADDR)
        enemy_info['future_move4'] = self.pyboy.get_memory_value(self.WHAT_MOVES_WILL_HAVE4_ADDR)
        enemy_info['item'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_ITEM_ADDR)
        enemy_info['move1'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_MOVE1_ADDR)
        enemy_info['move2'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_MOVE2_ADDR)
        enemy_info['move3'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_MOVE3_ADDR)
        enemy_info['move4'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_MOVE4_ADDR)
        enemy_info['attack_dv'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_ATTACK_DEFENSE_DV_ADDR) & 0b1111
        enemy_info['defense_dv'] = (self.pyboy.get_memory_value(
            self.ENEMY_POKEMON_ATTACK_DEFENSE_DV_ADDR) >> 4) & 0b1111
        enemy_info['speed_dv'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPEED_SPECIAL_DV_ADDR) & 0b1111
        enemy_info['special_dv'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPEED_SPECIAL_DV_ADDR) >> 4) & 0b1111
        enemy_info['level'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_LEVEL_ADDR)
        enemy_info['status'] = self.read_status(self.ENEMY_POKEMON_STATUS_ADDR)
        enemy_info['hp'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_CURRENT_HP_ADDRS[0]) * (2 ** 8)
                            + self.pyboy.get_memory_value(self.ENEMY_POKEMON_CURRENT_HP_ADDRS[1]))
        enemy_info['max_hp'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_TOTAL_HP_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.ENEMY_POKEMON_TOTAL_HP_ADDRS[1]))
        enemy_info['attack'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_ATTACK_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.ENEMY_POKEMON_ATTACK_ADDRS[1]))
        enemy_info['defense'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_DEFENSE_ADDRS[0]) * (2 ** 8)
                                 + self.pyboy.get_memory_value(self.ENEMY_POKEMON_DEFENSE_ADDRS[1]))
        enemy_info['speed'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPEED_ADDRS[0]) * (2 ** 8)
                               + self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPEED_ADDRS[1]))
        enemy_info['special_atk'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPECIAL_ATTACK_ADDRS[0]) * (2 ** 8)
                                     + self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPECIAL_ATTACK_ADDRS[1]))
        enemy_info['special_atk'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPECIAL_DEFENSE_ADDRS[0]) * (2 ** 8)
                                     + self.pyboy.get_memory_value(self.ENEMY_POKEMON_SPECIAL_DEFENSE_ADDRS[1]))
        enemy_info['sex'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_SEX_ADDR)
        enemy_info['type1'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_TYPE1_ADDR)
        enemy_info['type2'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_TYPE2_ADDR)
        enemy_info['damage'] = (self.pyboy.get_memory_value(self.ENEMY_POKEMON_DAMAGE_ADDRS[0]) * (2 ** 8)
                                + self.pyboy.get_memory_value(self.ENEMY_POKEMON_DAMAGE_ADDRS[1]))
        enemy_info['magnitude'] = self.pyboy.get_memory_value(self.ENEMY_POKEMON_MAGNITUDE_ADDR)

        player_info = dict()
        player_info['item'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_ITEM_ADDR)
        player_info['move1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE1_ADDR)
        player_info['move2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE2_ADDR)
        player_info['move3'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE3_ADDR)
        player_info['move4'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_MOVE4_ADDR)
        player_info['pp1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP1_ADDR)
        player_info['pp2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP2_ADDR)
        player_info['pp3'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP3_ADDR)
        player_info['pp4'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_PP3_ADDR)
        player_info['status'] = self.read_status(self.IN_BATTLE_POKEMON_STATUS_ADDR)
        player_info['hp'] = (self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_HP_ADDRS[0]) * (2 ** 8)
                             + self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_HP_ADDRS[1]))
        player_info['type1'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_TYPE1_ADDR)
        player_info['type2'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_TYPE2_ADDR)
        player_info['substitute'] = self.pyboy.get_memory_value(self.IN_BATTLE_POKEMON_SUBSTITUTE_ADDR)
        player_info['money_earned'] = (self.pyboy.get_memory_value(self.IN_BATTLE_MONEY_EARNED_ADDRS[0]) * (2 ** 8)
                                       + self.pyboy.get_memory_value(self.IN_BATTLE_MONEY_EARNED_ADDRS[1]))
        player_info['exp_given'] = (self.pyboy.get_memory_value(self.IN_BATTLE_EXP_GIVEN_ADDRS[0]) * (2 ** 8)
                                    + self.pyboy.get_memory_value(self.IN_BATTLE_EXP_GIVEN_ADDRS[1]))
        player_info['current_atk'] = self.pyboy.get_memory_value(self.IN_BATTLE_CURRENT_ATTACK_ADDR)

        return {'battle':
            {
                'type': self.pyboy.get_memory_value(self.BATTLE_TYPE_ADDR) == 0,
                'enemy': enemy_info,
                'player': player_info
            }
        }

    def get_wild_info(self) -> dict:
        return {'wild': {'number': self.pyboy.get_memory_value(self.WILD_POKEMON_NUMBER_ADDR),
                         'level': self.pyboy.get_memory_value(self.WILD_POKEMON_LEVEL_ADDR)}}

    def get_bug_contest_info(self) -> dict:
        return {'bug_contest': {'id': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_ID_ADDR),
                                'level': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_LEVEL_ADDR),
                                'hp': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_CURRENT_HP_ADDRS[0]) * (2 ** 8)
                                      + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_CURRENT_HP_ADDRS[1]),
                                'max_hp': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_TOTAL_HP_ADDRS[0]) * (2 ** 8)
                                          + self.pyboy.get_memory_value( self.BUG_CONTEST_POKEMON_TOTAL_HP_ADDRS[1]),
                                'atk': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_ATTACK_ADDRS[0]) * (2 ** 8)
                                       + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_ATTACK_ADDRS[1]),
                                'def': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_DEFENSE_ADDRS[0]) * (2 ** 8)
                                       + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_DEFENSE_ADDRS[1]),
                                'speed': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPEED_ADDRS[0]) * (2 ** 8)
                                         + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPEED_ADDRS[1]),
                                'special_atk': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPECIAL_ATTACK_ADDRS[0]) * (2 ** 8)
                                               + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPECIAL_ATTACK_ADDRS[1]),
                                'special_def': self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPECIAL_DEFENSE_ADDRS[0]) * (2 ** 8)
                                               + self.pyboy.get_memory_value(self.BUG_CONTEST_POKEMON_SPECIAL_DEFENSE_ADDRS[1])
                                }}

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

    def read_money(self, money_addr: list[int]) -> int:
        return (100 * 100 * self.read_bcd(self.pyboy.get_memory_value(money_addr[0])) +
                100 * self.read_bcd(self.pyboy.get_memory_value(money_addr[1])) +
                self.read_bcd(self.pyboy.get_memory_value(money_addr[2])))

    def decode_name(self, starting_addr: int, final_addr: int) -> str:
        name = ''
        for addr in range(starting_addr, final_addr + 1):
            char = self.TEXT_TABLE[self.pyboy.get_memory_value(addr).to_bytes(1, byteorder='big')]
            if char == 'END_MARKER':
                break
            name += char
        return name

    def decode_badges(self, addr: int) -> list:
        return self.list_one_bits_locations(addr)

    def list_one_bits_locations(self, addr: int) -> list:
        value = self.pyboy.get_memory_value(addr)
        ones = list()
        for i in range(8):
            if value & (1 << i):
                ones.append(i)
        return ones

    def read_bcd(self, value):
        return 10 * ((value >> 4) & 0x0f) + (value & 0x0f)
