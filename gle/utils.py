from gymnasium.wrappers import NormalizeObservation

from gle.envs.finalfantasyadventure import FinalFantasyAdventure
from gle.envs.solomonsclub import SolomonsClub
from gle.envs.donkeykongland3 import DonkeyKongLand3
from gle.envs.supermarioland import SuperMarioLand
from gle.envs.castlevaniaadventure import CastlevaniaTheAdventure
from gle.envs.castlevania2 import CastlevaniaIIBelmontsRevenge
from gle.envs.kirbysdreamland import KirbysDreamLand
from gle.envs.zeldalinksawakening import ZeldaLinksAwakening
from gle.envs.pokemonbluered import PokemonBlueRed
from gle.envs.donkeykongland2 import DonkeyKongLand2
from gle.envs.pokemongoldsilver import PokemonGoldSilver
from gle.envs.megamanxtreme import MegaManXtreme
from gle.envs.megamandrwilysrevenge import MegaManDrWilysRevenge


def create_env(name: str = 'pokemonred', window_type: str = 'headless', seed: int = 42,
               normalize_observation: bool = False, *args, **kwargs):
    if name in ['pokemonred', 'pokemonblue', 'pokemonbluered', 'pokemonredblue']:
        env = PokemonBlueRed(window_type=window_type, *args, **kwargs)
    elif name == 'finalfantasyadventure':
        env = FinalFantasyAdventure(window_type=window_type, *args, **kwargs)
    elif name == 'solomonsclub':
        env = SolomonsClub(window_type=window_type, *args, **kwargs)
    elif name == 'donkeykongland3':
        env = DonkeyKongLand3(window_type=window_type, *args, **kwargs)
    elif name == 'donkeykongland2':
        env = DonkeyKongLand2(window_type=window_type, *args, **kwargs)
    elif name in ['castlevaniaiibelmontsrevenge', 'castlevaniaii', 'castlevania2', 'castlevania2belmont']:
        env = CastlevaniaIIBelmontsRevenge(window_type=window_type, *args, **kwargs)
    elif name == 'castlevaniaadventure':
        env = CastlevaniaTheAdventure(window_type=window_type, *args, **kwargs)
    elif name == 'kirbysdreamland':
        env = KirbysDreamLand(window_type=window_type, *args, **kwargs)
    elif name == 'supermarioland':
        env = SuperMarioLand(window_type=window_type, *args, **kwargs)
    elif name == 'zeldalinksawakening':
        env = ZeldaLinksAwakening(window_type=window_type, *args, **kwargs)
    elif name == 'pokemongoldsilver':
        env = PokemonGoldSilver(window_type=window_type, *args, **kwargs)
    elif name == 'megamandrwilysrevenge':
        env = MegaManDrWilysRevenge(window_type=window_type, *args, **kwargs)
    elif name == 'megamanxtreme':
        env = MegaManXtreme(window_type=window_type, *args, **kwargs)
    else:
        print('Environment not available please select one from the following list (you can use lowercase):\n'
              '\t- PokemonRed\n'
              '\t- FinalFantasyAdventure\n'
              '\t- SolomonsClub\n'
              '\t- DonkeyKongLand3\n'
              '\t- DonkeyKongLand2\n'
              '\t- Castlevania2\n'
              '\t- CastlevaniaAdventure\n'
              '\t- KirbysDreamLand\n'
              '\t- SuperMarioLand\n'
              '\t- ZeldaLinksAwakening\n'
              '\t- PokemonSilver\n'
              '\t- MegaManDrWilysRevenge\n'
              '\t- MegaManXtreme\n')
        return None

    if normalize_observation:
        env = NormalizeObservation(env)
    env.reset(seed=seed)
    return env
