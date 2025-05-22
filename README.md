# GameBoy Learning Environment (GLE)

For installing the package you can:

```bash
pip install GameBoyLearningEnvironment
```

We provide also a set of ROM states, available in the `states` folder. For a description of them look at the paper Supplementary Material.

Here a simple example of how to use our library: 

```python
from gle import create_env, PokemonBlueRed

# calling the create_env function
env = create_env('environment_name')
# or by invocaking the environment class
env = PokemonBlueRed(load_path='states/pokemon_red/pokemon_red_squirtle_after_rival_battle.state', window_type=window_type, max_actions=1000)
```

To cite the article please use:

```latex
In progress...
```