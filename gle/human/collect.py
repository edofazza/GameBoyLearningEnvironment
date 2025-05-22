import pygame
from PIL import Image
from datetime import datetime
from gymnasium import Env

from gle.human.data import GamePlay, Record


def save_state(env, gameplay_path: str, gameplay_name: str, original_frame_count: int):
    env.save_path = f'{gameplay_path}/{gameplay_name}/{datetime.now().strftime("%Y-%m-%d_%H-%M:%S_") + str(env.pyboy.frame_count + original_frame_count)}.state'
    env.save()


def render_obs_to_screen(screen, obs, screen_width, screen_height, scale):
    obs = obs[:, :, [2, 1, 0]]
    observation_image = Image.fromarray(obs, "RGB")

    # Convert image to Pygame surface
    observation_surface = pygame.image.fromstring(observation_image.tobytes(), observation_image.size,
                                                  observation_image.mode)

    observation_surface = pygame.transform.scale(observation_surface,
                                                 (screen_width * scale, screen_height * scale))
    screen.blit(observation_surface, (0, 0))

    # Update the display
    pygame.display.update()


class CollectHumanExperience(object):
    def __init__(self, env: Env, gameplay_path: str, gameplay_name: str, frame_count: int = 0) -> None:
        self.env = env
        self.gameplay_path = gameplay_path
        self.gameplay_name = gameplay_name
        self.frame_count = frame_count
        self.gameplay = GamePlay(self.gameplay_path, self.gameplay_name)

    def play_game(self):
        # (144, 160, 3) needed for human initial reshape
        # (3, 144, 160) for training and original env
        # (160, 144, 3) for pygame
        pygame.init()
        obs, info = self.env.reset()
        obs = obs.reshape((obs.shape[1], obs.shape[2], obs.shape[0]))
        screen_width = obs.shape[1]
        screen_height = obs.shape[0]
        scale = 3

        screen = pygame.display.set_mode((screen_width * scale, screen_height * scale))
        pygame.display.set_caption(self.gameplay_name)
        prev_obs = obs.reshape((obs.shape[2], obs.shape[0], obs.shape[1])).copy()
        render_obs_to_screen(screen, obs, screen_width, screen_height, scale)
        running = True
        done = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 0
                    elif event.key == pygame.K_s:
                        action = 1
                    elif event.key == pygame.K_UP:
                        action = 2
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    elif event.key == pygame.K_RIGHT:
                        action = 4
                    elif event.key == pygame.K_LEFT:
                        action = 5
                    elif event.key == pygame.K_ESCAPE:
                        action = 6
                    elif event.key == pygame.K_RETURN:
                        action = 7
                    elif event.key == pygame.K_SPACE:
                        save_state(self.env, self.gameplay_path, self.gameplay_name, self.frame_count)
                        continue
                    else:
                        continue

                    obs, reward, done, trunc, info = self.env.step(action)
                    obs = obs.reshape((obs.shape[1], obs.shape[2], obs.shape[0]))
                    # GET RECORD FOR SAVING
                    self.gameplay.update(Record(prev_obs, action, info), datetime.now().isoformat()) # TODO: add None
                    # Save new observation as prev_obs
                    prev_obs = obs.reshape((obs.shape[2], obs.shape[0], obs.shape[1])).copy()
                    # render on screen
                    render_obs_to_screen(screen, obs, screen_width, screen_height, scale)

            if done or not running:
                break

