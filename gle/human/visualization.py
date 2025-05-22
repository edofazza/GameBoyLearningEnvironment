import cv2
from typing import List
from gle.human.data import Record


def create_video(trajectory: List[Record], output_path: str, fps: int = 60):
    frame_height, frame_width = trajectory[0].observation.shape[1:]
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f'{output_path}.mp4', fourcc, fps, (frame_width, frame_height))
    for record in trajectory:
        if record.observation.shape[:2] != (frame_height, frame_width):
            record.observation = record.observation.reshape((record.observation.shape[1],
                                                             record.observation.shape[2],
                                                             record.observation.shape[0]))
        video_writer.write(cv2.cvtColor(record.observation, cv2.COLOR_RGB2BGR))
    video_writer.release()
    cv2.destroyAllWindows()
