import moviepy.editor as mp
import imageio
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent


video_path = "ACT_SIM_SUCCESS.mp4"
output_gif_path = "ACT_SIM_SUCCESS.gif"

clip = mp.VideoFileClip(video_path)
clip.write_gif(output_gif_path)