import moviepy.editor as mp
import imageio
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent


video_path = "IBC_SIM_FAIL.mp4"
output_gif_path = "IBC_SIM_FAIL.gif"

clip = mp.VideoFileClip(video_path)
clip.write_gif(output_gif_path)