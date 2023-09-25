import os
import shutil
from PIL import Image
from collect_icons import collect_icons
from draw_my_plot import draw_my_plot


if __name__ == '__main__':

    # 1080p_low_repeat.png 20px
    # 1080p_high_repeat.png 10px
    # 4k_low_repeat.png 40px
    # 4k_high_repeat.png 30px

    ROOT_URL = "https://wiki.melvoridle.com"
    IMG_URL = ROOT_URL + "/images/"
    PIXEL_SIZE = 30

    bg_image_path = "background/bg_4k.png"
    image = Image.open(bg_image_path)
    width, height = image.size
    expected_num = height * width / (PIXEL_SIZE * PIXEL_SIZE)

    opt_path = '4k_high_repeat.png'

    # Collect icons
    collect_icons(IMG_URL, int(expected_num))

    # Draw plot
    draw_my_plot(bg_image_path, opt_path, PIXEL_SIZE, recovery=False, distance_method='weight_rgb')
