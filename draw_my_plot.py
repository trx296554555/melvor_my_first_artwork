"""
pip install numpy
pip install pillow
pip install scikit-learn
pip install tqdm
pip install colormath
"""
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
# https://blog.csdn.net/u011888840/article/details/112982156#%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0
from scipy.optimize import linear_sum_assignment


def split_image(image_path, output_directory, pixel_size=40):

    image = Image.open(image_path)
    width, height = image.size

    small_width = pixel_size
    small_height = pixel_size
    # Count the number of small graphs in landscape and portrait orientation
    width_times = width / small_width
    height_times = height / small_height

    for i in range(int(height_times)):
        for j in range(int(width_times)):
            # Calculate the position
            left = j * small_width
            top = i * small_height
            right = (j + 1) * small_width
            bottom = (i + 1) * small_height
            cropped = image.crop((left, top, right, bottom))
            cropped.save(os.path.join(output_directory, f"{i}_{j}.png"))


def resize_image(img_path, out_dir, new_size):
    image = Image.open(img_path)
    resized_image = image.resize(new_size)
    resized_image.save(os.path.join(out_dir, os.path.basename(img_path)))


def image_color_clustering(image_path):
    image_raw = Image.open(image_path).convert("RGBA")
    image_raw_array = np.array(image_raw)

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Remove pixels that are completely transparent
    try:
        alpha = image_raw_array[:, :, 3]
        non_transparent_indices = alpha > 0
        image_array = image_array[non_transparent_indices]
    except IndexError:
        print(image_path)
        return {image_path: 'error'}

    flattened_array = image_array[:, :3].reshape(-1, 3)
    remaining_pixels_num, _ = image_array.shape

    if 'split_images' in image_path:
        nc = 2
    else:
        nc = 3

    # Perform K-means clustering
    try:
        kmeans = KMeans(n_clusters=nc, random_state=42, n_init='auto')
        kmeans.fit(flattened_array)
    except:
        print(f'{image_path} pixels: {remaining_pixels_num}')
        return {1.0: [0, 0, 0]}

    # Get the clustering results and the number of pixels per cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Calculate the proportion of each cluster
    proportions = np.round(counts / remaining_pixels_num, 6)

    # Get the RGB value for the cluster centers
    cluster_centers = kmeans.cluster_centers_.astype(int)

    result = {}
    for proportion, center in zip(proportions, cluster_centers):
        result[proportion] = center.tolist()
    result = dict(sorted(result.items(), key=lambda x: x[0], reverse=True))

    return result


def avg_delta_e_2000(main_col_dict_1, main_col_dict_2):
    """
    Calculate the average color difference between two images
    A more accurate method, but too computationally intensive, not for the time being

    # The latest version of numpy does not support numpy.asscalar()
    # so modify the return value of the delta_e_cie2000 here to directly return the calculated value
    :param main_col_dict_1:
    :param main_col_dict_2:
    :return:
    """
    distance_list = []
    for weight_1 in main_col_dict_1:
        for weight_2 in main_col_dict_2:
            col_1_rgb = sRGBColor(*main_col_dict_1[weight_1])
            col_1_lab = convert_color(col_1_rgb, LabColor)
            col_2_rgb = sRGBColor(*main_col_dict_2[weight_2])
            col_2_lab = convert_color(col_2_rgb, LabColor)
            distance = delta_e_cie2000(col_1_lab, col_2_lab) * weight_1 * weight_2
            distance_list.append(distance)
    return sum(distance_list)


def weighted_rgb_euclidean_distance(main_col_dict_1, main_col_dict_2):
    """
    Calculate the average color difference between two images
    A weighted RGB Euclidean distance that is faster but not accurate enough
    """
    def ColourDistance(rgb_1, rgb_2):
        R_1, G_1, B_1 = rgb_1
        R_2, G_2, B_2 = rgb_2
        rmean = (R_1 + R_2) / 2
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2
        return (2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2)
    distance_list = []
    for weight_1 in main_col_dict_1:
        for weight_2 in main_col_dict_2:
            distance = ColourDistance(main_col_dict_1[weight_1], main_col_dict_2[weight_2]) * weight_1 * weight_2
            distance_list.append(distance)
    return sum(distance_list)


def draw_my_plot(bg_image_path, output_path, pixel_size=40, recovery=False, distance_method=['avg_delta_e_2000', 'weight_rgb'][1]):

    split_dir = "tmp/split_images"
    resize_dir = "tmp/resize_png"

    # Clean up the previous results
    if not recovery:
        shutil.rmtree('tmp', ignore_errors=True)
        os.mkdir('tmp')
        # Split image
        print("1. Split image")
        os.makedirs(split_dir, exist_ok=False)
        split_image(bg_image_path, split_dir, pixel_size)

        # Resize image
        print("2. Resize image")
        os.makedirs(resize_dir, exist_ok=False)
        for img in tqdm(os.listdir('png'), desc='Resize image'):
            resize_image(os.path.join('png', img), resize_dir, (pixel_size, pixel_size))

    # Calculate main color
    print("3. Calculate main color")
    if not os.path.exists('tmp/main_color_dict.pickle'):
        with open('tmp/main_color_dict.pickle', 'wb') as file:
            loaded_png_dict = {}
            for img in os.listdir(resize_dir):
                data = image_color_clustering(os.path.join(resize_dir, img))
                loaded_png_dict[img] = data
            pickle.dump(loaded_png_dict, file)
            loaded_split_dict = {}
            for img in os.listdir(split_dir):
                data = image_color_clustering(os.path.join(split_dir, img))
                loaded_split_dict[img] = data
            pickle.dump(loaded_split_dict, file)
    else:
        with open('tmp/main_color_dict.pickle', 'rb') as file:
            loaded_png_dict = pickle.load(file)
            loaded_split_dict = pickle.load(file)

    # Calculate main color distance
    print("4. Calculate main color distance")
    if distance_method == 'avg_delta_e_2000':
        distance_method = avg_delta_e_2000
    elif distance_method == 'weight_rgb':
        distance_method = weighted_rgb_euclidean_distance
    else:
        raise Exception('Unknown distance method')
    if not os.path.exists('tmp/distance_matrix.pickle'):
        with open('tmp/distance_matrix.pickle', 'wb') as file:
            distance_list = []
            for split_img in tqdm(loaded_split_dict, desc='Calculate main color distance'):
                for png_img in tqdm(loaded_png_dict, leave=False, disable=True):
                    dis = distance_method(loaded_split_dict[split_img], loaded_png_dict[png_img])
                    distance_list.append(dis)

            distance_matrix = np.array(distance_list).reshape(len(loaded_split_dict), len(loaded_png_dict))
            pickle.dump(distance_matrix, file)
    else:
        with open('tmp/distance_matrix.pickle', 'rb') as file:
            distance_matrix = pickle.load(file)

    # Calculate the minimum combination
    print("5. Calculate the minimum combination")
    if not os.path.exists('tmp/optimize_paired.pickle'):
        optimize_row, optimize_col = linear_sum_assignment(distance_matrix)
        print("Row Index:", optimize_row, "Col Index:", optimize_col, "Minimal combination:",
              distance_matrix[optimize_row, optimize_col])
        with open('tmp/optimize_paired.pickle', 'wb') as file:
            pickle.dump(optimize_row, file)
            pickle.dump(optimize_col, file)
    else:
        with open('tmp/optimize_paired.pickle', 'rb') as file:
            optimize_row = pickle.load(file)
            optimize_col = pickle.load(file)

    # Generate a large image
    print("6. Generate a large image")
    output_image = Image.new('RGBA', Image.open(bg_image_path).size)
    split_list = list(loaded_split_dict.keys())
    replace_list = list(loaded_png_dict.keys())
    for i in optimize_row:
        split_img_name = split_list[i]
        position = split_img_name.split('.')[0].split('_')
        replace_img_name = replace_list[optimize_col[i]]
        replace_img = Image.open(os.path.join(resize_dir, replace_img_name))
        output_image.paste(replace_img, (int(position[1])*pixel_size, int(position[0])*pixel_size))
    output_image.save(output_path)




