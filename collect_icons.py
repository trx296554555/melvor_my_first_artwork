"""
python.exe -m pip install --upgrade pip
pip install httpx
pip install beautifulsoup4
pip install cairosvg
"""
import os
import shutil
import random
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import httpx
import asyncio
from bs4 import BeautifulSoup
import cairosvg


def get_category_urls(main_url):
    """
    No Use
    """
    if os.path.exists("urls.dict") and os.path.getsize("urls.dict") > 0:
        print('Find urls.dict, start to read urls.dict')
        with open("urls.dict", "r") as f:
            res_dict = {item.split('\t')[0]: item.split('\t')[1].strip() for item in f.readlines()}
            return res_dict
    else:
        print('Not find urls.dict, start to collect urls.dict')
        res_dict = {}
        response = httpx.get(main_url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            links = table.find_all('a')
            for link in links:
                res_dict[link.get('title')] = link.get('href')
        res_dict.pop(None)
        with open("urls.dict", "w") as f:
            f.writelines(i + '\t' + j + '\n' for i, j in res_dict.items())
        return res_dict


def get_img_urls_from_page(page_name, url):
    """
    No Use
    """
    response = httpx.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.find_all('img')
    image_urls = []
    for img_tag in img_tags:
        image_urls.append(img_tag['src'])
    with open(page_name + "_img_urls.txt", "w") as f:
        f.writelines(i + '\n' for i in image_urls)


def get_img_urls_from_images(url, res_list):
    print(f'----------Start to get img urls from {url}-----------')
    response = httpx.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    links = soup.find_all('a')
    for link in links:
        link_path = link.get('href')
        if link_path.endswith('.png') or link_path.endswith('.jpg') or link_path.endswith('.svg'):
            print(url + link_path)
            res_list.append(url + link_path)
        elif link_path.startswith('/'):
            pass
        elif link_path in ['/', 'archive/', 'lockdir/', 'thumb/']:
            pass
        elif link_path.endswith('/'):
            print(link_path, 'is a folder')
            get_img_urls_from_images(url + link_path, res_list)
        else:
            pass
    return res_list


async def download_file(sem, url):
    # print(f"Task {sem} started")
    async with sem:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            if response.status_code == 200:
                file_extension = url.rsplit('.', 1)[-1]
                if '(' not in url:
                    folder = 'other'
                else:
                    folder = 'svg' if file_extension == 'svg' else 'raw_png'
                file_name = url.rsplit('/', 1)[-1]
                file_path = os.path.join(folder, file_name)

                with open(file_path, 'wb') as file:
                    file.write(response.content)
                # print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download: {url}")


async def main_download(url_list):
    tasks = []
    semaphore = asyncio.Semaphore(16)
    for url in url_list:
        tasks.append(download_file(semaphore, url))

    await tqdm_asyncio.gather(*tasks, desc='Download Images')


def download_images(urls_list):
    urls_dict = {link.split('/')[-1]: link for link in urls_list}
    already_downloaded = []
    if os.path.exists('svg'):
        already_downloaded += os.listdir('svg')
    else:
        os.mkdir('svg')
    if os.path.exists('raw_png'):
        already_downloaded += os.listdir('raw_png')
    else:
        os.mkdir('raw_png')
    if os.path.exists('other'):
        already_downloaded += os.listdir('other')
    else:
        os.mkdir('other')
    unfinished_list = [urls_dict[url] for url in urls_dict if url not in already_downloaded]
    unfinished_list.remove('https://wiki.melvoridle.com/images/d/dd/Interface_Settings_(thin).png')
    print(f'Already downloaded {len(already_downloaded)} images, {len(unfinished_list)} need download.')

    # Use httpx to download images asynchronously
    asyncio.run(main_download(unfinished_list))


def fill_image_num(expected_num):

    # Clean up the previous results
    shutil.rmtree('png', ignore_errors=True)
    os.mkdir('png')

    # Convert svg to png
    # https://www.modb.pro/db/175191
    for svg in tqdm(os.listdir('svg'), desc='Convert SVG to PNG'):
        svg_path = os.path.join('svg', svg)
        png_path = os.path.join('png', svg + '.png')
        cairosvg.svg2png(url=svg_path, write_to=png_path)

    # Copy raw_png to png
    for raw_png in os.listdir('raw_png'):
        raw_png_path = os.path.join('raw_png', raw_png)
        png_path = os.path.join('png', raw_png)
        shutil.copyfile(raw_png_path, png_path)

    if (now_num := len(os.listdir('png'))) < expected_num:
        missing_num = expected_num - now_num
        # Randomly extract existing files from the PNG folder and copy them to achieve the expected number
        for i in range(int(missing_num + 9)):
            random_file = random.choice(os.listdir('png'))
            random_file_path = os.path.join('png', random_file)
            shutil.copyfile(random_file_path, os.path.join('png', str(i) + '.png'))

    # Add Author signature
    shutil.copyfile('background/Robin.png', os.path.join('png', 'author.png'))


def collect_icons(image_url, expected_num):
    # Get img urls
    if os.path.exists("img_urls.txt") and os.path.getsize("img_urls.txt") > 0:
        print('Find img_urls.txt, start to read img_urls.txt')
        with open("img_urls.txt", "r") as f:
            img_urls_list = [item.strip() for item in f.readlines()]
    else:
        print('Not find img_urls.txt, start to collect img_urls.txt')
        img_urls_list = get_img_urls_from_images(image_url, [])
        with open("img_urls.txt", "w") as f:
            f.writelines(i + '\n' for i in img_urls_list)

    # Download images
    download_images(img_urls_list)

    # Fill image num
    fill_image_num(expected_num)
