import requests
import re
import urllib.request
from bs4 import BeautifulSoup


def get_text(link):
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/71.0.3578.98 Safari/537.36'
    }
    r = requests.get(link, headers=headers, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text


if __name__ == '__main__':
    URL = 'http://www.pearvideo.com/category_8'
    HTML = get_text(URL)
    soup = BeautifulSoup(HTML, 'lxml')
    result = soup.find_all('a', class_='vervideo-lilink actplay')
    video_urls = []
    
    for i in result:
        video_id = i.get('href')
        video_urls.append('http://www.pearvideo.com/' + video_id)

    for j in video_urls:
        html = get_text(j)
        soup = BeautifulSoup(html, 'html.parser')
        name_tag = soup.find('h1', class_='video-tt')
        video_name = name_tag.string
        print(video_name)
        play_url = re.search(r'ldUrl="",srcUrl="(.*?)"', html)
        print(play_url.group(1))
        urllib.request.urlretrieve(play_url.group(1), '../data/'+video_name+'.mp4')
