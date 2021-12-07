import concurrent
import os
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup


URL = 'https://www.zxf33.com/zptp/70661.html'
    
  
HEADERS = {
    'Host': 'www.zxf33.com',
    'Referer': 'https://www.zxf33.com',
    'Pragma': 'no-cache',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
}

def get_html(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=30)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return res.text
    except Exception as e:
        return None
    
def get_image_urls():
    html = get_html(URL)
    soup = BeautifulSoup(html, 'lxml')
    title = soup.find('title').string
    tag_list = soup.find(class_='ttnr').find_all('p')[1].find_all('img')
    url_list = [item.get('src') for item in tag_list]
    return title, url_list

def download_image(title, url_list):
    os.mkdir(title)
    for index, url in enumerate(url_list):
        with open('{}/{}.jpg'.format(title, index), 'wb') as f:
            img = requests.get(url, headers=HEADERS).content
            f.write(img)


if __name__ == '__main__':
    title, url_list = get_image_urls()
    print(url_list)
    download_image(title, url_list)

#     html = get_text('https://www.tooopen.com/img/88_878.aspx')
#     soup = BeautifulSoup(html, 'lxml')
#     result = soup.find_all('img', src=re.compile('.*\.jpg'))
#     result = [img.get('src') for img in result]

#     for i, url in enumerate(result):
#         print(url)
#         with open('D:/images/' + str(i + 1) + '.jpg', 'wb') as f:
#             f.write(requests.get(url).content)
