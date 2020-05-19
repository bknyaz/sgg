import urllib, json
from http import client as httplib
from urllib import parse as urlparse
import urllib.parse
import os
import subprocess
import sys
import tarfile

VG_link = 'https://yadi.sk/d/unHhlZ0YOjCMQQ'
GQA_link = 'https://yadi.sk/d/FGOzRP649rZ2kQ'

def download(url, data_dir):
    api = httplib.HTTPSConnection('cloud-api.yandex.net')
    url ='/v1/disk/public/resources/download?public_key=%s' % urllib.parse.quote(url)
    api.request('GET', url)
    resp = api.getresponse()
    file_info = json.loads(resp.read())
    api.close()
    filename = os.path.join(data_dir, urlparse.parse_qs(urlparse.urlparse(file_info['href']).query)['filename'][0])
    if not os.path.isfile(filename):
        print('downloading %s ...' % filename)
        cmd = "wget -O \"/%s\" \"%s\"" % (filename, file_info['href'])
        return_code = subprocess.call(cmd, shell=True)
        print('return code for %s = %d' % (filename, return_code))

    print('extracting %s to %s' % (filename, data_dir))
    tar = tarfile.open(filename)
    tar.extractall(path=data_dir)
    tar.close()

    return filename

def download_all_data(root_dir):
    if not os.path.exists(root_dir):
        if len(root_dir) == 0:
            raise ValueError("root_dir must be a valid path")
        os.mkdir(root_dir)

    for name, link in zip(['GQA', 'VG'], [GQA_link, VG_link]):
        data_dir = os.path.join(root_dir, name)
        print(name, data_dir)
        if not os.path.exists(data_dir):
            if len(data_dir) == 0:
                raise ValueError("data_dir must be a valid path")
            os.mkdir(data_dir)
        print('Downloading %s (can take a few hours' % name)
        download(link, data_dir)

if __name__ == '__main__':
    download_all_data(sys.argv[1])



