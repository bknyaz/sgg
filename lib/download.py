import urllib, json
from http import client as httplib
from urllib import parse as urlparse
import urllib.parse
import os
import subprocess
import sys
import tarfile


VG_link = ('https://yadi.sk/d/unHhlZ0YOjCMQQ', 'VG.tar')
GQA_link = ('https://yadi.sk/d/FGOzRP649rZ2kQ', 'GQA_scenegraphs.tar')


def download(url_name_pair, data_dir):
    url, name = url_name_pair
    filename = os.path.join(data_dir, name)  # urlparse.parse_qs(urlparse.urlparse(file_info['href']).query)['filename'][0]
    if not os.path.isfile(filename):
        api = httplib.HTTPSConnection('cloud-api.yandex.net')
        url = '/v1/disk/public/resources/download?public_key=%s' % urllib.parse.quote(url)
        api.request('GET', url)
        resp = api.getresponse()
        file_info = json.loads(resp.read())
        api.close()

        print('\nDownloading %s  (can take a few hours) ...' % filename)

        if 'href' not in file_info:
            raise ValueError(file_info['error'], 'Try running the script later or go to %s in your browser '
                                                 'and download the archive manually and put it in the data_path folder '
                                                 'according to the folder structure in README.md ' % url)

        cmd = "wget -O \"%s\" \"%s\"" % (filename, file_info['href'])
        return_code = subprocess.call(cmd, shell=True)
        print('return code for %s = %d' % (filename, return_code))

    print('extracting %s to %s' % (filename, data_dir))
    try:
        tar = tarfile.open(filename)
        tar.extractall(path=data_dir)
        tar.close()
    except Exception as e:
        print('Error extracting %s. It is likely due to downloading being terminated, try remove this file and run the script again.' % filename)
        raise

    return filename


def download_all_data(root_dir, gqa=True, vg=True):
    if not os.path.exists(root_dir):
        if len(root_dir) == 0:
            raise ValueError("root_dir must be a valid path")
        os.mkdir(root_dir)

    for name, link in zip(['GQA', 'VG'], [GQA_link, VG_link]):
        if name == 'GQA' and not gqa:
            continue
        if name == 'VG' and not vg:
            continue
        data_dir = os.path.join(root_dir, name)
        print(name, data_dir)
        if not os.path.exists(data_dir):
            if len(data_dir) == 0:
                raise ValueError("data_dir must be a valid path")
            os.mkdir(data_dir)
        download(link, data_dir)

if __name__ == '__main__':
    download_all_data(sys.argv[1])



