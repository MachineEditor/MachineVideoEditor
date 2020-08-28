#!/usr/bin/env python3
import os, urllib.request, re, threading, posixpath, urllib.parse, argparse, socket, time, hashlib, pickle, signal, imghdr

#config
output_dir = './bing' #default output dir
adult_filter = True #Do not disable adult filter by default
socket.setdefaulttimeout(2)

in_progress = tried_urls = []
image_md5s = {}
urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}

def download(pool_sema: threading.Semaphore, url: str, output_dir: str):
    if url in tried_urls:
        return
    pool_sema.acquire()
    path = urllib.parse.urlsplit(url).path
    filename = posixpath.basename(path).split('?')[0] #Strip GET parameters from filename
    name, ext = os.path.splitext(filename)
    name = name[:36]
    filename = name + ext

    i = 0
    while os.path.exists(os.path.join(output_dir, filename)) or filename in in_progress:
        i += 1
        filename = "%s-%d%s" % (name, i, ext)
    in_progress.append(filename)
    try:
        request=urllib.request.Request(url,None,urlopenheader)
        image=urllib.request.urlopen(request).read()
        if not imghdr.what(None, image):
            print('FAIL: Invalid image, not saving ' + filename)
            return

        md5_key = hashlib.md5(image).hexdigest()
        if md5_key in image_md5s:
            print('FAIL: Image is a duplicate of ' + image_md5s[md5_key] + ', not saving ' + filename)
            return

        image_md5s[md5_key] = filename

        imagefile=open(os.path.join(output_dir, filename),'wb')
        imagefile.write(image)
        imagefile.close()
        print("OK: " + filename)
        tried_urls.append(url)
    except Exception as e:
        print("FAIL: " + filename)
    finally:
        in_progress.remove(filename)
        pool_sema.release()

def fetch_images_from_keyword(pool_sema: threading.Semaphore, keyword: str, output_dir: str, filters: str, limit: int):
    current = 0
    last = ''
    while True:
        request_url='https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(keyword) + '&first=' + str(current) + '&count=35&adlt=' + adlt + '&qft=' + ('' if filters is None else filters)
        request=urllib.request.Request(request_url,None,headers=urlopenheader)
        response=urllib.request.urlopen(request)
        html = response.read().decode('utf8')
        links = re.findall('murl&quot;:&quot;(.*?)&quot;',html)
        try:
            if links[-1] == last:
                return
            for index, link in enumerate(links):
                if limit is not None and current + index >= limit:
                    return
                t = threading.Thread(target = download,args = (pool_sema, link, output_dir))
                t.start()
                current += 1
            last = links[-1]
        except IndexError:
            print('No search results for "{0}"'.format(keyword))
            return
        time.sleep(0.1)

def backup_history(*args):
    download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'wb')
    pickle.dump(tried_urls,download_history)
    copied_image_md5s = dict(image_md5s)  #We are working with the copy, because length of input variable for pickle must not be changed during dumping
    pickle.dump(copied_image_md5s, download_history)
    download_history.close()
    print('history_dumped')
    if args:
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Bing image bulk downloader')
    parser.add_argument('-s', '--search-string', help = 'Keyword to search', required = False)
    parser.add_argument('-f', '--search-file', help = 'Path to a file containing search strings line by line', required = False)
    parser.add_argument('-o', '--output', help = 'Output directory', required = False)
    parser.add_argument('--adult-filter-on', help ='Enable adult filter', action = 'store_true', required = False)
    parser.add_argument('--adult-filter-off', help = 'Disable adult filter', action = 'store_true', required = False)
    parser.add_argument('--filters', help = 'Any query based filters you want to append when searching for images, e.g. +filterui:license-L1', required = False)
    parser.add_argument('--limit', help = 'Make sure not to search for more than specified amount of images.', required = False, type = int)
    parser.add_argument('--threads', help = 'Number of threads', type = int, default = 20)
    args = parser.parse_args()
    if (not args.search_string) and (not args.search_file):
        parser.error('Provide Either search string or path to file containing search strings')
    if args.output:
        output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_origin = output_dir
    signal.signal(signal.SIGINT, backup_history)
    try:
        download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'rb')
        tried_urls=pickle.load(download_history)
        image_md5s=pickle.load(download_history)
        download_history.close()
    except (OSError, IOError):
        tried_urls=[]
    if adult_filter:
        adlt = ''
    else:
        adlt = 'off'
    if args.adult_filter_off:
        adlt = 'off'
    elif args.adult_filter_on:
        adlt = ''
    pool_sema = threading.BoundedSemaphore(args.threads)
    if args.search_string:
        fetch_images_from_keyword(pool_sema, args.search_string,output_dir, args.filters, args.limit)
    elif args.search_file:
        try:
            inputFile=open(args.search_file)
        except (OSError, IOError):
            print("Couldn't open file {}".format(args.search_file))
            exit(1)
        for keyword in inputFile.readlines():
            output_sub_dir = os.path.join(output_dir_origin, keyword.strip().replace(' ', '_'))
            if not os.path.exists(output_sub_dir):
                os.makedirs(output_sub_dir)
            fetch_images_from_keyword(pool_sema, keyword,output_sub_dir, args.filters, args.limit)
            backup_history()
        inputFile.close()