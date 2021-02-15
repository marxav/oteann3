import os
import tarfile
import zipfile
import requests

CONFIG = {
    'subdatasets_dir': 'subdatasets',
    'input_file': 'dataset.tar.gz',
    'subdataset': 'wikt_samples.csv',
    'root_dir': '.',
    'languages': ['ent', 'eno', 'ar', 'br', 'de', 
                  'en', 'eo', 'es', 'fi', 'fr', 
                  'fro', 'it', 'ko', 'nl', 'pt', 
                  'ru', 'sh', 'tr', 'zh']
}

def extract_files(config):

    cwd = os.getcwd()
    os.chdir(config['root_dir'])
    dest_dir = config['root_dir'] + '/' + config['subdatasets_dir']
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    try:
        print(os.listdir())
        print('"""""""""""""""""""""') 
        filename = config['input_file']
        file = tarfile.open(filename, 'r:gz')
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd) 


def concatenate_all(config):
    print('concatenate all')
    with open(config['subdataset'], 'w') as outfile:
        for language in config['languages']:
            fname = config['subdatasets_dir'] + '/' + language + '_wikt_samples.csv'
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def main():
    config = CONFIG
    extract_files(config)
    concatenate_all(config)

if __name__ == "__main__":
    main()



