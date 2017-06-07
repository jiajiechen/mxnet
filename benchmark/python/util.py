import os
import random


def get_data(data_dir, data_name, url, data_origin_name):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        import urllib
        zippath = os.path.join(data_dir, data_origin_name)
        urllib.urlretrieve(url, zippath)
        # decompress
        os.system("bzip2 -d %r" % data_origin_name)
    os.chdir("..")


def estimate_density():
    """sample 10 times of a size of 1000 for estimating the density of the sparse dataset"""
    if not os.path.exists(DATA_PATH):
        raise Exception("Data is not there!")
    density = []
    for _ in xrange(10):
        num_non_zero = 0
        num_sample = 0
        with open(DATA_PATH) as f:
            for line in f:
                if (random.random() < P):
                    num_non_zero += len(line.split(" ")) - 1
                    num_sample += 1
        density.append(num_non_zero * 1.0 / (feature_size * num_sample))
    return sum(density) * 100 / len(density)


def slide_mini(data_name, new_name, sample_size=3000):
    if os.path.exists(data_name):
        with open(data_name, 'rb') as f:
            new_f = open(new_name, 'wb')
            for _ in xrange(sample_size):
                new_f.write(f.readline())
            new_f.close()


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data')
    data_name = 'kdda.t'
    data_origin_name = 'kdda.t.bz2'
                    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2"
                         feature_dim = 20216883
                              density = 0.0001866
                                   path = os.path.join(data_dir, data_name)
    DATA_PATH = "data/kdda.t"
    P = 0.01 #sampling proportion
    feature_size = 20216830
    random.seed(10002)
    print estimate_density()
