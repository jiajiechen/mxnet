#!/bin/bash

dmlc_download() {
    url=http://data.mxnet.io/mxnet/data
    dir=$1
    file=$2
    if [ ! -e data/${dir}/$file ]; then
        wget ${url}/${dir}/${file} -P data/${dir}/ || exit -1
    else
        echo "data/${dir}/$file already exits"
    fi
}

dmlc_download mnist t10k-images-idx3-ubyte.gz
dmlc_download mnist t10k-labels-idx1-ubyte.gz
dmlc_download mnist train-images-idx3-ubyte.gz
dmlc_download mnist train-labels-idx1-ubyte.gz

dmlc_download cifar10 cifar10_train.rec
dmlc_download cifar10 cifar10_val.rec
