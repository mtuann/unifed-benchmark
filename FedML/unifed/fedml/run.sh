docker run --rm --gpus all --net=host -v $PWD/config.json:/FedML/config.json -v $PWD/../data/:/data/ -v $PWD/log/:/FedML/log  flbenchmark/fedml
