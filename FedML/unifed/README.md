## just one step to run the code
1. download `./fedml` and cd into that dir
2. type `docker build -t flbenchmark/fedml .`
3. after that has been done, type `sh run.sh` to start the training
  - delete `--gpus all` in `sh run.sh` can make the training run on `CPU`



## about the config
- before you type `sh run.sh`, make sure the `fedml/config.json` in this dir is right what you want to set.
- `/config_examples` contains some config by myself, please check whether they can run successfully.

## how to get the final result
- I use `flbenchmark.logging` to record my result, you can parse what in `/log` to get the result using this lib.
