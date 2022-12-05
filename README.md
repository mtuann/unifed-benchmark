# UniFed-benchmark

## Setup exps for FLBenchmark-toolkit:
- `conda create -n fedml python=3.7`
- `conda activate fedml`
- `git clone https://github.com/AI-secure/FLBenchmark-toolkit.git`
- `git checkout 166a7a42a6906af1190a15c2f9122ddaf808f39a`
- `cd FLBenchmark-toolkit`
- `pip install .`

### Setup UniFed for FedML
- `git config http.postBuffer 524288000` for the remote end hung up unexpectedly problem
- `git clone https://github.com/FedML-AI/FedML.git`
- `git checkout 7d53a4f39ee41a9769560f3255ad4f6eddd1a01f`
- `python setup.py install`
- `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
- `pip install scikit-learn==1.0.2 urllib3==1.26.1`
- `pip install tqdm wget matplotlib sklearn boto3 gensim networkx pyvnml psutil sentry_sdk shortuuid promise protobuf GitPython docker-pycreds pathtools setproctitle wandb==0.13.2`
<!-- RUN pip install nest_asyncio scikit-learn && \
    pip install numpy && \
    pip install h5py && \
    pip install setproctitle && \
    pip install networkx && \
    pip install tqdm && \
    pip install pympler && pip install sklearn
     -->
Support datasets:
- Horizontal
  - [x] student_horizontal
  - [x] breast_horizontal
  - [x] default_credit_horizontal
  - [x] give_credit_horizontal
  - [x] vehicle_scale_horizontal
- Vertical
  - [x] breast_vertical
  - [] give_credit_vertical
  - [] default_credit_vertical
- Leaf
  - [x] femnist
  - [x] reddit
  - [x] celeba


## Note for some cli:
- `git log --all --decorate --oneline --graph`

<!-- fedml 0.7.355 requires boto3, which is not installed.
fedml 0.7.355 requires gensim, which is not installed.
fedml 0.7.355 requires h5py, which is not installed.
fedml 0.7.355 requires matplotlib, which is not installed.
fedml 0.7.355 requires networkx, which is not installed.
fedml 0.7.355 requires nvidia-ml-py3, which is not installed.
fedml 0.7.355 requires paho-mqtt, which is not installed.
fedml 0.7.355 requires pynvml, which is not installed.
fedml 0.7.355 requires PyYAML, which is not installed.
fedml 0.7.355 requires sklearn, which is not installed.
fedml 0.7.355 requires spacy, which is not installed. -->
