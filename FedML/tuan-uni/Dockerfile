FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget unzip

RUN pip install --upgrade pip 
RUN pip install scikit-learn==1.1.3
RUN pip install flbenchmark==0.1.3
RUN pip install urllib3==1.26.11

RUN pip install nest_asyncio scikit-learn && \
    pip install numpy && \
    pip install h5py && \
    pip install setproctitle && \
    pip install networkx && \
    pip install tqdm && \
    pip install pympler && pip install sklearn

WORKDIR /
RUN git clone https://github.com/FedML-AI/FedML.git
WORKDIR /FedML 
RUN git checkout 7d53a4f39ee41a9769560f3255ad4f6eddd1a01f
WORKDIR /FedML/python
RUN python setup.py install

WORKDIR /

# COPY src/ /FedML/src/
COPY wrap/ /FedML/wrap/


COPY download.py /FedML
COPY initialize.py /FedML

COPY main_fedavg.py /FedML
COPY vertical_exp.py /FedML

COPY get_report.py /FedML

COPY entrypoint.sh /FedML

WORKDIR /FedML

ENTRYPOINT [ "bash", "entrypoint.sh"]