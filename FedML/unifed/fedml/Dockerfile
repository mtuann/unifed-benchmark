FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget unzip

WORKDIR /
RUN pip install --upgrade pip && \
    pip install torch==1.10.2 && \
    pip install flbenchmark && \
    pip install nest_asyncio

RUN git clone --branch fedml_v0.6_before_fundraising https://github.com/FedML-AI/FedML.git


RUN pip install scikit-learn && \
    pip install numpy && \
    pip install h5py && \
    pip install setproctitle && \
    pip install networkx && \
    pip install tqdm && \
    pip install pympler && pip install sklearn

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY commit.patch /FedML
COPY /untracked/breast_horizontal/data_loader.py /FedML/fedml_api/data_preprocessing/breast_horizontal/
COPY /untracked/breast_vertical/breast_dataset.py /FedML/fedml_api/data_preprocessing/breast_vertical/
COPY /untracked/default_credit_horizontal/data_loader.py /FedML/fedml_api/data_preprocessing/default_credit_horizontal/
COPY /untracked/give_credit_horizontal/data_loader.py  /FedML/fedml_api/data_preprocessing/give_credit_horizontal/
COPY /untracked/student_horizontal/data_loader.py  /FedML/fedml_api/data_preprocessing/student_horizontal/
COPY /untracked/vehicle_scale_horizontal/data_loader.py  /FedML/fedml_api/data_preprocessing/vehicle_scale_horizontal/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/breast_horizontal/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/breast_vertical/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/default_credit_horizontal/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/give_credit_horizontal/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/student_horizontal/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/vehicle_scale_horizontal/
COPY /untracked/preprocessing_data.py /FedML/fedml_api/data_preprocessing/
COPY /untracked/my_model_trainer_regression.py /FedML/fedml_api/standalone/fedavg/
RUN rm /FedML/fedml_api/standalone/fedavg/my_model_trainer_classification.py
COPY /untracked/my_model_trainer_classification.py /FedML/fedml_api/standalone/fedavg/
COPY /untracked/my_model_trainer_nwp.py /FedML/fedml_api/standalone/fedavg/
COPY /untracked/non_linear/__init__.py /FedML/fedml_api/model/non_linear/
COPY /untracked/non_linear/mlp.py /FedML/fedml_api/model/non_linear/
COPY /untracked/nlp/rnn.py /FedML/fedml_api/model/nlp/
COPY /untracked/__init__.py /FedML/fedml_api/model/nlp/
COPY /untracked/cv/lenet.py /FedML/fedml_api/model/cv/
COPY /untracked/cv/alexnet.py /FedML/fedml_api/model/cv/
COPY /untracked/__init__.py /FedML/fedml_api/model/cv/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/our_femnist/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/our_reddit/
COPY /untracked/__init__.py /FedML/fedml_api/data_preprocessing/our_celeba/
COPY /untracked/our_femnist/data_loader.py /FedML/fedml_api/data_preprocessing/our_femnist/
COPY /untracked/our_reddit/data_loader.py /FedML/fedml_api/data_preprocessing/our_reddit/
COPY /untracked/our_celeba/data_loader.py /FedML/fedml_api/data_preprocessing/our_celeba/



WORKDIR /FedML/
RUN git apply --whitespace=warn commit.patch


COPY download.py /FedML

COPY get_report.py /FedML

COPY main_fedavg.py /FedML
COPY entrypoint.sh /FedML
COPY vertical_exp.py /FedML

COPY /untracked/vertical/party_models.py /untracked/vertical/vfl.py /untracked/vertical/vfl_fixture.py /FedML/fedml_api/standalone/classical_vertical_fl/
COPY initialize.py /FedML

ENTRYPOINT [ "bash", "entrypoint.sh"]