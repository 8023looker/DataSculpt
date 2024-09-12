FROM emr-vke-public-cn-beijing.cr.volces.com/emr/ray:2.33.0-py3.9-ubuntu20.04-248-1.5.0

RUN apt-get update -y
RUN apt-get install build-essential -y

COPY ./requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
