FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        python3 \
	python3-dev \
	python3-pip \
        openjdk-8-jdk-headless \
        cmake \
        pkg-config \
        libgoogle-perftools-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN git clone https://github.com/google/sentencepiece.git

RUN cd sentencepiece \
    mkdir build \
    cd build \
    cmake .. \
    make -j $(nproc) \
    sudo make install \
    sudo ldconfig -v
RUN cd ..

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH "$PATH:$HOME/.local/bin"
ENV PYTHONUNBUFFERED TRUE

COPY Pipfile . 
COPY config/config.properties . 
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY models /opt/ml/model

RUN pip3 install pipenv psutil sentencepiece 
RUN pipenv install


EXPOSE 8080 8081 8443 8444

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]

LABEL version="2.1.1" \
      maintainer="james.e.woo@gmail.com" \
      source="https://github.com/jamesewoo/sockeye-serving" \
      description="A containerized service for neural machine translation"
