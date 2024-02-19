FROM ubuntu:20.04

ARG TF_VERSION=2.7.0
ARG GENSIM_VERSION=4.1.2
ARG TORCH_VERSION=cu113

RUN \
  apt update -y && \
  apt install software-properties-common -y && \
  apt update -y && \
  apt install -y openjdk-8-jdk \
    build-essential subversion \
    git vim perl curl unzip \
    cpanminus make maven

RUN apt update -y && \
  apt install 

# Java version
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

######################################################################
# Setup Defects4J
######################################################################

# ----------- Step 1. Clone defects4j from github --------------
WORKDIR /
RUN git clone https://github.com/rjust/defects4j.git defects4j

# ----------- Step 2. Initialize Defects4J ---------------------
WORKDIR /defects4j
RUN cpanm --installdeps .
RUN ./init.sh

# ----------- Step 3. Add Defects4J's executables to PATH: ------
ENV PATH="/defects4j/framework/bin:${PATH}"
#--------------


#############################################################################
# Setup GZOLTAR
#############################################################################
RUN git clone https://github.com/GZoltar/gzoltar.git /root/gzoltar && \
    cd /root/gzoltar && \
    mvn install
ENV GZOLTAR_CLI_JAR="/root/gzoltar/com.gzoltar.cli/target/com.gzoltar.cli-1.7.4-SNAPSHOT-jar-with-dependencies.jar"

#############################################################################
# Setup Python
#############################################################################
RUN apt install -y libreadline-dev \
  libncursesw5-dev libssl-dev \
  libsqlite3-dev libgdbm-dev \
  libbz2-dev liblzma-dev zlib1g-dev \
  uuid-dev libffi-dev libdb-dev
RUN curl -o Python-3.9.10.tar.xz https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tar.xz && \
  tar -xf Python-3.9.10.tar.xz && cd Python-3.9.10 && \
  ./configure --enable-optimizations && \
  make -s -j4 && make install
RUN rm -rf Python-3.9.10.tar.xz Python-3.9.10
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python
RUN pip3 install lxml numpy
RUN pip3 install pandas
RUN pip3 install pyyaml

#############################################################################
# Setup Jacoco
#############################################################################
RUN curl -o /root/jacoco.zip -L https://search.maven.org/remotecontent?filepath=org/jacoco/jacoco/0.8.8/jacoco-0.8.8.zip && \
  unzip -q /root/jacoco.zip -d /root/jacoco && rm /root/jacoco.zip
  
#############################################################################
# Setup Clover
#############################################################################
RUN curl -o /root/clover.zip -L https://openclover.org/download/openclover/clover-ant-4.4.1.zip && \
  unzip -q /root/clover.zip && \
  mv clover-ant-4.4.1 /root/clover

#############################################################################
# Setup 色々
#############################################################################
RUN for proj in "Chart" "Math" "Lang"; do mkdir -p /tmp/$proj && defects4j query -p $proj -q "bug.id,classes.modified" -o /tmp/$proj/query.txt; done
ENV PATH="/workspace/bin:${PATH}"

#############################################################################
# Setup tensorflow
#############################################################################
# Install the packages
RUN python -m pip install --no-cache-dir tensorflow-gpu==${TF_VERSION}
RUN pip3 install protobuf==3.20.1
RUN pip3 install tensorflow_addons

#############################################################################
# Setup gensim etc.
#############################################################################
# Install the packages
RUN pip3 install gensim==${GENSIM_VERSION}
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_VERSION}

RUN rm -rf /var/lib/apt/lists/*