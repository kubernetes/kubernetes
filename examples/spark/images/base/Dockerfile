FROM java:latest

RUN apt-get update -y
RUN apt-get install -y scala

ENV version 1.5.1

# Get Spark from some apache mirror.
RUN mkdir -p /opt && \
    cd /opt && \
    wget http://apache.mirrors.pair.com/spark/spark-${version}/spark-${version}-bin-hadoop2.6.tgz && \
    tar -zvxf spark-${version}-bin-hadoop2.6.tgz && \
    rm spark-${version}-bin-hadoop2.6.tgz && \
    ln -s spark-${version}-bin-hadoop2.6 spark && \
    echo Spark ${version} installed in /opt

ADD log4j.properties /opt/spark/conf/log4j.properties
ADD setup_client.sh /
ENV PATH $PATH:/opt/spark/bin
