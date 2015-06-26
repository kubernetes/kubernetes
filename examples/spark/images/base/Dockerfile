FROM java:latest

RUN apt-get update -y
RUN apt-get install -y scala

# Get Spark from some apache mirror.
RUN mkdir -p /opt && \
    cd /opt && \
    wget http://apache.mirrors.pair.com/spark/spark-1.4.0/spark-1.4.0-bin-hadoop2.6.tgz && \
    tar -zvxf spark-1.4.0-bin-hadoop2.6.tgz && \
    rm spark-1.4.0-bin-hadoop2.6.tgz && \
    ln -s spark-1.4.0-bin-hadoop2.6 spark && \
    echo Spark installed in /opt

ADD log4j.properties /opt/spark/conf/log4j.properties
ADD setup_client.sh /
ENV PATH $PATH:/opt/spark/bin
