FROM java:7-jre

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

RUN cd / && \
    curl -O https://download.elastic.co/elasticsearch/elasticsearch/elasticsearch-1.5.2.tar.gz && \
    tar xf elasticsearch-1.5.2.tar.gz && \
    rm elasticsearch-1.5.2.tar.gz

COPY elasticsearch.yml /elasticsearch-1.5.2/config/elasticsearch.yml
COPY run.sh /
COPY elasticsearch_discovery /

EXPOSE 9200 9300

CMD ["/run.sh"]