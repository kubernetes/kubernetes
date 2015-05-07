# A Dockerfile for creating a Kibana container that is designed
# to work with Kubernetes logging.

FROM ubuntu:14.04
MAINTAINER Satnam Singh "satnam@google.com"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y nginx-full curl && \
    apt-get clean

RUN curl -O https://download.elasticsearch.org/kibana/kibana/kibana-3.1.1.tar.gz && \
    tar xf kibana-3.1.1.tar.gz && \
    mv kibana-3.1.1/* /usr/share/nginx/html && \
    rm kibana-3.1.1.tar.gz

RUN echo "daemon off;" >> /etc/nginx/nginx.conf

# ADD default /etc/nginx/sites-available/default
ADD run_kibana_nginx.sh /usr/local/bin/run_kibana_nginx.sh

EXPOSE 80
CMD ["/usr/local/bin/run_kibana_nginx.sh"]
