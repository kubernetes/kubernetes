FROM ubuntu:14.04
LABEL maintainer Erik Hollensbe <erik@hollensbe.org>

RUN apt-get update && apt-get install nginx-full -y
RUN rm -rf /etc/nginx
ADD etc /etc/nginx
RUN chown -R root:root /etc/nginx
RUN /usr/sbin/nginx -qt
RUN mkdir /www

CMD ["/usr/sbin/nginx"]

VOLUME /www
EXPOSE 80
