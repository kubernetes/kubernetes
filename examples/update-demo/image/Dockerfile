FROM dockerfile/nginx

ADD data.json /usr/share/nginx/html/data.json
ADD default /etc/nginx/sites-available/default
RUN chmod a+r /usr/share/nginx/html/data.json

CMD ["nginx"]
