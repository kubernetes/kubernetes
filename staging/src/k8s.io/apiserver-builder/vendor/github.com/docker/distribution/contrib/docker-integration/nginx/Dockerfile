FROM nginx:1.9

COPY nginx.conf /etc/nginx/nginx.conf
COPY registry.conf /etc/nginx/conf.d/registry.conf
COPY docker-registry-v2.conf /etc/nginx/docker-registry-v2.conf
COPY registry-noauth.conf /etc/nginx/registry-noauth.conf
COPY registry-basic.conf /etc/nginx/registry-basic.conf
COPY test.passwd /etc/nginx/test.passwd
COPY ssl /etc/nginx/ssl
