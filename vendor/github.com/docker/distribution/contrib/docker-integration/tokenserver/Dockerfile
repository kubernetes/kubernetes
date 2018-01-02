FROM dmcgowan/token-server@sha256:0eab50ebdff5b6b95b3addf4edbd8bd2f5b940f27b41b43c94afdf05863a81af

WORKDIR /

COPY ./.htpasswd /.htpasswd
COPY ./certs/auth.localregistry.cert /tls.cert
COPY ./certs/auth.localregistry.key /tls.key
COPY ./certs/signing.key /sign.key
