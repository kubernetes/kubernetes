FROM dmcgowan/token-server:simple

WORKDIR /

COPY ./.htpasswd /.htpasswd
COPY ./certs/auth.localregistry.cert /tls.cert
COPY ./certs/auth.localregistry.key /tls.key
COPY ./certs/signing.key /sign.key
