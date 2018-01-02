FROM dmcgowan/token-server@sha256:5a6f76d3086cdf63249c77b521108387b49d85a30c5e1c4fe82fdf5ae3b76ba7

WORKDIR /

COPY ./.htpasswd /.htpasswd
COPY ./certs/auth.localregistry.cert /tls.cert
COPY ./certs/auth.localregistry.key /tls.key
COPY ./certs/signing.key /sign.key
