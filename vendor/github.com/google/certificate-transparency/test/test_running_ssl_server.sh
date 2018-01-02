PORT=${1:-4433}

ln -sf /tmp/ct-ca/ca-cert.pem /tmp/ct-ca/`LD_LIBRARY_PATH=$OPENSSLDIR $OPENSSLDIR/apps/openssl x509 -hash -noout -in /tmp/ct-ca/ca-cert.pem`.0

../cpp/client/ct connect --ct_server_public_key=testdata/ct-server-key-public.pem --ssl_server=127.0.0.1 --ssl_server_port=$PORT --ssl_client_trusted_cert_dir=/tmp/ct-ca --ssl_client_require_sct
