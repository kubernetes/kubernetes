This directory contains go source, Dockerfile and Makefile for making a test
container which serves requested data on ports specified in ENV variables.

The included localhost.crt is a PEM-encoded TLS cert with SAN IPs
"127.0.0.1" and "[::1]", expiring at the last second of 2049 (the end
of ASN.1 time), generated from src/crypto/tls:
go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h

To use a different cert/key, mount them into the pod and set the 
CERT_FILE and KEY_FILE environment variables to the desired paths.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/porter/README.md?pixel)]()
