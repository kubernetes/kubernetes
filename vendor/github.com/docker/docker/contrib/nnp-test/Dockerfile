FROM buildpack-deps:jessie

COPY . /usr/src/

WORKDIR /usr/src/

RUN gcc -g -Wall -static nnp-test.c -o /usr/bin/nnp-test

RUN chmod +s /usr/bin/nnp-test
