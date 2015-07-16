FROM debian:jessie

RUN apt-get update
RUN apt-get -qy install python-seqdiag make curl

WORKDIR /diagrams

RUN curl -sLo DroidSansMono.ttf https://googlefontdirectory.googlecode.com/hg/apache/droidsansmono/DroidSansMono.ttf

ADD . /diagrams

CMD bash -c 'make >/dev/stderr && tar cf - *.png'