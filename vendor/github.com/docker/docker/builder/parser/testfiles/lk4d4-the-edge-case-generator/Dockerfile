FROM busybox:buildroot-2014.02

MAINTAINER docker <docker@docker.io>

ONBUILD RUN ["echo", "test"]
ONBUILD RUN echo test
ONBUILD COPY . /


# RUN Commands \
# linebreak in comment \
RUN ["ls", "-la"]
RUN ["echo", "'1234'"]
RUN echo "1234"
RUN echo 1234
RUN echo '1234' && \
    echo "456" && \
    echo 789
RUN    sh -c 'echo root:testpass \
        > /tmp/passwd'
RUN mkdir -p /test /test2 /test3/test

# ENV \
ENV SCUBA 1 DUBA 3
ENV SCUBA "1 DUBA 3"

# CMD \
CMD ["echo", "test"]
CMD echo test
CMD echo "test"
CMD echo 'test'
CMD echo 'test' | wc -

#EXPOSE\
EXPOSE 3000
EXPOSE 9000 5000 6000

USER docker
USER docker:root

VOLUME ["/test"]
VOLUME ["/test", "/test2"]
VOLUME /test3

WORKDIR /test

ADD . /
COPY . copy
