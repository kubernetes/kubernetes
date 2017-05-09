FROM busybox

ADD _output/dockerized/bin/linux/amd64 /_output/dockerized/bin/linux/amd64

CMD ["/bin/true"]

VOLUME ["/_output"]
