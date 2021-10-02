FROM golang

RUN mkdir -p $GOPATH/src/github.com/nxadm/tail/
ADD . $GOPATH/src/github.com/nxadm/tail/

# expecting to fetch dependencies successfully.
RUN go get -v github.com/nxadm/tail

# expecting to run the test successfully.
RUN go test -v github.com/nxadm/tail

# expecting to install successfully
RUN go install -v github.com/nxadm/tail
RUN go install -v github.com/nxadm/tail/cmd/gotail

RUN $GOPATH/bin/gotail -h || true

ENV PATH $GOPATH/bin:$PATH
CMD ["gotail"]
