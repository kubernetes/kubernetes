FROM google/golang:latest

ADD . /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/sleepy-pod/_src

WORKDIR /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/sleepy-pod/
RUN cd _src/ && go get && go build -o sleepy
RUN cp _src/sleepy-container/Dockerfile .

CMD echo "Copying sleepy binary to directory mounted at /sleepy-bin" && cp _src/sleepy /sleepy-bin
