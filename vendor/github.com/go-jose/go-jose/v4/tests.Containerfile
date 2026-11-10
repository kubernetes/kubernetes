# This containerfile can be used to run the go-jose tests and cli

FROM golang:1.27-trixie
ENV GOTOOLCHAIN=local

COPY . /src
WORKDIR /src

# Install the jose-util cli, which will also build the library
RUN go build -o /usr/local/bin/jose-util ./jose-util

# By default, run the tests
CMD ["go", "test", "./..."]
