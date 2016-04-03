.PHONY: all deps test validate

all: deps test validate

deps:
	go get -t ./...
	go get github.com/golang/lint/golint

test:
	go test -race -cover ./...

validate:
	go vet ./...
	test -z "$(golint ./... | tee /dev/stderr)"
	test -z "$(gofmt -s -l . | tee /dev/stderr)"
