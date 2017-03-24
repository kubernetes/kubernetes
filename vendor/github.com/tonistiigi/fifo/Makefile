.PHONY: fmt vet test deps

test: deps
	go test -v ./...

deps:
	go get -d -t ./...

fmt:
	gofmt -s -l .

vet:
	go vet ./...
