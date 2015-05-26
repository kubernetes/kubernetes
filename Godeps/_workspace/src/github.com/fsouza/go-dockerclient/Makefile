.PHONY: \
	all \
	vendor \
	lint \
	vet \
	fmt \
	fmtcheck \
	pretest \
	test \
	cov \
	clean

all: test

vendor:
	go get -v github.com/mjibson/party
	party -d vendor -c -u

lint:
	go get -v github.com/golang/lint/golint
	for file in $(shell git ls-files '*.go' | grep -v '^vendor/'); do \
		golint $$file; \
	done

vet:
	go get -v golang.org/x/tools/cmd/vet
	go vet ./...

fmt:
	gofmt -w $(shell git ls-files '*.go' | grep -v '^vendor/')

fmtcheck:
	for file in $(shell git ls-files '*.go' | grep -v '^vendor/'); do \
		gofmt $$file | diff -u $$file -; \
		if [ -n "$$(gofmt $$file | diff -u $$file -)" ]; then\
			exit 1; \
		fi; \
	done

pretest: lint vet fmtcheck

test: pretest
	go test
	go test ./testing

cov:
	go get -v github.com/axw/gocov/gocov
	go get golang.org/x/tools/cmd/cover
	gocov test | gocov report

clean:
	go clean ./...
