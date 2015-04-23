.PHONY: \
	all \
	deps \
	updatedeps \
	testdeps \
	updatetestdeps \
	cov \
	test \
	clean

all: test

deps:
	go get -d -v ./...

updatedeps:
	go get -d -v -u -f ./...

testdeps:
	go get -d -v -t ./...

updatetestdeps:
	go get -d -v -t -u -f ./...

cov: testdeps
	go get -v github.com/axw/gocov/gocov
	go get golang.org/x/tools/cmd/cover
	gocov test | gocov report

test: testdeps
	go test ./...
	./testing/bin/fmtpolice

clean:
	go clean ./...
