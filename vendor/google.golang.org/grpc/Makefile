all: vet test testrace

build: deps
	go build google.golang.org/grpc/...

clean:
	go clean -i google.golang.org/grpc/...

deps:
	go get -d -v google.golang.org/grpc/...

proto:
	@ if ! which protoc > /dev/null; then \
		echo "error: protoc not installed" >&2; \
		exit 1; \
	fi
	go generate google.golang.org/grpc/...

test: testdeps
	go test -cpu 1,4 -timeout 7m google.golang.org/grpc/...

testsubmodule: testdeps
	cd security/advancedtls && go test -cpu 1,4 -timeout 7m google.golang.org/grpc/security/advancedtls/...

testappengine: testappenginedeps
	goapp test -cpu 1,4 -timeout 7m google.golang.org/grpc/...

testappenginedeps:
	goapp get -d -v -t -tags 'appengine appenginevm' google.golang.org/grpc/...

testdeps:
	go get -d -v -t google.golang.org/grpc/...

testrace: testdeps
	go test -race -cpu 1,4 -timeout 7m google.golang.org/grpc/...

updatedeps:
	go get -d -v -u -f google.golang.org/grpc/...

updatetestdeps:
	go get -d -v -t -u -f google.golang.org/grpc/...

vet: vetdeps
	./vet.sh

vetdeps:
	./vet.sh -install

.PHONY: \
	all \
	build \
	clean \
	deps \
	proto \
	test \
	testappengine \
	testappenginedeps \
	testdeps \
	testrace \
	updatedeps \
	updatetestdeps \
	vet \
	vetdeps
