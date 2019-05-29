all: vet test testrace

deps:
	go get -d -v google.golang.org/grpc/...

updatedeps:
	go get -d -v -u -f google.golang.org/grpc/...

testdeps:
	go get -d -v -t google.golang.org/grpc/...

updatetestdeps:
	go get -d -v -t -u -f google.golang.org/grpc/...

build: deps
	go build google.golang.org/grpc/...

proto:
	@ if ! which protoc > /dev/null; then \
		echo "error: protoc not installed" >&2; \
		exit 1; \
	fi
	go generate google.golang.org/grpc/...

vet:
	./vet.sh

test: testdeps
	go test -cpu 1,4 -timeout 5m google.golang.org/grpc/...

testrace: testdeps
	go test -race -cpu 1,4 -timeout 7m google.golang.org/grpc/...

clean:
	go clean -i google.golang.org/grpc/...

.PHONY: \
	all \
	deps \
	updatedeps \
	testdeps \
	updatetestdeps \
	build \
	proto \
	vet \
	test \
	testrace \
	clean
