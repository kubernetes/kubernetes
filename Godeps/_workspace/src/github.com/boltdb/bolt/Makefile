TEST=.
BENCH=.
COVERPROFILE=/tmp/c.out
BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse --short HEAD`
GOLDFLAGS="-X main.branch $(BRANCH) -X main.commit $(COMMIT)"

default: build

bench:
	go test -v -test.run=NOTHINCONTAINSTHIS -test.bench=$(BENCH)

# http://cloc.sourceforge.net/
cloc:
	@cloc --not-match-f='Makefile|_test.go' .

cover: fmt
	go test -coverprofile=$(COVERPROFILE) -test.run=$(TEST) $(COVERFLAG) .
	go tool cover -html=$(COVERPROFILE)
	rm $(COVERPROFILE)

cpuprofile: fmt
	@go test -c
	@./bolt.test -test.v -test.run=$(TEST) -test.cpuprofile cpu.prof

# go get github.com/kisielk/errcheck
errcheck:
	@echo "=== errcheck ==="
	@errcheck github.com/boltdb/bolt

fmt:
	@go fmt ./...

get:
	@go get -d ./...

build: get
	@mkdir -p bin
	@go build -ldflags=$(GOLDFLAGS) -a -o bin/bolt ./cmd/bolt

test: fmt
	@go get github.com/stretchr/testify/assert
	@echo "=== TESTS ==="
	@go test -v -cover -test.run=$(TEST)
	@echo ""
	@echo ""
	@echo "=== CLI ==="
	@go test -v -test.run=$(TEST) ./cmd/bolt
	@echo ""
	@echo ""
	@echo "=== RACE DETECTOR ==="
	@go test -v -race -test.run="TestSimulate_(100op|1000op)"

.PHONY: bench cloc cover cpuprofile fmt memprofile test
