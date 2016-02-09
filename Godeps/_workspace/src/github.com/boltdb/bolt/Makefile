BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse --short HEAD`
GOLDFLAGS="-X main.branch $(BRANCH) -X main.commit $(COMMIT)"

default: build

race:
	@go test -v -race -test.run="TestSimulate_(100op|1000op)"

# go get github.com/kisielk/errcheck
errcheck:
	@errcheck -ignorepkg=bytes -ignore=os:Remove github.com/boltdb/bolt

test: 
	@go test -v -cover .
	@go test -v ./cmd/bolt

.PHONY: fmt test
