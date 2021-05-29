
CMD = jpgo

SRC_PKGS=./ ./cmd/... ./fuzz/...

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  test                    to run all the tests"
	@echo "  build                   to build the library and jp executable"
	@echo "  generate                to run codegen"


generate:
	go generate ${SRC_PKGS}

build:
	rm -f $(CMD)
	go build ${SRC_PKGS}
	rm -f cmd/$(CMD)/$(CMD) && cd cmd/$(CMD)/ && go build ./...
	mv cmd/$(CMD)/$(CMD) .

test: test-internal-testify
	echo "making tests ${SRC_PKGS}"
	go test -v ${SRC_PKGS}

check:
	go vet ${SRC_PKGS}
	@echo "golint ${SRC_PKGS}"
	@lint=`golint ${SRC_PKGS}`; \
	lint=`echo "$$lint" | grep -v "astnodetype_string.go" | grep -v "toktype_string.go"`; \
	echo "$$lint"; \
	if [ "$$lint" != "" ]; then exit 1; fi

htmlc:
	go test -coverprofile="/tmp/jpcov"  && go tool cover -html="/tmp/jpcov" && unlink /tmp/jpcov

buildfuzz:
	go-fuzz-build github.com/jmespath/go-jmespath/fuzz

fuzz: buildfuzz
	go-fuzz -bin=./jmespath-fuzz.zip -workdir=fuzz/testdata

bench:
	go test -bench . -cpuprofile cpu.out

pprof-cpu:
	go tool pprof ./go-jmespath.test ./cpu.out

test-internal-testify:
	cd internal/testify && go test ./...

