
CMD = jpgo

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  test                    to run all the tests"
	@echo "  build                   to build the library and jp executable"
	@echo "  generate                to run codegen"


generate:
	go generate ./...

build:
	rm -f $(CMD)
	go build ./...
	rm -f cmd/$(CMD)/$(CMD) && cd cmd/$(CMD)/ && go build ./...
	mv cmd/$(CMD)/$(CMD) .

test:
	go test -v ./...

check:
	go vet ./...
	@echo "golint ./..."
	@lint=`golint ./...`; \
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
