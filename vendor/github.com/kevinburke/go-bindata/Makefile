SHELL = /bin/bash -o pipefail

BENCHSTAT := $(GOPATH)/bin/benchstat
DIFFER := $(GOPATH)/bin/differ
STATICCHECK := $(GOPATH)/bin/staticcheck
RELEASE := $(GOPATH)/bin/github-release
WRITE_MAILMAP := $(GOPATH)/bin/write_mailmap

UNAME := $(shell uname)

all:
	$(MAKE) -C testdata

$(DIFFER):
ifeq ($(UNAME), Darwin)
	curl --silent --location --output $(GOPATH)/bin/differ https://github.com/kevinburke/differ/releases/download/0.5/differ-darwin-amd64 && chmod 755 $(GOPATH)/bin/differ
endif
ifeq ($(UNAME), Linux)
	curl --silent --location --output $(GOPATH)/bin/differ https://github.com/kevinburke/differ/releases/download/0.5/differ-linux-amd64 && chmod 755 $(GOPATH)/bin/differ
endif

diff-testdata: | $(DIFFER)
	$(DIFFER) $(MAKE) -C testdata
	$(DIFFER) go fmt ./testdata/out/...

$(STATICCHECK):
	go get honnef.co/go/tools/cmd/staticcheck

lint: | $(STATICCHECK)
	go vet ./...
	$(STATICCHECK) ./...

go-test:
	go test ./...

go-race-test:
	go test -race ./...

test: go-test
	$(MAKE) -C testdata

race-test: lint go-race-test
	$(MAKE) -C testdata

$(GOPATH)/bin/go-bindata:
	go install -v ./...

$(BENCHSTAT):
	go get golang.org/x/perf/cmd/benchstat

bench: $(GOPATH)/bin/go-bindata | $(BENCHSTAT)
	go list ./... | grep -v vendor | xargs go test -benchtime=5s -bench=. -run='^$$' 2>&1 | $(BENCHSTAT) /dev/stdin

$(WRITE_MAILMAP):
	go get -u github.com/kevinburke/write_mailmap

force: ;

AUTHORS.txt: force | $(WRITE_MAILMAP)
	$(WRITE_MAILMAP) > AUTHORS.txt

authors: AUTHORS.txt

ci: lint go-race-test diff-testdata

# Ensure you have updated go-bindata/version.go manually.
release: | $(RELEASE) race-test diff-testdata
ifndef version
	@echo "Please provide a version"
	exit 1
endif
ifndef GITHUB_TOKEN
	@echo "Please set GITHUB_TOKEN in the environment"
	exit 1
endif
	# If you don't push these, Github creates a tagged release for you from the
	# wrong commit.
	git push origin master
	git push origin --tags
	mkdir -p releases/$(version)
	GOOS=linux GOARCH=amd64 go build -o releases/$(version)/go-bindata-linux-amd64 ./go-bindata
	GOOS=darwin GOARCH=amd64 go build -o releases/$(version)/go-bindata-darwin-amd64 ./go-bindata
	GOOS=windows GOARCH=amd64 go build -o releases/$(version)/go-bindata-windows-amd64 ./go-bindata
	# these commands are not idempotent so ignore failures if an upload repeats
	$(RELEASE) release --user kevinburke --repo go-bindata --tag $(version) || true
	$(RELEASE) upload --user kevinburke --repo go-bindata --tag $(version) --name go-bindata-linux-amd64 --file releases/$(version)/go-bindata-linux-amd64 || true
	$(RELEASE) upload --user kevinburke --repo go-bindata --tag $(version) --name go-bindata-darwin-amd64 --file releases/$(version)/go-bindata-darwin-amd64 || true
	$(RELEASE) upload --user kevinburke --repo go-bindata --tag $(version) --name go-bindata-windows-amd64 --file releases/$(version)/go-bindata-windows-amd64 || true
