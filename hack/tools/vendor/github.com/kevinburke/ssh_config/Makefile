BUMP_VERSION := $(GOPATH)/bin/bump_version
STATICCHECK := $(GOPATH)/bin/staticcheck
WRITE_MAILMAP := $(GOPATH)/bin/write_mailmap

$(STATICCHECK):
	go get honnef.co/go/tools/cmd/staticcheck

lint: $(STATICCHECK)
	go vet ./...
	$(STATICCHECK)

test: lint
	@# the timeout helps guard against infinite recursion
	go test -timeout=250ms ./...

race-test: lint
	go test -timeout=500ms -race ./...

$(BUMP_VERSION):
	go get -u github.com/kevinburke/bump_version

release: test | $(BUMP_VERSION)
	$(BUMP_VERSION) minor config.go

force: ;

AUTHORS.txt: force | $(WRITE_MAILMAP)
	$(WRITE_MAILMAP) > AUTHORS.txt

authors: AUTHORS.txt
