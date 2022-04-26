export CGO_ENABLED=0
go := go
go.goos ?= $(shell echo `go version`|cut -f4 -d ' '|cut -d '/' -f1)
go.goarch ?= $(shell echo `go version`|cut -f4 -d ' '|cut -d '/' -f2)

out.tools := tomll tomljson jsontoml
out.dist := $(out.tools:=_$(go.goos)_$(go.goarch).tar.xz)
sources := $(wildcard **/*.go)


.PHONY:
tools: $(out.tools)

$(out.tools): $(sources)
	GOOS=$(go.goos) GOARCH=$(go.goarch) $(go) build ./cmd/$@

.PHONY:
dist: $(out.dist)

$(out.dist):%_$(go.goos)_$(go.goarch).tar.xz: %
	if [ "$(go.goos)" = "windows" ]; then \
		tar -cJf $@ $^.exe; \
	else \
		tar -cJf $@ $^; \
	fi

.PHONY:
clean:
	rm -rf $(out.tools) $(out.dist)
