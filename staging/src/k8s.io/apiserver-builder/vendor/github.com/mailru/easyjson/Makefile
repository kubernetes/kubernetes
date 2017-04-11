PKG=github.com/mailru/easyjson
GOPATH:=$(PWD)/.root:$(GOPATH)
export GOPATH

all: test

.root/src/$(PKG): 	
	mkdir -p $@
	for i in $$PWD/* ; do ln -s $$i $@/`basename $$i` ; done 

root: .root/src/$(PKG)

clean:
	rm -rf .root

build:
	go build -i -o .root/bin/easyjson $(PKG)/easyjson

generate: root build
	.root/bin/easyjson -stubs \
		.root/src/$(PKG)/tests/snake.go \
		.root/src/$(PKG)/tests/data.go \
		.root/src/$(PKG)/tests/omitempty.go \
		.root/src/$(PKG)/tests/nothing.go

	.root/bin/easyjson -all .root/src/$(PKG)/tests/data.go 
	.root/bin/easyjson -all .root/src/$(PKG)/tests/nothing.go
	.root/bin/easyjson -snake_case .root/src/$(PKG)/tests/snake.go
	.root/bin/easyjson -omit_empty .root/src/$(PKG)/tests/omitempty.go
	.root/bin/easyjson -build_tags=use_easyjson .root/src/$(PKG)/benchmark/data.go

test: generate root
	go test \
		$(PKG)/tests \
		$(PKG)/jlexer \
		$(PKG)/gen \
		$(PKG)/buffer
	go test -benchmem -tags use_easyjson -bench . $(PKG)/benchmark
	golint -set_exit_status .root/src/$(PKG)/tests/*_easyjson.go

bench-other: generate root
	@go test -benchmem -bench . $(PKG)/benchmark
	@go test -benchmem -tags use_ffjson -bench . $(PKG)/benchmark
	@go test -benchmem -tags use_codec -bench . $(PKG)/benchmark

bench-python:
	benchmark/ujson.sh


.PHONY: root clean generate test build
