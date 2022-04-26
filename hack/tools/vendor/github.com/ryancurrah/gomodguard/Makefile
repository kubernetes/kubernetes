current_dir = $(shell pwd)

.PHONY: lint
lint:
	golangci-lint run ./...

.PHONY: build
build:
	go build -o gomodguard cmd/gomodguard/main.go

.PHONY: run
run: build
	./gomodguard

.PHONY: test
test:
	go test -v -coverprofile coverage.out 

.PHONY: cover
cover:
	gocover-cobertura < coverage.out > coverage.xml

.PHONY: dockerrun
dockerrun: dockerbuild
	docker run -v "${current_dir}/.gomodguard.yaml:/.gomodguard.yaml" ryancurrah/gomodguard:latest

.PHONY: release
release:
	goreleaser --rm-dist

.PHONY: clean
clean:
	rm -rf dist/
	rm -f gomodguard coverage.xml coverage.out

.PHONY: install-tools-mac
install-tools-mac:
	brew install goreleaser/tap/goreleaser

.PHONY: install-go-tools
install-go-tools:
	go get github.com/t-yuki/gocover-cobertura
