all: build

.PHONY: build
build: 
	go build -o tparallel ./cmd/tparallel

.PHONY: build_race
build_race:
	go build -race -o tparallel ./cmd/tparallel

.PHONY: test
test: build_race
	go test -v ./...
