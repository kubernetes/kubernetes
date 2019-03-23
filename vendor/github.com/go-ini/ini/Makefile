.PHONY: build test bench vet

build: vet bench

test:
	go test -v -cover -race

bench:
	go test -v -cover -race -test.bench=. -test.benchmem

vet:
	go vet
