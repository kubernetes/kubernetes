.PHONY: build test bench vet coverage

build: vet bench

test:
	go test -v -cover -race

bench:
	go test -v -cover -race -test.bench=. -test.benchmem

vet:
	go vet

coverage:
	go test -coverprofile=c.out && go tool cover -html=c.out && rm c.out
