.PHONY: ci generate clean

ci: clean generate
	go test -v ./...

generate:
	go generate .

clean:
	rm -rf *_generated*.go
