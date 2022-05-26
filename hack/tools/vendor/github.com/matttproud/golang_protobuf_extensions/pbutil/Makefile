all:

cover:
	go test -cover -v -coverprofile=cover.dat ./...
	go tool cover -func cover.dat

.PHONY: cover
