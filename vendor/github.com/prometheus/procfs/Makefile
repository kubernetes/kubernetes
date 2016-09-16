ci:
	go fmt
	go vet
	go test -v ./...
	go get github.com/golang/lint/golint
	golint *.go
