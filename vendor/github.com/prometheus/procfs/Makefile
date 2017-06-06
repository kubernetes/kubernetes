ci:
	! gofmt -l *.go | read nothing
	go vet
	go test -v ./...
	go get github.com/golang/lint/golint
	golint *.go
