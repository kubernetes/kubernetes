test: 
	gofmt -s -w .
	go test ./...
	go get bitbucket.org/bertimus9/systemstat

coverage: 
	go get github.com/axw/gocov/gocov
	gocov test . | gocov report
