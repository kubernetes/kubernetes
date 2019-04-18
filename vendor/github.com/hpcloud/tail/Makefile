default:	test

test:	*.go
	go test -v -race ./...

fmt:
	gofmt -w .

# Run the test in an isolated environment.
fulltest:
	docker build -t hpcloud/tail .
