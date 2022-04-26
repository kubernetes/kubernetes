.PHONY: benchmark docs lint test

docs:
	which godoc2ghmd || ( go get github.com/DevotedHealth/godoc2ghmd && go mod tidy )
	godoc2ghmd -template .readme.tmpl github.com/ettle/strcase > README.md

test:
	go test -cover ./...

lint:
	which golangci-lint || ( go get github.com/golangci/golangci-lint/cmd/golangci-lint@v1.27.0 && go mod tidy )
	golangci-lint run
	golangci-lint run benchmark/*.go

benchmark:
	cd benchmark && go test -bench=. -test.benchmem && go mod tidy
