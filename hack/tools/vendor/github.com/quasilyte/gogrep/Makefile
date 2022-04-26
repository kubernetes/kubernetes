GOPATH_DIR=`go env GOPATH`

test:
	go test -count 2 -coverpkg=./... -coverprofile=coverage.txt -covermode=atomic ./...
	go test -bench=. ./...
	@echo "everything is OK"

ci-lint:
	curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(GOPATH_DIR)/bin v1.43.0
	$(GOPATH_DIR)/bin/golangci-lint run ./...
	go install github.com/quasilyte/go-consistent@latest
	$(GOPATH_DIR)/bin/go-consistent . ./internal/... ./nodetag/... ./filters/...
	@echo "everything is OK"

lint:
	golangci-lint run ./...
	@echo "everything is OK"

.PHONY: ci-lint lint test
