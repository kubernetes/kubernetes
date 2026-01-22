# default task since it's first
.PHONY: all
all:  vet test

.PHONY: test
test:
	go run github.com/onsi/ginkgo/v2/ginkgo -r -p -randomize-all -keep-going

.PHONY: vet
vet:
	go vet ./...

.PHONY: update-deps
update-deps:
	go get -u ./...
	go mod tidy