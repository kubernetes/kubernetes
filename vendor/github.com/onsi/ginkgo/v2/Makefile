# default task since it's first
.PHONY: all
all:  vet test

.PHONY: test
test:
	go run github.com/onsi/ginkgo/v2/ginkgo -r -p

.PHONY: vet
vet:
	go vet ./...
