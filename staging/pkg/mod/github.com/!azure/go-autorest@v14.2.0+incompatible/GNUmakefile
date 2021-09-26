DIR?=./autorest/

default: build

build: fmt
	go install $(DIR)

test:
	go test $(DIR) || exit 1

vet:
	@echo "go vet ."
	@go vet $(DIR)... ; if [ $$? -eq 1 ]; then \
		echo ""; \
		echo "Vet found suspicious constructs. Please check the reported constructs"; \
		echo "and fix them if necessary before submitting the code for review."; \
		exit 1; \
	fi

fmt:
	gofmt -w $(DIR)

.PHONY: build test vet fmt
