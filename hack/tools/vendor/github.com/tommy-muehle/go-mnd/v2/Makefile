GIT_TAG?= $(shell git describe --abbrev=0)

GO_VERSION = 1.16
BUILDFLAGS := '-w -s'

IMAGE_REPO = "tommymuehle"
BIN = "go-mnd"

clean:
	rm -rf build dist coverage.txt

test:
	go test -race ./...

test-coverage:
	go test -race -coverprofile=coverage.txt -covermode=atomic -coverpkg=./checks,./config

build:
	go build -o build/$(BIN) cmd/mnd/main.go

image:
	@echo "Building the Docker image..."
	docker build --rm -t $(IMAGE_REPO)/$(BIN):$(GIT_TAG) --build-arg GO_VERSION=$(GO_VERSION) .
	docker tag $(IMAGE_REPO)/$(BIN):$(GIT_TAG) $(IMAGE_REPO)/$(BIN):$(GIT_TAG)
	docker tag $(IMAGE_REPO)/$(BIN):$(GIT_TAG) $(IMAGE_REPO)/$(BIN):latest

image-push: image
	@echo "Pushing the Docker image..."
	docker push $(IMAGE_REPO)/$(BIN):$(GIT_TAG)
	docker push $(IMAGE_REPO)/$(BIN):latest

.PHONY: clean test test-coverage build image image-push
