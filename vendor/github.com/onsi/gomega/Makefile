###### Help ###################################################################

.DEFAULT_GOAL = help

.PHONY: help

help:  ## list Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

###### Targets ################################################################

test: version download fmt vet ginkgo ## Runs all build, static analysis, and test steps

download: ## Download dependencies
	go mod download

vet: ## Run static code analysis
	go vet ./...

ginkgo: ## Run tests using Ginkgo
	go run github.com/onsi/ginkgo/ginkgo -p -r --randomizeAllSpecs --failOnPending --randomizeSuites --race

fmt: ## Checks that the code is formatted correcty
	@@if [ -n "$$(gofmt -s -e -l -d .)" ]; then                   \
		echo "gofmt check failed: run 'gofmt -s -e -l -w .'"; \
		exit 1;                                               \
	fi

docker_test: ## Run tests in a container via docker-compose
	docker-compose build test && docker-compose run --rm test make test

version: ## Display the version of Go
	@@go version
