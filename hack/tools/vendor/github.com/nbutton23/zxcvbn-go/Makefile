PKG_LIST =  $$( go list ./...  | grep -v /vendor/ | grep -v "zxcvbn-go/data" )

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: test
test: ## Run `go test {Package list}` on the packages
	go test $(PKG_LIST)

.PHONY: lint
lint: ## Run `golint {Package list}`
	golint $(PKG_LIST)