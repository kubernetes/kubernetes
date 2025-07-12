SHELL = /usr/bin/env bash
.SHELLFLAGS = -ecuo pipefail
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
.DEFAULT_GOAL := help

# ==============================================================================
# Variables
# ==============================================================================
CI ?=
OUTPUTDIR ?= output
$(shell mkdir -p $(OUTPUTDIR))
COVERPROFILE ?= coverage.out
COVERHTML ?= coverage.html
GOFILES := $(shell go list -f '{{ range $$file := .GoFiles }}{{ printf "%s/%s\n" $$.Dir $$file }} {{- end }}' ./... | sort)
GOTESTFILES := $(shell go list -f '{{ range $$file := .TestGoFiles }}{{ printf "%s/%s\n" $$.Dir $$file }} {{- end }}' ./... | sort)
FMTSTAMP := .fmt.stamp
LINTSTAMP := .lint.stamp

# ==============================================================================
# Functions
# ==============================================================================
define open_browser
	@case $$(uname -s) in \
		Linux) xdg-open $(1) ;; \
		Darwin) open $(1) ;; \
		*) echo "Unsupported platform" ;; \
	esac
endef

# ==============================================================================
# Targets
# ==============================================================================

## Dependencies:
.PHONY: deps/get
deps/get: ## Get dependencies
	go get -v ./...
	go mod tidy -v

.PHONY: deps/update
deps/update: ## Update dependencies
	go get -v -u ./...
	go mod tidy -v

## Generators:
.PHONY: gif
gif: ## Generate GIF for example application
	./scripts/gif.sh

## Code Quality:
fmt: $(FMTSTAMP) ## Format code

$(FMTSTAMP): $(GOFILES) $(GOTESTFILES)
	golangci-lint run --fix --verbose

lint: $(LINTSTAMP) ## Run linters

$(LINTSTAMP): $(GOFILES) $(GOTESTFILES)
	golangci-lint run --verbose
	touch $@

## Testing:
test: $(OUTPUTDIR)/$(COVERHTML) ## Run tests

$(OUTPUTDIR)/$(COVERPROFILE): $(GOFILES) $(GOTESTFILES) go.mod ## Run tests with coverage
	mkdir -pv $(dir $@)
	go test -v \
		-outputdir=$(dir $@) \
		-coverprofile=$(notdir $@) \
		-coverpkg=./... \
		-run= ./... | \
		tee $(OUTPUTDIR)/test.log

$(OUTPUTDIR)/$(COVERHTML): $(OUTPUTDIR)/$(COVERPROFILE) ## Generate HTML coverage report
ifeq ($(CI),)
	mkdir -pv $(dir $@)
	go tool cover -html=$(OUTPUTDIR)/$(COVERPROFILE) -o $(OUTPUTDIR)/$(COVERHTML)
	@echo "🌐 Run 'make browser/cover' to open coverage report in browser"
else
	@echo "CI detected, skipping HTML coverage report generation"
endif

.PHONY: browser/cover
browser/cover: ## Open coverage report in browser
	$(call open_browser,$(OUTPUTDIR)/$(COVERHTML))

.PHONY: clean
clean: ## Clean up build artifacts
	rm -rfv $(OUTPUTDIR) || true

## Help:
.PHONY: help
help: GREEN  := $(shell tput -Txterm setaf 2)
help: YELLOW := $(shell tput -Txterm setaf 3)
help: CYAN   := $(shell tput -Txterm setaf 6)
help: RESET  := $(shell tput -Txterm sgr0)
help: ## Show this help
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z0-9_\/-]+:.*?##.*$$/) {printf "    ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)
