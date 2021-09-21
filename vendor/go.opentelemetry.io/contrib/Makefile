TOOLS_MOD_DIR := ./tools

# All source code and documents. Used in spell check.
ALL_DOCS := $(shell find . -name '*.md' -type f | sort)
# All directories with go.mod files related to opentelemetry library. Used for building, testing and linting.
ALL_GO_MOD_DIRS := $(filter-out $(TOOLS_MOD_DIR), $(shell find . -type f -name 'go.mod' -exec dirname {} \; | sort))
ALL_COVERAGE_MOD_DIRS := $(shell find . -type f -name 'go.mod' -exec dirname {} \; | egrep -v '^./example|^$(TOOLS_MOD_DIR)' | sort)

# URLs to check if all contrib entries exist in the registry.
REGISTRY_BASE_URL = https://raw.githubusercontent.com/open-telemetry/opentelemetry.io/main/content/en/registry
CONTRIB_REPO_URL = https://github.com/open-telemetry/opentelemetry-go-contrib/tree/main

# Mac OS Catalina 10.5.x doesn't support 386. Hence skip 386 test
SKIP_386_TEST = false
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SW_VERS := $(shell sw_vers -productVersion)
	ifeq ($(shell echo $(SW_VERS) | egrep '^(10.1[5-9]|1[1-9]|[2-9])'), $(SW_VERS))
		SKIP_386_TEST = true
	endif
endif


GOTEST_MIN = go test -v -timeout 30s
GOTEST = $(GOTEST_MIN) -race
GOTEST_WITH_COVERAGE = $(GOTEST) -coverprofile=coverage.out -covermode=atomic -coverpkg=./...

.DEFAULT_GOAL := precommit

.PHONY: precommit

TOOLS_DIR := $(abspath ./.tools)

$(TOOLS_DIR)/golangci-lint: $(TOOLS_MOD_DIR)/go.mod $(TOOLS_MOD_DIR)/go.sum $(TOOLS_MOD_DIR)/tools.go
	cd $(TOOLS_MOD_DIR) && \
	go build -o $(TOOLS_DIR)/golangci-lint github.com/golangci/golangci-lint/cmd/golangci-lint

$(TOOLS_DIR)/misspell: $(TOOLS_MOD_DIR)/go.mod $(TOOLS_MOD_DIR)/go.sum $(TOOLS_MOD_DIR)/tools.go
	cd $(TOOLS_MOD_DIR) && \
	go build -o $(TOOLS_DIR)/misspell github.com/client9/misspell/cmd/misspell

$(TOOLS_DIR)/stringer: $(TOOLS_MOD_DIR)/go.mod $(TOOLS_MOD_DIR)/go.sum $(TOOLS_MOD_DIR)/tools.go
	cd $(TOOLS_MOD_DIR) && \
	go build -o $(TOOLS_DIR)/stringer golang.org/x/tools/cmd/stringer

precommit: dependabot-check license-check generate build lint test

.PHONY: test-with-coverage
test-with-coverage:
	set -e; \
	printf "" > coverage.txt; \
	for dir in $(ALL_COVERAGE_MOD_DIRS); do \
	  echo "go test ./... + coverage in $${dir}"; \
	  (cd "$${dir}" && \
	    $(GOTEST_WITH_COVERAGE) ./... && \
	    go tool cover -html=coverage.out -o coverage.html); \
	  [ -f "$${dir}/coverage.out" ] && cat "$${dir}/coverage.out" >> coverage.txt; \
	done; \
	sed -i.bak -e '2,$$ { /^mode: /d; }' coverage.txt && rm coverage.txt.bak

.PHONY: ci
ci: precommit check-clean-work-tree test-with-coverage test-386

.PHONY: test-gocql
test-gocql:
	@if ./tools/should_build.sh gocql; then \
	  set -e; \
	  docker run --name cass-integ --rm -p 9042:9042 -d cassandra:3; \
	  CMD=cassandra IMG_NAME=cass-integ ./tools/wait.sh; \
	  (cd instrumentation/github.com/gocql/gocql/otelgocql && \
	    $(GOTEST_WITH_COVERAGE) . && \
	    go tool cover -html=coverage.out -o coverage.html); \
	  docker stop cass-integ; \
	fi

.PHONY: test-mongo-driver
test-mongo-driver:
	@if ./tools/should_build.sh mongo-driver; then \
	  set -e; \
	  docker run --name mongo-integ --rm -p 27017:27017 -d mongo; \
	  CMD=mongo IMG_NAME=mongo-integ ./tools/wait.sh; \
	  (cd instrumentation/go.mongodb.org/mongo-driver/mongo/otelmongo && \
	    $(GOTEST_WITH_COVERAGE) . && \
	    go tool cover -html=coverage.out -o coverage.html); \
	  docker stop mongo-integ; \
	fi

.PHONY: test-gomemcache
test-gomemcache:
	@if ./tools/should_build.sh gomemcache; then \
	  set -e; \
	  docker run --name gomemcache-integ --rm -p 11211:11211 -d memcached; \
	  CMD=gomemcache IMG_NAME=gomemcache-integ  ./tools/wait.sh; \
	  (cd instrumentation/github.com/bradfitz/gomemcache/memcache/otelmemcache && \
	    $(GOTEST_WITH_COVERAGE) . && \
	    go tool cover -html=coverage.out -o coverage.html); \
	  docker stop gomemcache-integ ; \
	fi

.PHONY: check-clean-work-tree
check-clean-work-tree:
	@if ! git diff --quiet; then \
	  echo; \
	  echo 'Working tree is not clean, did you forget to run "make precommit"?'; \
	  echo; \
	  git status; \
	  exit 1; \
	fi

.PHONY: build
build:
	# TODO: Fix this on windows.
	set -e; for dir in $(ALL_GO_MOD_DIRS); do \
	  echo "compiling all packages in $${dir}"; \
	  (cd "$${dir}" && \
	    go build ./... && \
	    go test -run xxxxxMatchNothingxxxxx ./... >/dev/null); \
	done

.PHONY: test
test:
	set -e; for dir in $(ALL_GO_MOD_DIRS); do \
	  echo "go test ./... + race in $${dir}"; \
	  (cd "$${dir}" && \
	    $(GOTEST) ./...); \
	done

.PHONY: test-386
test-386:
	if [ $(SKIP_386_TEST) = true ] ; then \
		echo "skipping the test for GOARCH 386 as it is not supported on the current OS"; \
  else \
    set -e; for dir in $(ALL_GO_MOD_DIRS); do \
        echo "go test ./... GOARCH 386 in $${dir}"; \
        (cd "$${dir}" && \
          GOARCH=386 $(GOTEST_MIN) ./...); \
	  done; \
	fi

.PHONY: lint
lint: $(TOOLS_DIR)/golangci-lint $(TOOLS_DIR)/misspell
	set -e; for dir in $(ALL_GO_MOD_DIRS); do \
	  echo "golangci-lint in $${dir}"; \
	  (cd "$${dir}" && \
	    $(TOOLS_DIR)/golangci-lint run --fix && \
	    $(TOOLS_DIR)/golangci-lint run); \
	done
	$(TOOLS_DIR)/misspell -w $(ALL_DOCS)
	set -e; for dir in $(ALL_GO_MOD_DIRS) $(TOOLS_MOD_DIR); do \
	  echo "go mod tidy in $${dir}"; \
	  (cd "$${dir}" && \
	    go mod tidy); \
	done

.PHONY: generate
generate: $(TOOLS_DIR)/stringer
	PATH="$(TOOLS_DIR):$${PATH}" go generate ./...

.PHONY: license-check
license-check:
	@licRes=$$(for f in $$(find . -type f \( -iname '*.go' -o -iname '*.sh' \) ! -path './vendor/*' ! -path './exporters/otlp/internal/opentelemetry-proto/*') ; do \
	           awk '/Copyright The OpenTelemetry Authors|generated|GENERATED/ && NR<=3 { found=1; next } END { if (!found) print FILENAME }' $$f; \
	   done); \
	   if [ -n "$${licRes}" ]; then \
	           echo "license header checking failed:"; echo "$${licRes}"; \
	           exit 1; \
	   fi

.PHONY: registry-links-check
registry-links-check:
	@checkRes=$$( \
		for f in $$( find ./instrumentation ./exporters ./detectors ! -path './instrumentation/net/*' -type f -name 'go.mod' -exec dirname {} \; | egrep -v '/example|/utils' | sort ) \
			./instrumentation/net/http; do \
			TYPE="instrumentation"; \
			if $$(echo "$$f" | grep -q "exporters"); then \
				TYPE="exporter"; \
			fi; \
			if $$(echo "$$f" | grep -q "detectors"); then \
				TYPE="detector"; \
			fi; \
			NAME=$$(echo "$$f" | sed -e 's/.*\///' -e 's/.*otel//'); \
			LINK=$(CONTRIB_REPO_URL)/$$(echo "$$f" | sed -e 's/..//' -e 's/\/otel.*$$//'); \
			if ! $$(curl -s $(REGISTRY_BASE_URL)/$${TYPE}-go-$${NAME}.md | grep -q "$${LINK}"); then \
				echo "$$f"; \
			fi \
		done; \
	); \
	if [ -n "$$checkRes" ]; then \
		echo "WARNING: registry link check failed for the following packages:"; echo "$${checkRes}"; \
	fi

.PHONY: dependabot-check
dependabot-check:
	@result=$$( \
		for f in $$( find . -type f -name go.mod -exec dirname {} \; | sed 's/^.\/\?/\//' ); \
			do grep -q "$$f" .github/dependabot.yml \
			|| echo "$$f"; \
		done; \
	); \
	if [ -n "$$result" ]; then \
		echo "missing go.mod dependabot check:"; echo "$$result"; \
		exit 1; \
	fi
