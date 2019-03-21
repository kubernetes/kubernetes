# Define some VCS context
PARENT_BRANCH ?= master

# Set the mode for code-coverage
GO_TEST_COVERAGE_MODE ?= count
GO_TEST_COVERAGE_FILE_NAME ?= coverage.out

# Set flags for `gofmt`
GOFMT_FLAGS ?= -s

# Set a default `min_confidence` value for `golint`
GOLINT_MIN_CONFIDENCE ?= 0.3


all: install-deps build install

clean:
	go clean -i -x ./...

build:
	go build -v ./...

install:
	go install ./...

install-deps:
	go get -d -t ./...

install-deps-dev: install-deps
	go get github.com/golang/lint/golint
	go get golang.org/x/tools/cmd/goimports

update-deps:
	go get -d -t -u ./...

update-deps-dev: update-deps
	go get -u github.com/golang/lint/golint
	go get -u golang.org/x/tools/cmd/goimports

test:
	go test -v ./...

test-with-coverage:
	go test -cover ./...

test-with-coverage-formatted:
	go test -cover ./... | column -t | sort -r

test-with-coverage-profile:
	echo "mode: ${GO_TEST_COVERAGE_MODE}" > ${GO_TEST_COVERAGE_FILE_NAME}
	for package in $$(go list ./...); do \
		go test -covermode ${GO_TEST_COVERAGE_MODE} -coverprofile "coverage_$${package##*/}.out" "$${package}"; \
		sed '1d' "coverage_$${package##*/}.out" >> ${GO_TEST_COVERAGE_FILE_NAME}; \
	done

format-lint:
	errors=$$(gofmt -l ${GOFMT_FLAGS} .); if [ "$${errors}" != "" ]; then echo "$${errors}"; exit 1; fi

import-lint:
	errors=$$(goimports -l .); if [ "$${errors}" != "" ]; then echo "$${errors}"; exit 1; fi

style-lint:
	errors=$$(golint -min_confidence=${GOLINT_MIN_CONFIDENCE} ./...); if [ "$${errors}" != "" ]; then echo "$${errors}"; exit 1; fi

copyright-lint:
	@old_dates=$$(git diff --diff-filter=ACMRTUXB --name-only "${PARENT_BRANCH}" | xargs grep -E '[Cc]opyright(\s+)[©Cc]?(\s+)[0-9]{4}' | grep -E -v "[Cc]opyright(\s+)[©Cc]?(\s+)$$(date '+%Y')"); if [ "$${old_dates}" != "" ]; then printf "The following files contain outdated copyrights:\n$${old_dates}\n\nThis can be fixed with 'make copyright-fix'\n"; exit 1; fi

lint: install-deps-dev format-lint import-lint style-lint copyright-lint

format-fix:
	gofmt -w ${GOFMT_FLAGS} .

import-fix:
	goimports -w .

copyright-fix:
	@git diff --diff-filter=ACMRTUXB --name-only "${PARENT_BRANCH}" | xargs -I '_FILENAME' -- sh -c 'sed -i.bak "s/\([Cc]opyright\([[:space:]][©Cc]\{0,1\}[[:space:]]*\)\)[0-9]\{4\}/\1"$$(date '+%Y')"/g" _FILENAME && rm _FILENAME.bak'

vet:
	go vet ./...


.PHONY: all clean build install install-deps install-deps-dev update-deps update-deps-dev test test-with-coverage test-with-coverage-formatted test-with-coverage-profile format-lint import-lint style-lint copyright-lint lint format-fix import-fix copyright-fix vet
