LINTIGNOREDOT='awstesting/integration.+should not use dot imports'
LINTIGNOREDOC='service/[^/]+/(api|service|waiters)\.go:.+(comment on exported|should have comment or be unexported)'
LINTIGNORECONST='service/[^/]+/(api|service|waiters)\.go:.+(type|struct field|const|func) ([^ ]+) should be ([^ ]+)'
LINTIGNORESTUTTER='service/[^/]+/(api|service)\.go:.+(and that stutters)'
LINTIGNOREINFLECT='service/[^/]+/(api|service)\.go:.+method .+ should be '
LINTIGNOREDEPS='vendor/.+\.go'

SDK_WITH_VENDOR_PKGS=$(shell go list ./... | grep -v "/vendor/src")
SDK_ONLY_PKGS=$(shell go list ./... | grep -v "/vendor/")

all: generate unit

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  api_info                to print a list of services and versions"
	@echo "  docs                    to build SDK documentation"
	@echo "  build                   to go build the SDK"
	@echo "  unit                    to run unit tests"
	@echo "  integration             to run integration tests"
	@echo "  verify                  to verify tests"
	@echo "  lint                    to lint the SDK"
	@echo "  vet                     to vet the SDK"
	@echo "  generate                to go generate and make services"
	@echo "  gen-test                to generate protocol tests"
	@echo "  gen-services            to generate services"
	@echo "  get-deps                to go get the SDK dependencies"
	@echo "  get-deps-unit           to get the SDK's unit test dependencies"
	@echo "  get-deps-integ          to get the SDK's integration test dependencies"
	@echo "  get-deps-verify         to get the SDK's verification dependencies"

generate: gen-test gen-endpoints gen-services

gen-test: gen-protocol-test

gen-services:
	go generate ./service

gen-protocol-test:
	go generate ./private/protocol/...

gen-endpoints:
	go generate ./private/endpoints

build:
	@echo "go build SDK and vendor packages"
	@go build $(SDK_WITH_VENDOR_PKGS)

unit: get-deps-unit build verify
	@echo "go test SDK and vendor packages"
	@go test $(SDK_WITH_VENDOR_PKGS)

integration: get-deps-integ
	go test -tags=integration ./awstesting/integration/customizations/...
	gucumber ./awstesting/integration/smoke

verify: get-deps-verify lint vet

lint:
	@echo "go lint SDK and vendor packages"
	@lint=`golint ./...`; \
	lint=`echo "$$lint" | grep -E -v -e ${LINTIGNOREDOT} -e ${LINTIGNOREDOC} -e ${LINTIGNORECONST} -e ${LINTIGNORESTUTTER} -e ${LINTIGNOREINFLECT} -e ${LINTIGNOREDEPS}`; \
	echo "$$lint"; \
	if [ "$$lint" != "" ]; then exit 1; fi

vet:
	go tool vet -all -shadow $(shell ls -d */ | grep -v vendor)

get-deps: get-deps-unit get-deps-integ get-deps-verify
	@echo "go get SDK dependencies"
	@go get -v $(SDK_ONLY_PKGS)

get-deps-unit:
	@echo "go get SDK unit testing dependancies"
	go get github.com/stretchr/testify
	go get github.com/smartystreets/goconvey

get-deps-integ: get-deps-unit
	@echo "go get SDK integration testing dependencies"
	go get github.com/lsegal/gucumber/cmd/gucumber

get-deps-verify:
	@echo "go get SDK verification utilities"
	go get github.com/golang/lint/golint

bench:
	@echo "go bench SDK packages"
	@go test -run NONE -bench . -benchmem -tags 'bench' $(SDK_ONLY_PKGS)

bench-protocol:
	@echo "go bench SDK protocol marshallers"
	@go test -run NONE -bench . -benchmem -tags 'bench' ./private/protocol/...

docs:
	@echo "generate SDK docs"
	rm -rf doc && bundle install && bundle exec yard

api_info:
	@go run private/model/cli/api-info/api-info.go
