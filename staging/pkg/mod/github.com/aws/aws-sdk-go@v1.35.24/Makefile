LINTIGNOREDOC='service/[^/]+/(api|service|waiters)\.go:.+(comment on exported|should have comment or be unexported)'
LINTIGNORECONST='service/[^/]+/(api|service|waiters)\.go:.+(type|struct field|const|func) ([^ ]+) should be ([^ ]+)'
LINTIGNORESTUTTER='service/[^/]+/(api|service)\.go:.+(and that stutters)'
LINTIGNOREINFLECT='service/[^/]+/(api|errors|service)\.go:.+(method|const) .+ should be '
LINTIGNOREINFLECTS3UPLOAD='service/s3/s3manager/upload\.go:.+struct field SSEKMSKeyId should be '
LINTIGNOREENDPOINTS='aws/endpoints/(defaults|dep_service_ids).go:.+(method|const) .+ should be '
LINTIGNOREDEPS='vendor/.+\.go'
LINTIGNOREPKGCOMMENT='service/[^/]+/doc_custom.go:.+package comment should be of the form'
LINTIGNORESINGLEFIGHT='internal/sync/singleflight/singleflight.go:.+error should be the last type'
UNIT_TEST_TAGS="example codegen awsinclude"
ALL_TAGS="example codegen awsinclude integration perftest"

# SDK's Core and client packages that are compatable with Go 1.5+.
SDK_CORE_PKGS=./aws/... ./private/... ./internal/...
SDK_CLIENT_PKGS=./service/...
SDK_COMPA_PKGS=${SDK_CORE_PKGS} ${SDK_CLIENT_PKGS}

# SDK additional packages that are used for development of the SDK.
SDK_EXAMPLES_PKGS=./example/...
SDK_TESTING_PKGS=./awstesting/...
SDK_MODELS_PKGS=./models/...
SDK_ALL_PKGS=${SDK_COMPA_PKGS} ${SDK_TESTING_PKGS} ${SDK_EXAMPLES_PKGS} ${SDK_MODELS_PKGS}

TEST_TIMEOUT=-timeout 5m

all: generate unit

###################
# Code Generation #
###################
generate: cleanup-models gen-test gen-endpoints gen-services

gen-test: gen-protocol-test gen-codegen-test

gen-codegen-test:
	@echo "Generating SDK API tests"
	go generate ./private/model/api/codegentest/service

gen-services:
	@echo "Generating SDK clients"
	go generate ./service

gen-protocol-test:
	@echo "Generating SDK protocol tests"
	go generate ./private/protocol/...

gen-endpoints:
	@echo "Generating SDK endpoints"
	go generate ./models/endpoints

cleanup-models:
	@echo "Cleaning up stale model versions"
	go run -tags codegen ./private/model/cli/cleanup-models/* "./models/apis/*/*/api-2.json"

###################
# Unit/CI Testing #
###################

build:
	go build -o /dev/null -tags ${ALL_TAGS} ${SDK_ALL_PKGS}

unit-no-verify:
	@echo "go test SDK and vendor packages"
	go test ${TEST_TIMEOUT} -v -count=1 -tags ${UNIT_TEST_TAGS} ${SDK_ALL_PKGS}

unit: verify build unit-no-verify

unit-with-race-cover: verify build
	@echo "go test SDK and vendor packages"
	go test ${TEST_TIMEOUT} -v -count=1 -tags ${UNIT_TEST_TAGS} -race -cpu=1,2,4 ${SDK_ALL_PKGS}

unit-old-go-race-cover:
	@echo "go test SDK only packages for old Go versions"
	go test ${TEST_TIMEOUT} -v -count=1 -race -cpu=1,2,4 ${SDK_COMPA_PKGS}

ci-test: generate unit-with-race-cover ci-test-generate-validate

ci-test-generate-validate:
	@echo "CI test validate no generated code changes"
	git update-index --assume-unchanged go.mod go.sum
	git add . -A
	gitstatus=`git diff --cached --ignore-space-change`; \
	echo "$$gitstatus"; \
	if [ "$$gitstatus" != "" ] && [ "$$gitstatus" != "skipping validation" ]; then echo "$$gitstatus"; exit 1; fi
	git update-index --no-assume-unchanged go.mod go.sum

#######################
# Integration Testing #
#######################
integration: core-integ client-integ

core-integ:
	@echo "Integration Testing SDK core"
	AWS_REGION="" go test -count=1 -tags "integration" -v -run '^TestInteg_' ${SDK_CORE_PKGS} ./awstesting/...

client-integ:
	@echo "Integration Testing SDK clients"
	AWS_REGION="" go test -count=1 -tags "integration" -v -run '^TestInteg_' ./service/...

s3crypto-integ:
	@echo "Integration Testing S3 Cyrpto utility"
	AWS_REGION="" go test -count=1 -tags "s3crypto_integ integration" -v -run '^TestInteg_' ./service/s3/s3crypto

cleanup-integ-buckets:
	@echo "Cleaning up SDK integraiton resources"
	go run -tags "integration" ./awstesting/cmd/bucket_cleanup/main.go "aws-sdk-go-integration"

###################
# Sandbox Testing #
###################
sandbox-tests: sandbox-test-go1.5 sandbox-test-go1.6 sandbox-test-go1.7 sandbox-test-go1.8 sandbox-test-go1.9 sandbox-test-go1.10 sandbox-test-go1.11 sandbox-test-go1.12 sandbox-test-go1.13 sandbox-test-go1.14 sandbox-test-go1.15 sandbox-test-gotip

sandbox-build-go1.5:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.5 -t "aws-sdk-go-1.5" .
sandbox-go1.5: sandbox-build-go1.5
	docker run -i -t aws-sdk-go-1.5 bash
sandbox-test-go1.5: sandbox-build-go1.5
	docker run -t aws-sdk-go-1.5

sandbox-build-go1.6:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.6 -t "aws-sdk-go-1.6" .
sandbox-go1.6: sandbox-build-go1.6
	docker run -i -t aws-sdk-go-1.6 bash
sandbox-test-go1.6: sandbox-build-go1.6
	docker run -t aws-sdk-go-1.6

sandbox-build-go1.7:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.7 -t "aws-sdk-go-1.7" .
sandbox-go1.7: sandbox-build-go1.7
	docker run -i -t aws-sdk-go-1.7 bash
sandbox-test-go1.7: sandbox-build-go1.7
	docker run -t aws-sdk-go-1.7

sandbox-build-go1.8:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.8 -t "aws-sdk-go-1.8" .
sandbox-go1.8: sandbox-build-go1.8
	docker run -i -t aws-sdk-go-1.8 bash
sandbox-test-go1.8: sandbox-build-go1.8
	docker run -t aws-sdk-go-1.8

sandbox-build-go1.9:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.9 -t "aws-sdk-go-1.9" .
sandbox-go1.9: sandbox-build-go1.9
	docker run -i -t aws-sdk-go-1.9 bash
sandbox-test-go1.9: sandbox-build-go1.9
	docker run -t aws-sdk-go-1.9

sandbox-build-go1.10:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.10 -t "aws-sdk-go-1.10" .
sandbox-go1.10: sandbox-build-go1.10
	docker run -i -t aws-sdk-go-1.10 bash
sandbox-test-go1.10: sandbox-build-go1.10
	docker run -t aws-sdk-go-1.10

sandbox-build-go1.11:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.11 -t "aws-sdk-go-1.11" .
sandbox-go1.11: sandbox-build-go1.11
	docker run -i -t aws-sdk-go-1.11 bash
sandbox-test-go1.11: sandbox-build-go1.11
	docker run -t aws-sdk-go-1.11

sandbox-build-go1.12:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.12 -t "aws-sdk-go-1.12" .
sandbox-go1.12: sandbox-build-go1.12
	docker run -i -t aws-sdk-go-1.12 bash
sandbox-test-go1.12: sandbox-build-go1.12
	docker run -t aws-sdk-go-1.12

sandbox-build-go1.13:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.13 -t "aws-sdk-go-1.13" .
sandbox-go1.13: sandbox-build-go1.13
	docker run -i -t aws-sdk-go-1.13 bash
sandbox-test-go1.13: sandbox-build-go1.13
	docker run -t aws-sdk-go-1.13

sandbox-build-go1.14:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.14 -t "aws-sdk-go-1.14" .
sandbox-go1.14: sandbox-build-go1.14
	docker run -i -t aws-sdk-go-1.14 bash
sandbox-test-go1.14: sandbox-build-go1.14
	docker run -t aws-sdk-go-1.14

sandbox-build-go1.15:
	docker build -f ./awstesting/sandbox/Dockerfile.test.go1.15 -t "aws-sdk-go-1.15" .
sandbox-go1.15: sandbox-build-go1.15
	docker run -i -t aws-sdk-go-1.15 bash
sandbox-test-go1.15: sandbox-build-go1.15
	docker run -t aws-sdk-go-1.15

sandbox-build-gotip:
	@echo "Run make update-aws-golang-tip, if this test fails because missing aws-golang:tip container"
	docker build -f ./awstesting/sandbox/Dockerfile.test.gotip -t "aws-sdk-go-tip" .
sandbox-gotip: sandbox-build-gotip
	docker run -i -t aws-sdk-go-tip bash
sandbox-test-gotip: sandbox-build-gotip
	docker run -t aws-sdk-go-tip

update-aws-golang-tip:
	docker build --no-cache=true -f ./awstesting/sandbox/Dockerfile.golang-tip -t "aws-golang:tip" .

##################
# Linting/Verify #
##################
verify: lint vet

lint:
	@echo "go lint SDK and vendor packages"
	@lint=`golint ./...`; \
	dolint=`echo "$$lint" | grep -E -v -e ${LINTIGNOREDOC} -e ${LINTIGNORECONST} -e ${LINTIGNORESTUTTER} -e ${LINTIGNOREINFLECT} -e ${LINTIGNOREDEPS} -e ${LINTIGNOREINFLECTS3UPLOAD} -e ${LINTIGNOREPKGCOMMENT} -e ${LINTIGNOREENDPOINTS} -e ${LINTIGNORESINGLEFIGHT}`; \
	echo "$$dolint"; \
	if [ "$$dolint" != "" ]; then exit 1; fi

vet:
	go vet -tags "example codegen awsinclude integration" --all ${SDK_ALL_PKGS}

################
# Dependencies #
################
get-deps: 
	@echo "getting pre go module dependnecies"
	go get github.com/jmespath/go-jmespath

get-deps-x-tests:
	@echo "go get SDK testing golang.org/x dependencies"
	go get golang.org/x/net/http2

get-deps-verify:
	@echo "go get SDK verification utilities"
	go get golang.org/x/lint/golint

##############
# Benchmarks #
##############
bench:
	@echo "go bench SDK packages"
	go test -count=1 -run NONE -bench . -benchmem -tags 'bench' ${SDK_ALL_PKGS}

bench-protocol:
	@echo "go bench SDK protocol marshallers"
	go test -count=1 -run NONE -bench . -benchmem -tags 'bench' ./private/protocol/...

#############
# Utilities #
#############
docs:
	@echo "generate SDK docs"
	$(AWS_DOC_GEN_TOOL) `pwd`

api_info:
	@go run private/model/cli/api-info/api-info.go
