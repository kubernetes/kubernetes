BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse --short HEAD`
GOLDFLAGS="-X main.branch $(BRANCH) -X main.commit $(COMMIT)"

TESTFLAGS_RACE=-race=false
ifdef ENABLE_RACE
	TESTFLAGS_RACE=-race=true
endif

TESTFLAGS_CPU=
ifdef CPU
	TESTFLAGS_CPU=-cpu=$(CPU)
endif
TESTFLAGS = $(TESTFLAGS_RACE) $(TESTFLAGS_CPU) $(EXTRA_TESTFLAGS)

.PHONY: fmt
fmt:
	!(gofmt -l -s -d $(shell find . -name \*.go) | grep '[a-z]')

.PHONY: lint
lint:
	golangci-lint run ./...

.PHONY: test
test:
	@echo "hashmap freelist test"
	TEST_FREELIST_TYPE=hashmap go test -v ${TESTFLAGS} -timeout 30m
	TEST_FREELIST_TYPE=hashmap go test -v ${TESTFLAGS} ./cmd/bbolt

	@echo "array freelist test"
	TEST_FREELIST_TYPE=array go test -v ${TESTFLAGS} -timeout 30m
	TEST_FREELIST_TYPE=array go test -v ${TESTFLAGS} ./cmd/bbolt

.PHONY: coverage
coverage:
	@echo "hashmap freelist test"
	TEST_FREELIST_TYPE=hashmap go test -v -timeout 30m \
		-coverprofile cover-freelist-hashmap.out -covermode atomic

	@echo "array freelist test"
	TEST_FREELIST_TYPE=array go test -v -timeout 30m \
		-coverprofile cover-freelist-array.out -covermode atomic

.PHONY: gofail-enable
gofail-enable: install-gofail
	gofail enable .

.PHONY: gofail-disable
gofail-disable:
	gofail disable .

.PHONY: install-gofail
install-gofail:
	go install go.etcd.io/gofail

.PHONY: test-failpoint
test-failpoint:
	@echo "[failpoint] hashmap freelist test"
	TEST_FREELIST_TYPE=hashmap go test -v ${TESTFLAGS} -timeout 30m ./tests/failpoint

	@echo "[failpoint] array freelist test"
	TEST_FREELIST_TYPE=array go test -v ${TESTFLAGS} -timeout 30m ./tests/failpoint

