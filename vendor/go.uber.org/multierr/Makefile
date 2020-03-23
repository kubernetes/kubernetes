export GO15VENDOREXPERIMENT=1

PACKAGES := $(shell glide nv)

GO_FILES := $(shell \
	find . '(' -path '*/.*' -o -path './vendor' ')' -prune \
	-o -name '*.go' -print | cut -b3-)

.PHONY: install
install:
	glide --version || go get github.com/Masterminds/glide
	glide install

.PHONY: build
build:
	go build -i $(PACKAGES)

.PHONY: test
test:
	go test -cover -race $(PACKAGES)

.PHONY: gofmt
gofmt:
	$(eval FMT_LOG := $(shell mktemp -t gofmt.XXXXX))
	@gofmt -e -s -l $(GO_FILES) > $(FMT_LOG) || true
	@[ ! -s "$(FMT_LOG)" ] || (echo "gofmt failed:" | cat - $(FMT_LOG) && false)

.PHONY: govet
govet:
	$(eval VET_LOG := $(shell mktemp -t govet.XXXXX))
	@go vet $(PACKAGES) 2>&1 \
		| grep -v '^exit status' > $(VET_LOG) || true
	@[ ! -s "$(VET_LOG)" ] || (echo "govet failed:" | cat - $(VET_LOG) && false)

.PHONY: golint
golint:
	@go get github.com/golang/lint/golint
	$(eval LINT_LOG := $(shell mktemp -t golint.XXXXX))
	@cat /dev/null > $(LINT_LOG)
	@$(foreach pkg, $(PACKAGES), golint $(pkg) >> $(LINT_LOG) || true;)
	@[ ! -s "$(LINT_LOG)" ] || (echo "golint failed:" | cat - $(LINT_LOG) && false)

.PHONY: staticcheck
staticcheck:
	@go get honnef.co/go/tools/cmd/staticcheck
	$(eval STATICCHECK_LOG := $(shell mktemp -t staticcheck.XXXXX))
	@staticcheck $(PACKAGES) 2>&1 > $(STATICCHECK_LOG) || true
	@[ ! -s "$(STATICCHECK_LOG)" ] || (echo "staticcheck failed:" | cat - $(STATICCHECK_LOG) && false)

.PHONY: lint
lint: gofmt govet golint staticcheck

.PHONY: cover
cover:
	./scripts/cover.sh $(shell go list $(PACKAGES))
	go tool cover -html=cover.out -o cover.html

update-license:
	@go get go.uber.org/tools/update-license
	@update-license \
		$(shell go list -json $(PACKAGES) | \
			jq -r '.Dir + "/" + (.GoFiles | .[])')

##############################################################################

.PHONY: install_ci
install_ci: install
	go get github.com/wadey/gocovmerge
	go get github.com/mattn/goveralls
	go get golang.org/x/tools/cmd/cover

.PHONY: test_ci
test_ci: install_ci
	./scripts/cover.sh $(shell go list $(PACKAGES))
