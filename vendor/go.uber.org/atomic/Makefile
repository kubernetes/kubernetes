PACKAGES := $(shell glide nv)
# Many Go tools take file globs or directories as arguments instead of packages.
PACKAGE_FILES ?= *.go


# The linting tools evolve with each Go version, so run them only on the latest
# stable release.
GO_VERSION := $(shell go version | cut -d " " -f 3)
GO_MINOR_VERSION := $(word 2,$(subst ., ,$(GO_VERSION)))
LINTABLE_MINOR_VERSIONS := 7 8
ifneq ($(filter $(LINTABLE_MINOR_VERSIONS),$(GO_MINOR_VERSION)),)
SHOULD_LINT := true
endif


export GO15VENDOREXPERIMENT=1


.PHONY: build
build:
	go build -i $(PACKAGES)


.PHONY: install
install:
	glide --version || go get github.com/Masterminds/glide
	glide install


.PHONY: test
test:
	go test -cover -race $(PACKAGES)


.PHONY: install_ci
install_ci: install
	go get github.com/wadey/gocovmerge
	go get github.com/mattn/goveralls
	go get golang.org/x/tools/cmd/cover
ifdef SHOULD_LINT
	go get github.com/golang/lint/golint
endif

.PHONY: lint
lint:
ifdef SHOULD_LINT
	@rm -rf lint.log
	@echo "Checking formatting..."
	@gofmt -d -s $(PACKAGE_FILES) 2>&1 | tee lint.log
	@echo "Checking vet..."
	@$(foreach dir,$(PACKAGE_FILES),go tool vet $(dir) 2>&1 | tee -a lint.log;)
	@echo "Checking lint..."
	@$(foreach dir,$(PKGS),golint $(dir) 2>&1 | tee -a lint.log;)
	@echo "Checking for unresolved FIXMEs..."
	@git grep -i fixme | grep -v -e vendor -e Makefile | tee -a lint.log
	@[ ! -s lint.log ]
else
	@echo "Skipping linters on" $(GO_VERSION)
endif


.PHONY: test_ci
test_ci: install_ci build
	./scripts/cover.sh $(shell go list $(PACKAGES))
