# Many Go tools take file globs or directories as arguments instead of packages.
PACKAGE_FILES ?= *.go

# For pre go1.6
export GO15VENDOREXPERIMENT=1


.PHONY: build
build:
	go build -i ./...


.PHONY: install
install:
	glide --version || go get github.com/Masterminds/glide
	glide install


.PHONY: test
test:
	go test -cover -race ./...


.PHONY: install_ci
install_ci: install
	go get github.com/wadey/gocovmerge
	go get github.com/mattn/goveralls
	go get golang.org/x/tools/cmd/cover

.PHONY: install_lint
install_lint:
	go get golang.org/x/lint/golint


.PHONY: lint
lint:
	@rm -rf lint.log
	@echo "Checking formatting..."
	@gofmt -d -s $(PACKAGE_FILES) 2>&1 | tee lint.log
	@echo "Checking vet..."
	@go vet ./... 2>&1 | tee -a lint.log;)
	@echo "Checking lint..."
	@golint $$(go list ./...) 2>&1 | tee -a lint.log
	@echo "Checking for unresolved FIXMEs..."
	@git grep -i fixme | grep -v -e vendor -e Makefile | tee -a lint.log
	@[ ! -s lint.log ]


.PHONY: test_ci
test_ci: install_ci build
	./scripts/cover.sh $(shell go list $(PACKAGES))
