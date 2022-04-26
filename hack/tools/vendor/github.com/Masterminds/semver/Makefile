.PHONY: setup
setup:
	go get -u gopkg.in/alecthomas/gometalinter.v1
	gometalinter.v1 --install

.PHONY: test
test: validate lint
	@echo "==> Running tests"
	go test -v

.PHONY: validate
validate:
	@echo "==> Running static validations"
	@gometalinter.v1 \
	  --disable-all \
	  --enable deadcode \
	  --severity deadcode:error \
	  --enable gofmt \
	  --enable gosimple \
	  --enable ineffassign \
	  --enable misspell \
	  --enable vet \
	  --tests \
	  --vendor \
	  --deadline 60s \
	  ./... || exit_code=1

.PHONY: lint
lint:
	@echo "==> Running linters"
	@gometalinter.v1 \
	  --disable-all \
	  --enable golint \
	  --vendor \
	  --deadline 60s \
	  ./... || :
