PACKAGES := . ./mocktracer/... ./ext/...

.DEFAULT_GOAL := test-and-lint

.PHONE: test-and-lint

test-and-lint: test lint

.PHONY: test
test:
	go test -v -cover ./...

cover:
	@rm -rf cover-all.out
	$(foreach pkg, $(PACKAGES), $(MAKE) cover-pkg PKG=$(pkg) || true;)
	@grep mode: cover.out > coverage.out
	@cat cover-all.out >> coverage.out
	go tool cover -html=coverage.out -o cover.html
	@rm -rf cover.out cover-all.out coverage.out

cover-pkg:
	go test -coverprofile cover.out $(PKG)
	@grep -v mode: cover.out >> cover-all.out

.PHONY: lint
lint:
	go fmt ./...
	golint ./...
	@# Run again with magic to exit non-zero if golint outputs anything.
	@! (golint ./... | read dummy)
	go vet ./...

