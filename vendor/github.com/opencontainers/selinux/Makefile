BUILDTAGS := selinux

check-gopath:
ifndef GOPATH
	$(error GOPATH is not set)
endif

.PHONY: test
test: check-gopath
	go test -timeout 3m -tags "${BUILDTAGS}" ${TESTFLAGS} -v ./...

.PHONY:
lint:
	@out="$$(golint go-selinux)"; \
	if [ -n "$$out" ]; then \
		echo "$$out"; \
		exit 1; \
	fi
