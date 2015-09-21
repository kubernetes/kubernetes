
all:
	docker build -t dockercore/libcontainer .

test: 
	# we need NET_ADMIN for the netlink tests and SYS_ADMIN for mounting
	docker run --rm -it --privileged dockercore/libcontainer

sh:
	docker run --rm -it --privileged -w /busybox dockercore/libcontainer nsinit exec sh

GO_PACKAGES = $(shell find . -not \( -wholename ./vendor -prune -o -wholename ./.git -prune \) -name '*.go' -print0 | xargs -0n1 dirname | sort -u)

direct-test:
	go test $(TEST_TAGS) -cover -v $(GO_PACKAGES)

direct-test-short:
	go test $(TEST_TAGS) -cover -test.short -v $(GO_PACKAGES)

direct-build:
	go build -v $(GO_PACKAGES)

direct-install:
	go install -v $(GO_PACKAGES)

local:
	go test -v

validate:
	hack/validate.sh

binary: all
	docker run --rm --privileged -v $(CURDIR)/bundles:/go/bin dockercore/libcontainer make direct-install
