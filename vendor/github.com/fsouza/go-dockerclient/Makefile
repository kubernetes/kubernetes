.PHONY: \
	all \
	lint \
	vet \
	fmt \
	fmtcheck \
	pretest \
	test \
	integration \
	clean

all: test

lint:
	@ go get -v github.com/golang/lint/golint
	@ export output="$$(golint . | grep -v 'type name will be used as docker.DockerInfo')"; \
		[ -n "$${output}" ] && echo "$${output}" && export status=1; \
		exit $${status:-0}

vet:
	go vet ./...

fmt:
	gofmt -s -w .

fmtcheck:
	[ -z "$$(gofmt -s -d . | tee /dev/stderr)" ]

testdeps:
	go get -d -t ./...

pretest: testdeps lint vet fmtcheck

gotest:
	go test $(GO_TEST_FLAGS) ./...

test: pretest gotest

integration:
	go test -tags docker_integration -run TestIntegration -v

clean:
	go clean ./...
