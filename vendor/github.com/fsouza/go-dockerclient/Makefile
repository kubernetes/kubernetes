.PHONY: \
	all \
	vendor \
	lint \
	vet \
	fmt \
	fmtcheck \
	pretest \
	test \
	integration \
	cov \
	clean

PKGS = . ./testing

all: test

vendor:
	@ go get -v github.com/mjibson/party
	party -d external -c -u

lint:
	@ go get -v github.com/golang/lint/golint
	@for file in $$(git ls-files '*.go' | grep -v 'external/'); do \
		export output="$$(golint $${file} | grep -v 'type name will be used as docker.DockerInfo')"; \
		[ -n "$${output}" ] && echo "$${output}" && export status=1; \
	done; \
	exit $${status:-0}

vet:
	$(foreach pkg,$(PKGS),go vet $(pkg);)

fmt:
	gofmt -s -w $(PKGS)

fmtcheck:
	@ export output=$$(gofmt -s -d $(PKGS)); \
		[ -n "$${output}" ] && echo "$${output}" && export status=1; \
		exit $${status:-0}

pretest: lint vet fmtcheck

gotest:
	$(foreach pkg,$(PKGS),go test $(pkg) || exit;)

test: pretest gotest

integration:
	go test -tags docker_integration -run TestIntegration -v

cov:
	@ go get -v github.com/axw/gocov/gocov
	@ go get golang.org/x/tools/cmd/cover
	gocov test | gocov report

clean:
	$(foreach pkg,$(PKGS),go clean $(pkg) || exit;)
