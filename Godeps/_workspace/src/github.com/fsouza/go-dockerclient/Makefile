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

SRCS = $(shell git ls-files '*.go' | grep -v '^external/')
PKGS = ./. ./testing

all: test

vendor:
	@ go get -v github.com/mjibson/party
	party -d external -c -u

lint:
	@ go get -v github.com/golang/lint/golint
	$(foreach file,$(SRCS),golint $(file) || exit;)

vet:
	@-go get -v golang.org/x/tools/cmd/vet
	$(foreach pkg,$(PKGS),go vet $(pkg);)

fmt:
	gofmt -w $(SRCS)

fmtcheck:
	$(foreach file,$(SRCS),gofmt -d $(file);)

prepare_docker:
	sudo stop docker
	sudo rm -rf /var/lib/docker
	sudo rm -f `which docker`
	sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
	echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
	sudo apt-get update
	sudo apt-get install docker-engine=$(DOCKER_VERSION)-0~$(shell lsb_release -cs) -y --force-yes -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

pretest: lint vet fmtcheck

test: pretest
	$(foreach pkg,$(PKGS),go test $(pkg) || exit;)

integration:
	go test -tags docker_integration -run TestIntegration -v

cov:
	@ go get -v github.com/axw/gocov/gocov
	@ go get golang.org/x/tools/cmd/cover
	gocov test | gocov report

clean:
	$(foreach pkg,$(PKGS),go clean $(pkg) || exit;)
