.PHONY: all binary dynbinary build cross deb help init-go-pkg-cache install manpages rpm run shell test test-docker-py test-integration-cli test-unit tgz validate win

# set the graph driver as the current graphdriver if not set
DOCKER_GRAPHDRIVER := $(if $(DOCKER_GRAPHDRIVER),$(DOCKER_GRAPHDRIVER),$(shell docker info 2>&1 | grep "Storage Driver" | sed 's/.*: //'))
export DOCKER_GRAPHDRIVER
DOCKER_INCREMENTAL_BINARY := $(if $(DOCKER_INCREMENTAL_BINARY),$(DOCKER_INCREMENTAL_BINARY),1)
export DOCKER_INCREMENTAL_BINARY

# get OS/Arch of docker engine
DOCKER_OSARCH := $(shell bash -c 'source hack/make/.detect-daemon-osarch && echo $${DOCKER_ENGINE_OSARCH}')
DOCKERFILE := $(shell bash -c 'source hack/make/.detect-daemon-osarch && echo $${DOCKERFILE}')

DOCKER_GITCOMMIT := $(shell git rev-parse --short HEAD || echo unsupported)
export DOCKER_GITCOMMIT

# env vars passed through directly to Docker's build scripts
# to allow things like `make KEEPBUNDLE=1 binary` easily
# `project/PACKAGERS.md` have some limited documentation of some of these
DOCKER_ENVS := \
	-e DOCKER_CROSSPLATFORMS \
	-e BUILD_APT_MIRROR \
	-e BUILDFLAGS \
	-e KEEPBUNDLE \
	-e DOCKER_BUILD_ARGS \
	-e DOCKER_BUILD_GOGC \
	-e DOCKER_BUILD_PKGS \
	-e DOCKER_BASH_COMPLETION_PATH \
	-e DOCKER_CLI_PATH \
	-e DOCKER_DEBUG \
	-e DOCKER_EXPERIMENTAL \
	-e DOCKER_GITCOMMIT \
	-e DOCKER_GRAPHDRIVER \
	-e DOCKER_INCREMENTAL_BINARY \
	-e DOCKER_PORT \
	-e DOCKER_REMAP_ROOT \
	-e DOCKER_STORAGE_OPTS \
	-e DOCKER_USERLANDPROXY \
	-e TESTDIRS \
	-e TESTFLAGS \
	-e TIMEOUT \
	-e HTTP_PROXY \
	-e HTTPS_PROXY \
	-e NO_PROXY \
	-e http_proxy \
	-e https_proxy \
	-e no_proxy
# note: we _cannot_ add "-e DOCKER_BUILDTAGS" here because even if it's unset in the shell, that would shadow the "ENV DOCKER_BUILDTAGS" set in our Dockerfile, which is very important for our official builds

# to allow `make BIND_DIR=. shell` or `make BIND_DIR= test`
# (default to no bind mount if DOCKER_HOST is set)
# note: BINDDIR is supported for backwards-compatibility here
BIND_DIR := $(if $(BINDDIR),$(BINDDIR),$(if $(DOCKER_HOST),,bundles))
DOCKER_MOUNT := $(if $(BIND_DIR),-v "$(CURDIR)/$(BIND_DIR):/go/src/github.com/docker/docker/$(BIND_DIR)")

# This allows the test suite to be able to run without worrying about the underlying fs used by the container running the daemon (e.g. aufs-on-aufs), so long as the host running the container is running a supported fs.
# The volume will be cleaned up when the container is removed due to `--rm`.
# Note that `BIND_DIR` will already be set to `bundles` if `DOCKER_HOST` is not set (see above BIND_DIR line), in such case this will do nothing since `DOCKER_MOUNT` will already be set.
DOCKER_MOUNT := $(if $(DOCKER_MOUNT),$(DOCKER_MOUNT),-v /go/src/github.com/docker/docker/bundles) -v $(CURDIR)/.git:/go/src/github.com/docker/docker/.git

# This allows to set the docker-dev container name
DOCKER_CONTAINER_NAME := $(if $(CONTAINER_NAME),--name $(CONTAINER_NAME),)

# enable package cache if DOCKER_INCREMENTAL_BINARY and DOCKER_MOUNT (i.e.DOCKER_HOST) are set
PKGCACHE_MAP := gopath:/go/pkg goroot-linux_amd64:/usr/local/go/pkg/linux_amd64 goroot-linux_amd64_netgo:/usr/local/go/pkg/linux_amd64_netgo
PKGCACHE_VOLROOT := dockerdev-go-pkg-cache
PKGCACHE_VOL := $(if $(PKGCACHE_DIR),$(CURDIR)/$(PKGCACHE_DIR)/,$(PKGCACHE_VOLROOT)-)
DOCKER_MOUNT_PKGCACHE := $(if $(DOCKER_INCREMENTAL_BINARY),$(shell echo $(PKGCACHE_MAP) | sed -E 's@([^ ]*)@-v "$(PKGCACHE_VOL)\1"@g'),)
DOCKER_MOUNT_CLI := $(if $(DOCKER_CLI_PATH),-v $(shell dirname $(DOCKER_CLI_PATH)):/usr/local/cli,)
DOCKER_MOUNT_BASH_COMPLETION := $(if $(DOCKER_BASH_COMPLETION_PATH),-v $(shell dirname $(DOCKER_BASH_COMPLETION_PATH)):/usr/local/completion/bash,)
DOCKER_MOUNT := $(DOCKER_MOUNT) $(DOCKER_MOUNT_PKGCACHE) $(DOCKER_MOUNT_CLI) $(DOCKER_MOUNT_BASH_COMPLETION)

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
GIT_BRANCH_CLEAN := $(shell echo $(GIT_BRANCH) | sed -e "s/[^[:alnum:]]/-/g")
DOCKER_IMAGE := docker-dev$(if $(GIT_BRANCH_CLEAN),:$(GIT_BRANCH_CLEAN))
DOCKER_PORT_FORWARD := $(if $(DOCKER_PORT),-p "$(DOCKER_PORT)",)

DOCKER_FLAGS := docker run --rm -i --privileged $(DOCKER_CONTAINER_NAME) $(DOCKER_ENVS) $(DOCKER_MOUNT) $(DOCKER_PORT_FORWARD)
BUILD_APT_MIRROR := $(if $(DOCKER_BUILD_APT_MIRROR),--build-arg APT_MIRROR=$(DOCKER_BUILD_APT_MIRROR))
export BUILD_APT_MIRROR

SWAGGER_DOCS_PORT ?= 9000

INTEGRATION_CLI_MASTER_IMAGE := $(if $(INTEGRATION_CLI_MASTER_IMAGE), $(INTEGRATION_CLI_MASTER_IMAGE), integration-cli-master)
INTEGRATION_CLI_WORKER_IMAGE := $(if $(INTEGRATION_CLI_WORKER_IMAGE), $(INTEGRATION_CLI_WORKER_IMAGE), integration-cli-worker)

define \n


endef

# if this session isn't interactive, then we don't want to allocate a
# TTY, which would fail, but if it is interactive, we do want to attach
# so that the user can send e.g. ^C through.
INTERACTIVE := $(shell [ -t 0 ] && echo 1 || echo 0)
ifeq ($(INTERACTIVE), 1)
	DOCKER_FLAGS += -t
endif

DOCKER_RUN_DOCKER := $(DOCKER_FLAGS) "$(DOCKER_IMAGE)"

default: binary

all: build ## validate all checks, build linux binaries, run all tests\ncross build non-linux binaries and generate archives
	$(DOCKER_RUN_DOCKER) bash -c 'hack/validate/default && hack/make.sh'

binary: build ## build the linux binaries
	$(DOCKER_RUN_DOCKER) hack/make.sh binary

dynbinary: build ## build the linux dynbinaries
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary

build: bundles init-go-pkg-cache
	$(warning The docker client CLI has moved to github.com/docker/cli. By default, it is built from the git sha specified in hack/dockerfile/binaries-commits. For a dev-test cycle involving the CLI, run:${\n} DOCKER_CLI_PATH=/host/path/to/cli/binary make shell ${\n} then change the cli and compile into a binary at the same location.${\n})
	docker build ${BUILD_APT_MIRROR} ${DOCKER_BUILD_ARGS} -t "$(DOCKER_IMAGE)" -f "$(DOCKERFILE)" .

bundles:
	mkdir bundles

clean: clean-pkg-cache-vol ## clean up cached resources

clean-pkg-cache-vol:
	@- $(foreach mapping,$(PKGCACHE_MAP), \
		$(shell docker volume rm $(PKGCACHE_VOLROOT)-$(shell echo $(mapping) | awk -F':/' '{ print $$1 }') > /dev/null 2>&1) \
	)

cross: build ## cross build the binaries for darwin, freebsd and\nwindows
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary binary cross

deb: build  ## build the deb packages
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary build-deb


help: ## this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%22c"," "), $$2);printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init-go-pkg-cache:
	$(if $(PKGCACHE_DIR), mkdir -p $(shell echo $(PKGCACHE_MAP) | sed -E 's@([^: ]*):[^ ]*@$(PKGCACHE_DIR)/\1@g'))

install: ## install the linux binaries
	KEEPBUNDLE=1 hack/make.sh install-binary

rpm: build ## build the rpm packages
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary build-rpm

run: build ## run the docker daemon in a container
	$(DOCKER_RUN_DOCKER) sh -c "KEEPBUNDLE=1 hack/make.sh install-binary run"

shell: build ## start a shell inside the build env
	$(DOCKER_RUN_DOCKER) bash

test: build ## run the unit, integration and docker-py tests
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary cross test-unit test-integration-cli test-docker-py

test-docker-py: build ## run the docker-py tests
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary test-docker-py

test-integration-cli: build ## run the integration tests
	$(DOCKER_RUN_DOCKER) hack/make.sh build-integration-test-binary dynbinary test-integration-cli

test-unit: build ## run the unit tests
	$(DOCKER_RUN_DOCKER) hack/make.sh test-unit

tgz: build ## build the archives (.zip on windows and .tgz\notherwise) containing the binaries
	$(DOCKER_RUN_DOCKER) hack/make.sh dynbinary binary cross tgz

validate: build ## validate DCO, Seccomp profile generation, gofmt,\n./pkg/ isolation, golint, tests, tomls, go vet and vendor
	$(DOCKER_RUN_DOCKER) hack/validate/all

win: build ## cross build the binary for windows
	$(DOCKER_RUN_DOCKER) hack/make.sh win

.PHONY: swagger-gen
swagger-gen:
	docker run --rm -v $(PWD):/go/src/github.com/docker/docker \
		-w /go/src/github.com/docker/docker \
		--entrypoint hack/generate-swagger-api.sh \
		-e GOPATH=/go \
		quay.io/goswagger/swagger:0.7.4

.PHONY: swagger-docs
swagger-docs: ## preview the API documentation
	@echo "API docs preview will be running at http://localhost:$(SWAGGER_DOCS_PORT)"
	@docker run --rm -v $(PWD)/api/swagger.yaml:/usr/share/nginx/html/swagger.yaml \
		-e 'REDOC_OPTIONS=hide-hostname="true" lazy-rendering' \
		-p $(SWAGGER_DOCS_PORT):80 \
		bfirsh/redoc:1.6.2

build-integration-cli-on-swarm: build ## build images and binary for running integration-cli on Swarm in parallel
	@echo "Building hack/integration-cli-on-swarm (if build fails, please refer to hack/integration-cli-on-swarm/README.md)"
	go build -o ./hack/integration-cli-on-swarm/integration-cli-on-swarm ./hack/integration-cli-on-swarm/host
	@echo "Building $(INTEGRATION_CLI_MASTER_IMAGE)"
	docker build -t $(INTEGRATION_CLI_MASTER_IMAGE) hack/integration-cli-on-swarm/agent
# For worker, we don't use `docker build` so as to enable DOCKER_INCREMENTAL_BINARY and so on
	@echo "Building $(INTEGRATION_CLI_WORKER_IMAGE) from $(DOCKER_IMAGE)"
	$(eval tmp := integration-cli-worker-tmp)
# We mount pkgcache, but not bundle (bundle needs to be baked into the image)
# For avoiding bakings DOCKER_GRAPHDRIVER and so on to image, we cannot use $(DOCKER_ENVS) here
	docker run -t -d --name $(tmp) -e DOCKER_GITCOMMIT -e BUILDFLAGS -e DOCKER_INCREMENTAL_BINARY --privileged $(DOCKER_MOUNT_PKGCACHE) $(DOCKER_IMAGE) top
	docker exec $(tmp) hack/make.sh build-integration-test-binary dynbinary
	docker exec $(tmp) go build -o /worker github.com/docker/docker/hack/integration-cli-on-swarm/agent/worker
	docker commit -c 'ENTRYPOINT ["/worker"]' $(tmp) $(INTEGRATION_CLI_WORKER_IMAGE)
	docker rm -f $(tmp)
