.PHONY: all binary build cross default docs docs-build docs-shell shell test test-unit test-integration test-integration-cli test-docker-py validate

# env vars passed through directly to Docker's build scripts
# to allow things like `make DOCKER_CLIENTONLY=1 binary` easily
# `docs/sources/contributing/devenvironment.md ` and `project/PACKAGERS.md` have some limited documentation of some of these
DOCKER_ENVS := \
	-e BUILDFLAGS \
	-e DOCKER_CLIENTONLY \
	-e DOCKER_EXECDRIVER \
	-e DOCKER_GRAPHDRIVER \
	-e TESTDIRS \
	-e TESTFLAGS \
	-e TIMEOUT
# note: we _cannot_ add "-e DOCKER_BUILDTAGS" here because even if it's unset in the shell, that would shadow the "ENV DOCKER_BUILDTAGS" set in our Dockerfile, which is very important for our official builds

# to allow `make DOCSDIR=docs docs-shell` (to create a bind mount in docs)
DOCS_MOUNT := $(if $(DOCSDIR),-v $(CURDIR)/$(DOCSDIR):/$(DOCSDIR))

# to allow `make DOCSPORT=9000 docs`
DOCSPORT := 8000

# Get the IP ADDRESS
DOCKER_IP=$(shell python -c "import urlparse ; print urlparse.urlparse('$(DOCKER_HOST)').hostname or ''")
HUGO_BASE_URL=$(shell test -z "$(DOCKER_IP)" && echo localhost || echo "$(DOCKER_IP)")
HUGO_BIND_IP=0.0.0.0

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
DOCKER_IMAGE := docker$(if $(GIT_BRANCH),:$(GIT_BRANCH))
DOCKER_DOCS_IMAGE := docs-base$(if $(GIT_BRANCH),:$(GIT_BRANCH))


DOCKER_RUN_DOCS := docker run --rm -it $(DOCS_MOUNT) -e AWS_S3_BUCKET -e NOCACHE

# for some docs workarounds (see below in "docs-build" target)
GITCOMMIT := $(shell git rev-parse --short HEAD 2>/dev/null)

default: docs

docs: docs-build
	$(DOCKER_RUN_DOCS) -p $(if $(DOCSPORT),$(DOCSPORT):)8000 -e DOCKERHOST "$(DOCKER_DOCS_IMAGE)" hugo server --port=$(DOCSPORT) --baseUrl=$(HUGO_BASE_URL) --bind=$(HUGO_BIND_IP)

docs-draft: docs-build
	$(DOCKER_RUN_DOCS) -p $(if $(DOCSPORT),$(DOCSPORT):)8000 -e DOCKERHOST "$(DOCKER_DOCS_IMAGE)" hugo server --buildDrafts="true" --port=$(DOCSPORT) --baseUrl=$(HUGO_BASE_URL) --bind=$(HUGO_BIND_IP)


docs-shell: docs-build
	$(DOCKER_RUN_DOCS) -p $(if $(DOCSPORT),$(DOCSPORT):)8000 "$(DOCKER_DOCS_IMAGE)" bash


docs-build:
#	( git remote | grep -v upstream ) || git diff --name-status upstream/release..upstream/docs ./ > ./changed-files
#	echo "$(GIT_BRANCH)" > GIT_BRANCH
#	echo "$(AWS_S3_BUCKET)" > AWS_S3_BUCKET
#	echo "$(GITCOMMIT)" > GITCOMMIT
	docker build -t "$(DOCKER_DOCS_IMAGE)" .
