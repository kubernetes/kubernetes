#
# Based on http://chrismckenzie.io/post/deploying-with-golang/
#

.PHONY: version all run dist clean

APP_NAME := heketi
CLIENT_PKG_NAME := heketi-client
SHA := $(shell git rev-parse --short HEAD)
BRANCH := $(subst /,-,$(shell git rev-parse --abbrev-ref HEAD))
VER := $(shell git describe)
ARCH := $(shell go env GOARCH)
GOOS := $(shell go env GOOS)
GLIDEPATH := $(shell command -v glide 2> /dev/null)
DIR=.

ifdef APP_SUFFIX
  VERSION = $(VER)-$(subst /,-,$(APP_SUFFIX))
else
ifeq (master,$(BRANCH))
  VERSION = $(VER)
else
  VERSION = $(VER)-$(BRANCH)
endif
endif

# Go setup
GO=go

# Sources and Targets
EXECUTABLES :=$(APP_NAME)
# Build Binaries setting main.version and main.build vars
LDFLAGS :=-ldflags "-X main.HEKETI_VERSION=$(VERSION) -extldflags '-z relro -z now'"
# Package target
PACKAGE :=$(DIR)/dist/$(APP_NAME)-$(VERSION).$(GOOS).$(ARCH).tar.gz
CLIENT_PACKAGE :=$(DIR)/dist/$(APP_NAME)-client-$(VERSION).$(GOOS).$(ARCH).tar.gz
GOFILES=$(shell go list ./... | grep -v vendor)

.DEFAULT: all

all: server client

# print the version
version:
	@echo $(VERSION)

# print the name of the app
name:
	@echo $(APP_NAME)

# print the package path
package:
	@echo $(PACKAGE)

heketi: glide.lock vendor
	go build $(LDFLAGS) -o $(APP_NAME)

server: heketi

vendor:
ifndef GLIDEPATH
	$(info Please install glide.)
	$(info Install it using your package manager or)
	$(info by running: curl https://glide.sh/get | sh.)
	$(info )
	$(error glide is required to continue)
endif
	echo "Installing vendor directory"
	glide install -v

	echo "Building dependencies to make builds faster"
	go install github.com/heketi/heketi

glide.lock: glide.yaml
	echo "Glide.yaml has changed, updating glide.lock"
	glide update -v

client: glide.lock vendor
	@$(MAKE) -C client/cli/go

run: server
	./$(APP_NAME)

test: glide.lock vendor
	go test $(GOFILES)

clean:
	@echo Cleaning Workspace...
	rm -rf $(APP_NAME)
	rm -rf dist
	@$(MAKE) -C client/cli/go clean

$(PACKAGE): all
	@echo Packaging Binaries...
	@mkdir -p tmp/$(APP_NAME)
	@cp $(APP_NAME) tmp/$(APP_NAME)/
	@cp client/cli/go/heketi-cli tmp/$(APP_NAME)/
	@cp etc/heketi.json tmp/$(APP_NAME)/
	@mkdir -p $(DIR)/dist/
	tar -czf $@ -C tmp $(APP_NAME);
	@rm -rf tmp
	@echo
	@echo Package $@ saved in dist directory

$(CLIENT_PACKAGE): all
	@echo Packaging client Binaries...
	@mkdir -p tmp/$(CLIENT_PKG_NAME)/bin
	@mkdir -p tmp/$(CLIENT_PKG_NAME)/share/heketi/openshift/templates
	@mkdir -p tmp/$(CLIENT_PKG_NAME)/share/heketi/kubernetes
	@cp client/cli/go/topology-sample.json tmp/$(CLIENT_PKG_NAME)/share/heketi
	@cp client/cli/go/heketi-cli tmp/$(CLIENT_PKG_NAME)/bin
	@cp extras/openshift/templates/* tmp/$(CLIENT_PKG_NAME)/share/heketi/openshift/templates
	@cp extras/kubernetes/* tmp/$(CLIENT_PKG_NAME)/share/heketi/kubernetes
	@mkdir -p $(DIR)/dist/
	tar -czf $@ -C tmp $(CLIENT_PKG_NAME);
	@rm -rf tmp
	@echo
	@echo Package $@ saved in dist directory

dist: $(PACKAGE) $(CLIENT_PACKAGE)

linux_amd64_dist:
	GOOS=linux GOARCH=amd64 $(MAKE) dist

linux_arm_dist:
	GOOS=linux GOARCH=arm $(MAKE) dist

linux_arm64_dist:
	GOOS=linux GOARCH=arm64 $(MAKE) dist

darwin_amd64_dist:
	GOOS=darwin GOARCH=amd64 $(MAKE) dist

release: darwin_amd64_dist linux_arm64_dist linux_arm_dist linux_amd64_dist

.PHONY: server client test clean name run version release \
        darwin_amd64_dist linux_arm_dist linux_amd64_dist linux_arm64_dist \
        heketi
