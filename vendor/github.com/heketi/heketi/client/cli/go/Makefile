#
# Based on http://chrismckenzie.io/post/deploying-with-golang/
#

.PHONY: version all run dist clean
	
APP_NAME := heketi-cli
SHA := $(shell git rev-parse --short HEAD)
BRANCH := $(subst /,-,$(shell git rev-parse --abbrev-ref HEAD))
VER := $(shell git describe)
ARCH := $(shell go env GOARCH)
GOOS := $(shell go env GOOS)
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
TEST="go test"

# Sources and Targets
EXECUTABLES :=$(APP_NAME)
# Build Binaries setting main.version and main.build vars
LDFLAGS :=-ldflags "-X main.HEKETI_CLI_VERSION=$(VERSION) -extldflags '-z relro -z now'"
# Package target
PACKAGE :=$(DIR)/dist/$(APP_NAME)-$(VERSION).$(GOOS).$(ARCH).tar.gz

.DEFAULT: all

all: build

# print the version
version:
	@echo $(VERSION)

# print the name of the app
name:
	@echo $(APP_NAME)

# print the package path
package:
	@echo $(PACKAGE)

build:
	go build $(LDFLAGS) -o $(APP_NAME)

run: build
	./$(APP_NAME)

test: 
	go test ./...

clean:
	@echo Cleaning Workspace...
	rm -rf $(APP_NAME)
	rm -rf dist

$(PACKAGE): all
	@echo Packaging Binaries...
	@mkdir -p tmp/$(APP_NAME)
	@cp $(APP_NAME) tmp/$(APP_NAME)/
	@cp etc/heketi.json tmp/$(APP_NAME)/
	@mkdir -p $(DIR)/dist/
	tar -czf $@ -C tmp $(APP_NAME);
	@rm -rf tmp
	@echo
	@echo Package $@ saved in dist directory

dist: $(PACKAGE)
