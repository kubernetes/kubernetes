all: build

########################################################################
##                             GOLANG                                 ##
########################################################################

# If GOPATH isn't defined then set its default location.
ifeq (,$(strip $(GOPATH)))
GOPATH := $(HOME)/go
else
# If GOPATH is already set then update GOPATH to be its own
# first element.
GOPATH := $(word 1,$(subst :, ,$(GOPATH)))
endif
export GOPATH


########################################################################
##                             PROTOC                                 ##
########################################################################

# Only set PROTOC_VER if it has an empty value.
ifeq (,$(strip $(PROTOC_VER)))
PROTOC_VER := 3.3.0
endif

PROTOC_OS := $(shell uname -s)
ifeq (Darwin,$(PROTOC_OS))
PROTOC_OS := osx
endif

PROTOC_ARCH := $(shell uname -m)
ifeq (i386,$(PROTOC_ARCH))
PROTOC_ARCH := x86_32
endif

PROTOC := ./protoc
PROTOC_ZIP := protoc-$(PROTOC_VER)-$(PROTOC_OS)-$(PROTOC_ARCH).zip
PROTOC_URL := https://github.com/google/protobuf/releases/download/v$(PROTOC_VER)/$(PROTOC_ZIP)
PROTOC_TMP_DIR := .protoc
PROTOC_TMP_BIN := $(PROTOC_TMP_DIR)/bin/protoc

$(PROTOC):
	-mkdir -p "$(PROTOC_TMP_DIR)" && \
	  curl -L $(PROTOC_URL) -o "$(PROTOC_TMP_DIR)/$(PROTOC_ZIP)" && \
	  unzip "$(PROTOC_TMP_DIR)/$(PROTOC_ZIP)" -d "$(PROTOC_TMP_DIR)" && \
	  chmod 0755 "$(PROTOC_TMP_BIN)" && \
	  cp -f "$(PROTOC_TMP_BIN)" "$@"
	-rm -fr "$(PROTOC_TMP_DIR)"
	stat "$@" > /dev/null 2>&1


########################################################################
##                          PROTOC-GEN-GO                             ##
########################################################################

# This is the recipe for getting and installing the go plug-in
# for protoc
PROTOC_GEN_GO_PKG := github.com/golang/protobuf/protoc-gen-go
PROTOC_GEN_GO := protoc-gen-go
$(PROTOC_GEN_GO):
	go get -d $(PROTOC_GEN_GO_PKG) && \
	go build -o "$@" $(PROTOC_GEN_GO_PKG)


########################################################################
##                              PATH                                  ##
########################################################################

# Update PATH with the current directory. This enables the protoc
# binary to discover the protoc-gen-go binary, built inside this
# directory.
export PATH := $(shell pwd):$(PATH)


########################################################################
##                              BUILD                                 ##
########################################################################
CSI_GO := csi/csi.pb.go
CSI_A := csi.a
CSI_PROTO := ../../csi.proto
CSI_GO_TMP := csi/.build/csi.pb.go

# This recipe generates the go language bindings to a temp area.
$(CSI_GO_TMP): $(CSI_PROTO) | $(PROTOC) $(PROTOC_GEN_GO)
	@mkdir -p "$(@D)"
	$(PROTOC) -I "$(<D)" --go_out=plugins=grpc:"$(@D)" "$<"

# The temp language bindings are compared to the ones that are
# versioned. If they are different then it means the language
# bindings were not updated prior to being committed.
$(CSI_GO): $(CSI_GO_TMP)
ifeq (true,$(TRAVIS))
	diff "$@" "$?"
else
	@mkdir -p "$(@D)"
	diff "$@" "$?" > /dev/null 2>&1 || cp -f "$?" "$@"
endif

# This recipe builds the Go archive from the sources in three steps:
#
#   1. Go get any missing dependencies.
#   2. Cache the packages.
#   3. Build the archive file.
$(CSI_A): $(CSI_GO)
	go get -d ./...
	go install ./csi
	go build -o "$@" ./csi

build: $(CSI_A)

clean:
	go clean -i ./...
	rm -f "$(CSI_A)"

clobber: clean
	rm -fr "$(PROTOC)" "$(PROTOC_GEN_GO)" "$(dir $(CSI_GO))"

.PHONY: clean clobber
