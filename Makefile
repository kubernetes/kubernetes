# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DBG_MAKEFILE ?=
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** starting makefile for goal(s) "$(MAKECMDGOALS)")
    $(warning ***** $(shell date))
endif

# It's necessary to set this because some docker images don't make sh -> bash.
SHELL := /bin/bash

# We don't need make's built-in rules.
MAKEFLAGS += --no-builtin-rules --warn-undefined-variables
.SUFFIXES:

# Old-skool build tools.
#
# Targets (see each target for more information):
#   all: Build code.
#   check: Run tests.
#   test: Run tests.
#   clean: Clean up.

OUT_DIR ?= _output
BIN_DIR := $(OUT_DIR)/bin
PRJ_SRC_PATH := k8s.io/kubernetes
GENERATED_FILE_PREFIX := zz_generated.

# Metadata for driving the build lives here.
META_DIR := .make

GOFLAGS ?=
KUBE_GOFLAGS = $(GOFLAGS)
export KUBE_GOFLAGS

GOLDFLAGS ?=
KUBE_GOLDFLAGS = $(GOLDFLAGS)
export KUBE_GOLDFLAGS

# Build code.
#
# Args:
#   WHAT: Directory names to build.  If any of these directories has a 'main'
#     package, the build will produce executable files under $(OUT_DIR)/go/bin.
#     If not specified, "everything" will be built.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make
#   make all
#   make all WHAT=cmd/kubelet GOFLAGS=-v
#
# TODO: It's a not ideal that we build the tools on every invocation.
all: gen_deepcopy gen_conversion build
.PHONY: all

# Build ginkgo
#
# Example:
# make ginkgo
ginkgo:
	hack/make-rules/build.sh vendor/github.com/onsi/ginkgo/ginkgo
.PHONY: ginkgo

# This is a helper to break circular dependencies with recursive `make` calls.
build:
	@hack/make-rules/build.sh $(WHAT)
.PHONY: build

# Runs all the presubmission verifications.
#
# Args:
#   BRANCH: Branch to be passed to hack/verify-godeps.sh script.
#
# Example:
#   make verify
#   make verify BRANCH=branch_x
verify:
	@KUBE_VERIFY_GIT_BRANCH=$(BRANCH) hack/make-rules/verify.sh -v
.PHONY: verify

# Build and run tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make check
#   make test
#   make check WHAT=pkg/kubelet GOFLAGS=-v
check test: gen_deepcopy gen_conversion
	@hack/make-rules/test.sh $(WHAT) $(TESTS)
.PHONY: check test

# Build and run integration tests.
#
# Example:
#   make test-integration
test-integration: gen_deepcopy gen_conversion
	@hack/make-rules/test-integration.sh
.PHONY: test-integration

# Build and run end-to-end tests.
#
# Example:
#   make test-e2e
test-e2e: ginkgo gen_deepcopy gen_conversion
	@go run hack/e2e.go -v --build --up --test --down
.PHONY: test-e2e

# Build and run node end-to-end tests.
#
# Args:
#  FOCUS: regexp that matches the tests to be run.  Defaults to "".
#  SKIP: regexp that matches the tests that needs to be skipped.  Defaults to "".
#  RUN_UNTIL_FAILURE: Ff true, pass --untilItFails to ginkgo so tests are run repeatedly until they fail.  Defaults to false.
#  REMOTE: If true, run the tests on a remote host instance on GCE.  Defaults to false.
#  IMAGES: for REMOTE=true only.  Comma delimited list of images for creating remote hosts to run tests against.  Defaults to "e2e-node-containervm-v20160321-image".
#  LIST_IMAGES: If true, don't run tests.  Just output the list of available images for testing.  Defaults to false.
#  HOSTS: for REMOTE=true only.  Comma delimited list of running gce hosts to run tests against.  Defaults to "".
#  DELETE_INSTANCES: for REMOTE=true only.  Delete any instances created as part of this test run.  Defaults to false.
#  ARTIFACTS: for REMOTE=true only.  Local directory to scp test artifacts into from the remote hosts.  Defaults to ""/tmp/_artifacts".
#  REPORT: for REMOTE=false only.  Local directory to write juntil xml results to.  Defaults to "/tmp/".
#  CLEANUP: for REMOTE=true only.  If false, do not stop processes or delete test files on remote hosts.  Defaults to true.
#  IMAGE_PROJECT: for REMOTE=true only.  Project containing images provided to IMAGES.  Defaults to "kubernetes-node-e2e-images".
#  INSTANCE_PREFIX: for REMOTE=true only.  Instances created from images will have the name "${INSTANCE_PREFIX}-${IMAGE_NAME}".  Defaults to "test"/
#
# Example:
#   make test-e2e-node FOCUS=kubelet SKIP=container
#   make test-e2e-node REMOTE=true DELETE_INSTANCES=true
# Build and run tests.
test-e2e_node: ginkgo gen_deepcopy gen_conversion
	@hack/make-rules/test-e2e_node.sh
.PHONY: test-e2e_node

# Build and run cmdline tests.
#
# Example:
#   make test-cmd
test-cmd:
	@hack/make-rules/test-cmd.sh
.PHONY: test-cmd

# Remove all build artifacts.
#
# Example:
#   make clean
clean: clean_generated clean_meta
	build/make-clean.sh
	rm -rf $(OUT_DIR)
	rm -rf Godeps/_workspace # Just until we are sure it is gone
.PHONY: clean

# Remove make-related metadata files.
#
# Example:
#   make clean_meta
clean_meta:
	rm -rf $(META_DIR)
.PHONE: clean_meta

# Remove all auto-generated artifacts.
#
# Example:
#   make clean_generated
clean_generated:
	find . -type f -name $(GENERATED_FILE_PREFIX)\* | xargs rm -f
.PHONY: clean_generated

# Run 'go vet'.
#
# Args:
#   WHAT: Directory names to vet.  All *.go files under these
#     directories will be vetted.  If not specified, "everything" will be
#     vetted.
#
# Example:
#   make vet
#   make vet WHAT=pkg/kubelet
vet:
	@hack/make-rules/vet.sh $(WHAT)
.PHONY: vet

# Build a release
#
# Example:
#   make release
release: gen_deepcopy gen_conversion
	@build/release.sh
.PHONY: release

# Build a release, but skip tests
#
# Example:
#   make release-skip-tests
release-skip-tests quick-release: gen_deepcopy gen_conversion
	@KUBE_RELEASE_RUN_TESTS=n KUBE_FASTBUILD=true build/release.sh
.PHONY: release-skip-tests quick-release

# Cross-compile for all platforms
#
# Example:
#   make cross
.PHONY: cross
cross:
	@hack/make-rules/cross.sh

#
# Code-generation logic.
#

# This variable holds a list of every directory that contains Go files in this
# project.  Other rules and variables can use this as a starting point to
# reduce filesystem accesses.
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** finding all *.go dirs)
endif
ALL_GO_DIRS := $(shell             \
    find .                         \
        -not \(                    \
            \(                     \
                -path ./vendor -o  \
                -path ./_\* -o     \
                -path ./.\*        \
            \) -prune              \
        \)                         \
        -type f -name \*.go        \
        | sed 's|^./||'            \
        | xargs dirname            \
        | sort -u                  \
)

# The name of the make metadata file listing Go files.
GOFILES_META := gofiles.mk

# Establish a dependency between the deps file and the dir.  Whenever a dir
# changes (files added or removed) the deps file will be considered stale.
#
# The variable value was set in $(GOFILES_META) and included as part of the
# dependency management logic.
#
# This is looser than we really need (e.g. we don't really care about non *.go
# files or even *_test.go files), but this is much easier to represent.
#
# Because we 'sinclude' the deps file, it is considered for rebuilding, as part
# of make's normal evaluation.  If it gets rebuilt, make will restart.
#
# The '$(eval)' is needed because this has a different RHS for each LHS, and
# would otherwise produce results that make can't parse.
$(foreach dir, $(ALL_GO_DIRS), $(eval           \
    $(META_DIR)/$(dir)/$(GOFILES_META): $(dir)  \
))

# How to rebuild a deps file.  When make determines that the deps file is stale
# (see above), it executes this rule, and then re-loads the deps file.
#
# This is looser than we really need (e.g. we don't really care about test
# files), but this is MUCH faster than calling `go list`.
#
# We regenerate the output file in order to satisfy make's "newer than" rules,
# but we only need to rebuild targets if the contents actually changed.  That
# is what the .stamp file represents.
$(foreach dir, $(ALL_GO_DIRS), $(META_DIR)/$(dir)/$(GOFILES_META)):
	@FILES=$$(ls $</*.go | grep -v $(GENERATED_FILE_PREFIX)); \
	mkdir -p $(@D);                                           \
	echo "gofiles__$< := $$(echo $${FILES})" >$@.tmp;         \
	cmp -s $@.tmp $@ || touch $@.stamp;                       \
	mv $@.tmp $@

# Include any deps files as additional Makefile rules.  This triggers make to
# consider the deps files for rebuild, which makes the whole
# dependency-management logic work.  'sinclude' is "silent include" which does
# not fail if the file does not exist.
$(foreach dir, $(ALL_GO_DIRS), $(eval            \
    sinclude $(META_DIR)/$(dir)/$(GOFILES_META)  \
))

# Generate a list of all files that have a `+k8s:` comment-tag.  This will be
# used to derive lists of files/dirs for generation tools.
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** finding all +k8s: tags)
endif
ALL_K8S_TAG_FILES := $(shell                             \
    find $(ALL_GO_DIRS) -maxdepth 1 -type f -name \*.go  \
        | xargs grep -l '^// *+k8s:'                     \
)

#
# Deep-copy generation
#
# Any package that wants deep-copy functions generated must include a
# comment-tag in column 0 of one file of the form:
#     // +k8s:deepcopy-gen=<VALUE>
#
# The <VALUE> may be one of:
#     generate: generate deep-copy functions into the package
#     register: generate deep-copy functions and register them with a
#               scheme

# Find all the directories that request deep-copy generation.
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** finding all +k8s:deepcopy-gen tags)
endif
DEEP_COPY_DIRS := $(shell \
    grep -l '+k8s:deepcopy-gen=' $(ALL_K8S_TAG_FILES)  \
        | xargs dirname                                \
        | sort -u                                      \
)

# The result file, in each pkg, of deep-copy generation.
DEEP_COPY_BASENAME := $(GENERATED_FILE_PREFIX)deep_copy
DEEP_COPY_FILENAME := $(DEEP_COPY_BASENAME).go

# Unfortunately there's not a good way to use Go's build tools to check
# if a binary needs to be rebuilt.  We just have to try it.
gen_deepcopy:
	@$(MAKE) -s build WHAT=cmd/libs/go2idl/deepcopy-gen
	@$(MAKE) -s $(addsuffix /$(DEEP_COPY_FILENAME), $(DEEP_COPY_DIRS))

# For each dir in DEEP_COPY_DIRS, this establishes a dependency between the
# output file and the input files that should trigger a rebuild.
#
# Note that this is a deps-only statement, not a full rule (see below).
# This has to be done in a distinct step because wildcards don't seem to work
# in static pattern rules.
#
# The '$(eval)' is needed because this has a different RHS for each LHS, and
# would otherwise produce results that make can't parse.
#
# We depend on the $(GOFILES_META).stamp to detect when the set of input files
# has changed.  This allows us to detect deleted input files.
$(foreach dir, $(DEEP_COPY_DIRS), $(eval                                    \
    $(dir)/$(DEEP_COPY_FILENAME): $(META_DIR)/$(dir)/$(GOFILES_META).stamp  \
                                   $(gofiles__$(dir))                       \
                                   $(BIN_DIR)/deepcopy-gen                  \
))

# For each dir in DEEP_COPY_DIRS, handle deep-copy generation.
# This has to be done in two steps because wildcards don't seem to work in
# static pattern rules.
$(addsuffix /$(DEEP_COPY_FILENAME), $(DEEP_COPY_DIRS)):
	$(BIN_DIR)/deepcopy-gen \
	    -i $(PRJ_SRC_PATH)/$$(dirname $@) \
	    --bounding-dirs $(PRJ_SRC_PATH) \
	    -O $(DEEP_COPY_BASENAME)

#
# Conversion generation
#
# Any package that wants conversion functions generated must include one or
# more comment-tags in any .go file, in column 0, of the form:
#     // +k8s:conversion-gen=<CONVERSION_TARGET_DIR>
#
# The CONVERSION_TARGET_DIR is a project-local path to another directory which
# should be considered when evaluating peer types for conversions.  Types which
# are found in the source package (where conversions are being generated)
# but do not have a peer in one of the target directories will not have
# conversions generated.
#
# TODO: it might be better in the long term to make peer-types explicit in the
# IDL.

# All directories that request any form of conversion generation.
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** finding all +k8s:conversion-gen tags)
endif
CONVERSION_DIRS := $(shell                                \
    grep '^// *+k8s:conversion-gen=' $(ALL_K8S_TAG_FILES) \
        | cut -f1 -d:                                     \
        | xargs dirname                                   \
        | sort -u                                         \
)

# The result file, in each pkg, of conversion generation.
CONVERSION_BASENAME := $(GENERATED_FILE_PREFIX)conversion
CONVERSION_FILENAME := $(CONVERSION_BASENAME).go

# The name of the make metadata file controlling conversions.
CONVERSIONS_META := conversions.mk

# Unfortunately there's not a good way to use Go's build tools to check
# if a binary needs to be rebuilt.  We just have to try it.
gen_conversion:
	@$(MAKE) -s build WHAT=cmd/libs/go2idl/conversion-gen
	@$(MAKE) -s $(addsuffix /$(CONVERSION_FILENAME), $(CONVERSION_DIRS))

# Establish a dependency between the deps file and the dir.  Whenever a dir
# changes (files added or removed) the deps file will be considered stale.
#
# This is looser than we really need (e.g. we don't really care about non *.go
# files or even *_test.go files), but this is much easier to represent.
#
# Because we 'sinclude' the deps file, it is considered for rebuilding, as part
# of make's normal evaluation.  If it gets rebuilt, make will restart.
#
# The '$(eval)' is needed because this has a different RHS for each LHS, and
# would otherwise produce results that make can't parse.
$(foreach dir, $(CONVERSION_DIRS), $(eval           \
    $(META_DIR)/$(dir)/$(CONVERSIONS_META): $(dir)  \
))

# How to rebuild a deps file.  When make determines that the deps file is stale
# (see above), it executes this rule, and then re-loads the deps file.
#
# This is looser than we really need (e.g. we don't really care about test
# files), but this is MUCH faster than calling `go list`.
#
# We regenerate the output file in order to satisfy make's "newer than" rules,
# but we only need to rebuild targets if the contents actually changed.  That
# is what the .stamp file represents.
$(foreach dir, $(CONVERSION_DIRS), $(META_DIR)/$(dir)/$(CONVERSIONS_META)):
	@TAGS=$$(grep -h '^// *+k8s:conversion-gen=' $</*.go                   \
	    | cut -f2- -d=                                                     \
	    | sed 's|$(PRJ_SRC_PATH)/||');                                     \
	mkdir -p $(@D);                                                        \
	echo "conversions__$< := $$(echo $${TAGS})" >$@.tmp;                   \
	cmp -s $@.tmp $@ || touch $@.stamp;                                    \
	mv $@.tmp $@

# Include any deps files as additional Makefile rules.  This triggers make to
# consider the deps files for rebuild, which makes the whole
# dependency-management logic work.  'sinclude' is "silent include" which does
# not fail if the file does not exist.
$(foreach dir, $(CONVERSION_DIRS), $(eval            \
    sinclude $(META_DIR)/$(dir)/$(CONVERSIONS_META)  \
))

# For each dir in CONVERSION_DIRS, this establishes a dependency between the
# output file and the input files that should trigger a rebuild.
#
# The variable value was set in $(GOFILES_META) and included as part of the
# dependency management logic.
#
# Note that this is a deps-only statement, not a full rule (see below).
# This has to be done in a distinct step because wildcards don't seem to work
# in static pattern rules.
#
# The '$(eval)' is needed because this has a different RHS for each LHS, and
# would otherwise produce results that make can't parse.
#
# We depend on the $(GOFILES_META).stamp to detect when the set of input files
# has changed.  This allows us to detect deleted input files.
$(foreach dir, $(CONVERSION_DIRS), $(eval                                    \
    $(dir)/$(CONVERSION_FILENAME): $(META_DIR)/$(dir)/$(GOFILES_META).stamp  \
                                   $(gofiles__$(dir))                        \
                                   $(BIN_DIR)/conversion-gen                 \
))

# For each dir in CONVERSION_DIRS, for each target in $(conversions__$(dir)),
# this establishes a dependency between the output file and the input files
# that should trigger a rebuild.
#
# The variable value was set in $(GOFILES_META) and included as part of the
# dependency management logic.
#
# Note that this is a deps-only statement, not a full rule (see below).
# This has to be done in a distinct step because wildcards don't seem to work
# in static pattern rules.
#
# The '$(eval)' is needed because this has a different RHS for each LHS, and
# would otherwise produce results that make can't parse.
#
# We depend on the $(GOFILES_META).stamp to detect when the set of input files
# has changed.  This allows us to detect deleted input files.
$(foreach dir, $(CONVERSION_DIRS),                                               \
    $(foreach tgt, $(conversions__$(dir)), $(eval                                \
        $(dir)/$(CONVERSION_FILENAME): $(META_DIR)/$(tgt)/$(GOFILES_META).stamp  \
                                       $(gofiles__$(tgt))                        \
    ))                                                                           \
)

# For each dir in CONVERSION_DIRS, this generates a rule to auto-generate
# conversion code.  Dependencies of the target have been populated above.
$(addsuffix /$(CONVERSION_FILENAME), $(CONVERSION_DIRS)):
	@$(BIN_DIR)/conversion-gen \
	    -i $(PRJ_SRC_PATH)/$(@D) \
	    -O $(CONVERSION_BASENAME)
