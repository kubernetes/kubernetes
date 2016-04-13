$(call setup-stamp-file,TST_SHORT_TESTS_STAMP,/short-tests)

TST_DIRS_WITH_GOFILES := $(call go-find-directories,$(GO_TEST_PACKAGES),GoFiles)
TST_DIRS_WITH_TESTGOFILES := $(call go-find-directories,$(GO_TEST_PACKAGES),TestGoFiles,tests)

# gofmt takes list of directories
TST_GOFMT_DIRS := $(foreach d,$(TST_DIRS_WITH_GOFILES),./$d)
# go vet and go test take a list of packages
TST_GO_VET_PACKAGES := $(foreach d,$(TST_DIRS_WITH_GOFILES),$(REPO_PATH)/$d)
TST_GO_TEST_PACKAGES := $(foreach d,$(TST_DIRS_WITH_TESTGOFILES),$(REPO_PATH)/$d)

$(call forward-vars,$(TST_SHORT_TESTS_STAMP), \
	GOFMT TST_GOFMT_DIRS GO_ENV GO TST_GO_VET_PACKAGES RKT_TAGS \
	TST_GO_TEST_PACKAGES)
$(TST_SHORT_TESTS_STAMP):
	$(VQ) \
	set -e; \
	$(call vb,vt,GOFMT,$(TST_GOFMT_DIRS)) \
	res=$$($(GOFMT) -l $(TST_GOFMT_DIRS)); \
	if [ -n "$${res}" ]; then echo -e "gofmt checking failed:\n$${res}"; exit 1; fi; \
	$(call vb,vt,GO VET,$(TST_GO_VET_PACKAGES)) \
	res=$$($(GO_ENV) "$(GO)" vet $(TST_GO_VET_PACKAGES)); \
	if [ -n "$${res}" ]; then echo -e "govet checking failed:\n$${res}"; exit 1; fi; \
	$(call vb,vt,(C) CHECK) \
	res=$$( \
		for file in $$(find . -type f -iname '*.go' ! -path './Godeps/*'); do \
			head -n1 "$${file}" | grep -Eq "(Copyright|generated)" || echo -e "  $${file}"; \
		done; \
	); \
	if [ -n "$${res}" ]; then echo -e "license header checking failed:\n$${res}"; exit 1; fi; \
	$(call vb,vt,GO TEST,$(TST_GO_TEST_PACKAGES)) \
	$(GO_ENV) "$(GO)" test -timeout 60s -cover $(RKT_TAGS) $(TST_GO_TEST_PACKAGES) --race

TOPLEVEL_CHECK_STAMPS += $(TST_SHORT_TESTS_STAMP)
TOPLEVEL_UNIT_CHECK_STAMPS += $(TST_SHORT_TESTS_STAMP)

ifeq ($(RKT_RUN_FUNCTIONAL_TESTS),yes)

$(call inc-one,functional.mk)

else

$(call setup-stamp-file,TST_FUNC_TESTS_DISABLED_STAMP,/func-test-disabled)

TOPLEVEL_FUNCTIONAL_CHECK_STAMPS += $(TST_FUNC_TESTS_DISABLED_STAMP)

$(TST_FUNC_TESTS_DISABLED_STAMP):
	$(VQ) \
	echo 'Functional tests are disabled, pass --enable-functional-tests to configure script to enable them.' >&2; \
	false

endif

$(call undefine-namespaces,TST)
