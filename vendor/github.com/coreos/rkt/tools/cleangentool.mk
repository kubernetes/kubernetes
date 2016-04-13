$(call setup-stamp-file,CLEANGENTOOL_STAMP)

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(CLEANGENTOOL_STAMP)
BGB_PKG_IN_REPO := tools/cleangen
BGB_BINARY := $(CLEANGENTOOL)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)

CLEAN_FILES += $(CLEANGENTOOL)

$(call generate-stamp-rule,$(CLEANGENTOOL_STAMP))

$(CLEANGENTOOL): $(MK_PATH) | $(TOOLSDIR)

include makelib/build_go_bin.mk

# CLEANGENTOOL_STAMP deliberately not cleared
