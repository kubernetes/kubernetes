$(call setup-stamp-file,ACTOOL_STAMP)

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(ACTOOL_STAMP)
BGB_PKG_IN_REPO := vendor/github.com/appc/spec/actool
BGB_BINARY := $(ACTOOL)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD) CC=$(CC_FOR_BUILD)

CLEAN_FILES += $(ACTOOL)

$(call generate-stamp-rule,$(ACTOOL_STAMP))

$(ACTOOL): $(MK_PATH) | $(TOOLSDIR)

include makelib/build_go_bin.mk

# ACTOOL_STAMP deliberately not cleared
