$(call setup-stamp-file,FILELISTGENTOOL_STAMP)

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FILELISTGENTOOL_STAMP)
BGB_PKG_IN_REPO := tools/filelistgen
BGB_BINARY := $(FILELISTGENTOOL)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)

CLEAN_FILES += $(FILELISTGENTOOL)

$(call generate-stamp-rule,$(FILELISTGENTOOL_STAMP))

$(FILELISTGENTOOL): $(MK_PATH) | $(TOOLSDIR)

include makelib/build_go_bin.mk

# FILELISTGENTOOL_STAMP deliberately not cleared
