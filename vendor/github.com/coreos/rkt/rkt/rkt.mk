LOCAL_NAME := $(patsubst %.mk,%,$(MK_FILENAME))

$(call setup-stamp-file,RKT_STAMP)

RKT_BINARY := $(BINDIR)/$(LOCAL_NAME)

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(RKT_STAMP)
BGB_BINARY := $(RKT_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)
BGB_GO_FLAGS := $(strip \
	-ldflags " \
		   $(RKT_STAGE1_DEFAULT_NAME_LDFLAGS) \
		   $(RKT_STAGE1_DEFAULT_VERSION_LDFLAGS) \
		   $(RKT_STAGE1_DEFAULT_LOCATION_LDFLAGS) \
		   $(RKT_VERSION_LDFLAGS) \
		   $(RKT_FEATURES_LDFLAGS) \
		   $(RKT_STAGE1_DEFAULT_IMAGE_FILENAME_LDFLAGS) \
		   $(RKT_STAGE1_DEFAULT_IMAGES_DIRECTORY_LDFLAGS) \
		 " \
	$(RKT_TAGS))

CLEAN_FILES += $(BGB_BINARY)
TOPLEVEL_STAMPS += $(RKT_STAMP)

$(call generate-stamp-rule,$(RKT_STAMP))

$(BGB_BINARY): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

$(call undefine-namespaces,LOCAL)
# RKT_STAMP deliberately not cleared
# RKT_BINARY deliberately not cleared
