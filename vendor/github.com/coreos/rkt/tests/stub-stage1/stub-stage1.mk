# generated stage1 image
FTST_SS1_IMAGE := $(FTST_TMPDIR)/rkt-stub-stage1.aci
# stamp generating the stage1 image
$(call setup-stamp-file,FTST_SS1_STAMP,/ss1-image)
# stage1 image directory
FTST_SS1_ACIDIR := $(FTST_TMPDIR)/ss1
# stage1 image rootfs directory
FTST_SS1_ACIROOTFSDIR := $(FTST_SS1_ACIDIR)/rootfs
# stage1 image source manifest
FTST_SS1_MANIFEST_SRC := $(MK_SRCDIR)/manifest
# stage1 image manifest in aci dir
FTST_SS1_MANIFEST := $(FTST_SS1_ACIDIR)/manifest
# run binary in rootfs
FTST_SS1_RUN_BINARY := $(FTST_SS1_ACIROOTFSDIR)/run
# enter binary in rootfs
FTST_SS1_ENTER_BINARY := $(FTST_SS1_ACIROOTFSDIR)/enter
# gc binary in rootfs
FTST_SS1_GC_BINARY := $(FTST_SS1_ACIROOTFSDIR)/gc
# TODO: would be nice for stage0 to create those for us instead.
# special directories for stage1
FTST_SS1_RESERVED_DIRS := opt/stage1 rkt/env rkt/status
# special directories for stage1 in rootfs
FTST_SS1_RESERVED_DIRS_IN_ROOTFS := $(foreach d,opt/stage1 rkt/env rkt/status,$(FTST_SS1_ACIROOTFSDIR)/$d)
# chains of special directories for stage1 in rootfs
FTST_SS1_RESERVED_DIR_CHAINS := \
	$(foreach d,$(FTST_SS1_RESERVED_DIRS), \
		$(call dir-chain,$(FTST_SS1_ACIROOTFSDIR),$d))

INSTALL_FILES += $(FTST_SS1_MANIFEST_SRC):$(FTST_SS1_MANIFEST):0644
INSTALL_DIRS += \
	$(FTST_SS1_ACIDIR):- \
	$(FTST_SS1_ACIROOTFSDIR):- \
	$(foreach d,$(FTST_SS1_RESERVED_DIR_CHAINS),$d:-)
CLEAN_FILES += \
	$(FTST_SS1_IMAGE) \
	$(FTST_SS1_RUN_BINARY) \
	$(FTST_SS1_ENTER_BINARY) \
	$(FTST_SS1_GC_BINARY)

$(call generate-stamp-rule,$(FTST_SS1_STAMP),$(FTST_SS1_IMAGE))

$(call forward-vars,$(FTST_SS1_IMAGE), \
	FTST_SS1_ACIDIR)
$(FTST_SS1_IMAGE): $(FTST_SS1_MANIFEST) $(FTST_SS1_RUN_BINARY) $(FTST_SS1_ENTER_BINARY) $(FTST_SS1_GC_BINARY) $(ACTOOL_STAMP) | $(FTST_SS1_ACIDIR) $(FTST_SS1_RESERVED_DIRS_IN_ROOTFS)
	$(VQ) \
	$(call vb,vt,ACTOOL,$(call vsp,$@)) \
	"$(ACTOOL)" build --overwrite "$(FTST_SS1_ACIDIR)" "$@"

$(FTST_SS1_RUN_BINARY) $(FTST_SS1_ENTER_BINARY) $(FTST_SS1_GC_BINARY): | $(FTST_SS1_ACIROOTFSDIR)

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FTST_SS1_STAMP)
BGB_BINARY := $(FTST_SS1_RUN_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/run

include makelib/build_go_bin.mk

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FTST_SS1_STAMP)
BGB_BINARY := $(FTST_SS1_ENTER_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/enter

include makelib/build_go_bin.mk

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FTST_SS1_STAMP)
BGB_BINARY := $(FTST_SS1_GC_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/gc

include makelib/build_go_bin.mk

# do not undefine the FTST_SS1_IMAGE and FTST_SS1_STAMP variables, we
# will use them in functional.mk
$(call undefine-namespaces,FTST_SS1,FTST_SS1_IMAGE FTST_SS1_STAMP)
