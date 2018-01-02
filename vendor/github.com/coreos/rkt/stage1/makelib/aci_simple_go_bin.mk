# inputs cleared after including this file:
#
# ASGB_FLAVORS - which flavors want this app, set to STAGE1_FLAVORS if
# empty

# There are two parts:
#
# 1. Building the binary - it is done only once.
#
# 2. Installing the binary into the ACI rootfs directory - this is
# done once for each flavor.

# Initial stuff

# The path of this file. This file is included (or at least it should
# be) with a standard include directive instead of our inc-one (or
# inc-many), so the MK_PATH, MK_FILENAME and MK_SRCDIR are set to
# values specific to the parent file (that is - including this one).
_ASGB_PATH_ := $(lastword $(MAKEFILE_LIST))
# Name of a binary, deduced upon filename of a parent Makefile.
_ASGB_NAME_ := $(patsubst %.mk,%,$(MK_FILENAME))
# Path to built binary. Not the one in the ACI rootfs.
_ASGB_BINARY_ := $(TARGET_TOOLSDIR)/$(_ASGB_NAME_)

ifeq ($(ASGB_FLAVORS),)

ASGB_FLAVORS := $(STAGE1_FLAVORS)

endif

# 1.

$(call setup-stamp-file,_ASGB_BUILD_STAMP_,binary-build)
$(call generate-stamp-rule,$(_ASGB_BUILD_STAMP_))

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(_ASGB_BUILD_STAMP_)
BGB_BINARY := $(_ASGB_BINARY_)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)

include makelib/build_go_bin.mk

CLEAN_FILES += $(_ASGB_BINARY_)

$(_ASGB_BINARY_): $(_ASGB_PATH_) $(MK_PATH) | $(TARGET_TOOLSDIR)

# 2.

AIB_FLAVORS := $(ASGB_FLAVORS)
AIB_BUILD_STAMP := $(_ASGB_BUILD_STAMP_)
AIB_BINARY := $(_ASGB_BINARY_)
include stage1/makelib/aci_install_bin.mk

$(call undefine-namespaces,ASGB _ASGB)
