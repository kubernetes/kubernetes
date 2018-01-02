# The path of this file. This file is included (or at least it should
# be) with a standard include directive instead of our inc-one (or
# inc-many), so the MK_PATH, MK_FILENAME and MK_SRCDIR are set to
# values specific to the parent file (that is - including this one).
_FAB_PATH_ := $(lastword $(MAKEFILE_LIST))
# Name of a binary, deduced upon filename of a parent Makefile.
_FAB_NAME_ := $(patsubst %.mk,%,$(MK_FILENAME))
# Path to built binary. Not the one in the ACI rootfs.
_FAB_BINARY_ := $(FLY_TOOLSDIR)/$(_FAB_NAME_)
# Path to the built binary, copied to ACI rootfs directory
_FAB_ACI_BINARY_ := $(FLY_ACIROOTFSDIR)/$(_FAB_NAME_)

$(call setup-stamp-file,_FAB_STAMP_,binary-build)
$(call generate-stamp-rule,$(_FAB_STAMP_))

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(_FAB_STAMP_)
BGB_BINARY := $(_FAB_BINARY_)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)

include makelib/build_go_bin.mk

$(_FAB_BINARY_): $(_FAB_PATH_) $(MK_PATH) | $(FLY_TOOLSDIR)
$(_FAB_STAMP_): $(_FAB_ACI_BINARY_)

CLEAN_FILES += $(_FAB_BINARY_)
INSTALL_FILES += $(_FAB_BINARY_):$(_FAB_ACI_BINARY_):-
FLY_STAMPS += $(_FAB_STAMP_)

$(call undefine-namespaces,FAB _FAB)
