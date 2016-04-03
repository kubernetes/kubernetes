# Inputs:
#
# AIB_FLAVORS - a list of flavors which want this binary to be
# installed in their ACI rootfs
#
# AIB_BINARY - a binary to install
#
# AIB_BUILD_STAMP - a stamp used to build the go binary and generate
# the deps file in the meantime

# In detail for each flavor - generate a stamp which depends on a
# binary in the ACI rootfs directory. The binary will be copied to the
# ACI rootfs only after the initial /usr contents are prepared.

# The path of this file. This file is included (or at least it should
# be) with a standard include directive instead of our inc-one (or
# inc-many), so the MK_PATH, MK_FILENAME and MK_SRCDIR are set to
# values specific to the parent file (that is - including this one).
_AIB_PATH_ := $(lastword $(MAKEFILE_LIST))

$(foreach flavor,$(AIB_FLAVORS), \
	$(call setup-stamp-file,_AIB_STAMP_,$(flavor)) \
	$(eval _AIB_NAME_ := $(notdir $(AIB_BINARY))) \
	$(eval _AIB_ACI_BINARY_ := $(STAGE1_ACIROOTFSDIR_$(flavor))/$(_AIB_NAME_)) \
	$(eval STAGE1_SECONDARY_STAMPS_$(flavor) += $(_AIB_STAMP_)) \
	$(eval STAGE1_INSTALL_FILES_$(flavor) += $(AIB_BINARY):$(_AIB_ACI_BINARY_):-) \
	$(call add-dependency,$(_AIB_ACI_BINARY_),$(MK_PATH) $(_AIB_PATH_) $(AIB_BUILD_STAMP)) \
	$(call generate-stamp-rule,$(_AIB_STAMP_),$(_AIB_ACI_BINARY_),,,))

$(call undefine-namespaces,AIB _AIB)
