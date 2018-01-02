# Inputs:
#
# NMI_FLAVOR - flavor for which config files should be installed
#
# Copies directory rootfs to the ACI's rootfs

# main stamp ensuring that configuration is copied
$(call setup-stamp-file,NMI_COPY_STAMP,$(NMI_FLAVOR)-copy)

# stamp, a dep file and a filelist for generating dependencies on
# config files
$(call setup-stamp-file,NMI_CONF_DEPS_STAMP,$(NMI_FLAVOR)-deps)
$(call setup-dep-file,NMI_DEPMK,$(NMI_FLAVOR)-conf)
$(call setup-filelist-file,NMI_SRC_CONF_FILELIST,$(NMI_FLAVOR)-conf)

# stamp for removing the config dir in the ACI rootfs
$(call setup-stamp-file,NMI_CLEAN_DIR_STAMP,$(NMI_FLAVOR)-cleandir)


# Get source lists
NMI_SRC_DIR := $(MK_SRCDIR)/rootfs
NMI_SRC_FILES := $(call rlist-files,$(NMI_SRC_DIR))
# List of files relative to rootfs
NMI_FILES_REL := $(NMI_SRC_FILES:$(NMI_SRC_DIR)/%=%)

NMI_SRC_DIRS := $(sort $(dir $(NMI_SRC_FILES)))
NMI_DIRS_REL := $(sort $(NMI_SRC_DIRS:$(NMI_SRC_DIR)/%=%))
# $(info REL $(NMI_DIRS_REL))

# The ACI rootfs directory
NMI_DST_BASE := $(STAGE1_ACIROOTFSDIR_$(NMI_FLAVOR))

# Generate lists for INSTALL_FILES and INSTALL_DIRS
NMI_DST_FILES := $(addprefix $(NMI_DST_BASE)/,$(NMI_FILES_REL))
NMI_TRIPLETS := $(call install-file-triplets,$(NMI_SRC_FILES),$(NMI_DST_FILES),0644)
# $(info $$NMI_TRIPLETS is [${NMI_TRIPLETS}])

NMI_DST_DIRS_CHAIN := $(sort $(foreach d,$(NMI_DIRS_REL),$(call dir-chain,$(NMI_DST_BASE),$d)))
# $(info $$NMI_DST_DIRS_CHAIN is [$(NMI_DST_DIRS_CHAIN)])

# this makes sure that config files were copied to the ACI rootfs
$(call generate-stamp-rule,$(NMI_COPY_STAMP),$(NMI_DST_FILES) $(NMI_CONF_DEPS_STAMP))

# this removes the config directory in the ACI rootfs, will be invalidated
# when something changes in source config files
# TODO(CDC) This is no longer the case, as we put some files in rootfs/etc but don't control the dir
# This rule is now disabled
#$(call generate-rm-dir-rule,$(NMI_CLEAN_DIR_STAMP),$(NMI_CONFDIR))

STAGE1_INSTALL_DIRS_$(NMI_FLAVOR) += $(foreach d,$(NMI_DST_DIRS_CHAIN),$d:0755)
STAGE1_INSTALL_FILES_$(NMI_FLAVOR) += $(NMI_TRIPLETS)
STAGE1_SECONDARY_STAMPS_$(NMI_FLAVOR) += $(NMI_COPY_STAMP)

# $(info $$STAGE1_INSTALL_DIRS_$(NMI_FLAVOR) is ${STAGE1_INSTALL_DIRS_$(NMI_FLAVOR)} )

# This filelist of all config files can be generated any time.
$(call generate-deep-filelist,$(NMI_SRC_CONF_FILELIST),$(NMI_SRC_DIR))
# This dep.mk can be generated only after the filelist above was
# generated. Will trigger the removal of the config directory in all
# ACI rootfses if source config files change.
$(call generate-glob-deps,$(NMI_CONF_DEPS_STAMP),$(NMI_CLEAN_DIR_STAMP),$(NMI_DEPMK),,$(NMI_SRC_CONF_FILELIST),$(NMI_SRC_DIR),)

$(call undefine-namespaces,NMI)
