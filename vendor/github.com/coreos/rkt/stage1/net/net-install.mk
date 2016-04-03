# Inputs:
#
# NMI_FLAVOR - flavor for which config files should be installed

# main stamp ensuring that configuration is copied
$(call setup-stamp-file,NMI_COPY_STAMP,$(NMI_FLAVOR)-copy)

# stamp, a dep file and a filelist for generating dependencies on
# config files
$(call setup-stamp-file,NMI_CONF_DEPS_STAMP,$(NMI_FLAVOR)-deps)
$(call setup-dep-file,NMI_DEPMK,$(NMI_FLAVOR)-conf)
$(call setup-filelist-file,NMI_SRC_CONF_FILELIST,$(NMI_FLAVOR)-conf)

# stamp for removing the config dir in the ACI rootfs
$(call setup-stamp-file,NMI_CLEAN_DIR_STAMP,$(NMI_FLAVOR)-cleandir)

# directory with config files in source tree
NMI_SRC_CONFDIR := $(MK_SRCDIR)/conf
NMI_SUFFIX := .conf
NMI_SRC_CONFFILES := $(wildcard $(NMI_SRC_CONFDIR)/*$(NMI_SUFFIX))

# an etc directory in the ACI rootfs, we expect it to exist
NMI_DIRS_BASE := $(STAGE1_ACIROOTFSDIR_$(NMI_FLAVOR))/etc
# subdirectories of the etc directory created by this Makefile
NMI_DIRS_REST := rkt/net.d
NMI_DIR_CHAIN := $(call dir-chain,$(NMI_DIRS_BASE),$(NMI_DIRS_REST))
NMI_CONFDIR := $(NMI_DIRS_BASE)/$(NMI_DIRS_REST)
# configuration files in the ACI rootfs
NMI_CONFFILES := $(addprefix $(NMI_CONFDIR)/,$(notdir $(NMI_SRC_CONFFILES)))
# triplets for INSTALL_FILES
NMI_TRIPLETS := $(call install-file-triplets,$(NMI_SRC_CONFFILES),$(NMI_CONFFILES),-)

# this makes sure that config files were copied to the ACI rootfs
$(call generate-stamp-rule,$(NMI_COPY_STAMP),$(NMI_CONFFILES) $(NMI_CONF_DEPS_STAMP))

# this removes the config directory in the ACI rootfs, will be invalidated
# when something changes in source config files
$(call generate-rm-dir-rule,$(NMI_CLEAN_DIR_STAMP),$(NMI_CONFDIR))

STAGE1_INSTALL_DIRS_$(NMI_FLAVOR) += $(foreach d,$(NMI_DIR_CHAIN),$d:0755)
STAGE1_INSTALL_FILES_$(NMI_FLAVOR) += $(NMI_TRIPLETS)
STAGE1_SECONDARY_STAMPS_$(NMI_FLAVOR) += $(NMI_COPY_STAMP)

# This filelist of all config files can be generated any time.
$(call generate-shallow-filelist,$(NMI_SRC_CONF_FILELIST),$(NMI_SRC_CONFDIR),$(NMI_SUFFIX))
# This dep.mk can be generated only after the filelist above was
# generated. Will trigger the removal of the config directory in all
# ACI rootfses if source config files change.
$(call generate-glob-deps,$(NMI_CONF_DEPS_STAMP),$(NMI_CLEAN_DIR_STAMP),$(NMI_DEPMK),$(NMI_SUFFIX),$(NMI_SRC_CONF_FILELIST),$(NMI_SRC_CONFDIR),normal)

$(call undefine-namespaces,NMI)
