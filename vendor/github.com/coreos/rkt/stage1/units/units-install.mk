# This file puts all the units and symlinks into the ACI rootfs
# directory for given flavor. Also, makes sure that some directores
# are created.
#
# Inputs
#
# UNI_FLAVOR - a flavor

# unit dir in source tree
UNI_SRC_UNITDIR := $(MK_SRCDIR)/units
# all target unit files in source tree
UNI_SRC_TARGET_UNITS := $(wildcard $(UNI_SRC_UNITDIR)/*.target)
# all service unit files in source tree
UNI_SRC_SERVICE_UNITS := $(wildcard $(UNI_SRC_UNITDIR)/*.service)
# all units
UNI_SRC_UNITS := $(UNI_SRC_TARGET_UNITS) $(UNI_SRC_SERVICE_UNITS)
# the following two lists have equal length
# targets for symlinks in the ACI unit directory
UNI_SYMLINK_TARGETS := reboot.target
# filenames of the symlinks in the ACI unit directory
UNI_SYMLINK_FILENAMES := ctrl-alt-del.target

# ACI rootfs
UNI_ACIROOTFSDIR := $(STAGE1_ACIROOTFSDIR_$(UNI_FLAVOR))
# base and rest for units directory in the ACI rootfs
UNI_UNITDIR_BASE := $(UNI_ACIROOTFSDIR)
UNI_UNITDIR_REST := usr/lib/systemd/system
# rests for all the directories we want to have in the ACI rootfs
UNI_UNITDIR_RESTS := \
        $(UNI_UNITDIR_REST) \
        $(UNI_UNITDIR_REST)/default.target.wants \
        $(UNI_UNITDIR_REST)/sockets.target.wants
# main directory for the units
UNI_UNITDIR := $(UNI_UNITDIR_BASE)/$(UNI_UNITDIR_REST)
# all unit directories
UNI_UNIT_DIRS := $(foreach d,$(UNI_UNITDIR_RESTS),$(UNI_UNITDIR_BASE)/$d)
# all symlinks in aci
UNI_SYMLINKS := $(addprefix $(UNI_UNITDIR)/,$(UNI_SYMLINK_FILENAMES))
# all units in aci
UNI_UNITS := $(addprefix $(UNI_UNITDIR)/,$(notdir $(UNI_SRC_UNITS)))
# all directory chains in aci
UNI_DIR_CHAINS := $(sort $(foreach r,$(UNI_UNITDIR_RESTS),$(call dir-chain,$(UNI_UNITDIR_BASE),$r)))
# stuff for INSTALL_* variables
UNI_SYMLINK_PAIRS := $(join $(addsuffix :,$(UNI_SYMLINK_TARGETS)),$(UNI_SYMLINKS))
UNI_INSTALL_FILES_TRIPLETS := $(call install-file-triplets,$(UNI_SRC_UNITS),$(UNI_UNITS),0644)
UNI_INSTALL_DIRS_PAIRS := $(addsuffix :0755,$(UNI_DIR_CHAINS))

# This stamps make sure that all stuff were copied and deps were
# generated
$(call setup-stamp-file,UNI_STAMP,$(UNI_FLAVOR)-main)
# This stamp makes sure that directories and symlinks were created and
# files were copied.
$(call setup-stamp-file,UNI_COPY_STAMP,$(UNI_FLAVOR)-copy)
# This stamp makes sure that the unit dir is removed in case something
# changes in units
$(call setup-stamp-file,UNI_REMOVE_ACIROOTFSDIR_STAMP,$(UNI_FLAVOR)-remove-acirootfs)

# stamp, filelist and a dep file for generating dependencies on target
# files
$(call setup-stamp-file,UNI_UNIT_TARGET_DEPS_STAMP,$(UNI_FLAVOR)-target-deps)
$(call setup-filelist-file,UNI_UNIT_TARGET_FILELIST,$(UNI_FLAVOR)-target)
$(call setup-dep-file,UNI_UNIT_TARGET_DEPMK,$(UNI_FLAVOR)-target)

# stamp, filelist and a dep file for generating dependencies on
# service files
$(call setup-stamp-file,UNI_UNIT_SERVICE_DEPS_STAMP,$(UNI_FLAVOR)-service-deps)
$(call setup-filelist-file,UNI_UNIT_SERVICE_FILELIST,$(UNI_FLAVOR)-service)
$(call setup-dep-file,UNI_UNIT_SERVICE_DEPMK,$(UNI_FLAVOR)-service)

# stamp and a dep file for generating dependencies on symlinks
$(call setup-stamp-file,UNI_UNIT_SYMLINKS_KV_DEPS_STAMP,$(UNI_FLAVOR)-symlinks-kv-deps)
$(call setup-dep-file,UNI_UNIT_SYMLINKS_KV_DEPMK,$(UNI_FLAVOR)-symlinks-kv)

# this stamp makes sure that everything was copied and deps were
# generated
$(call generate-stamp-rule,$(UNI_STAMP),$(UNI_COPY_STAMP) $(UNI_UNIT_TARGET_DEPS_STAMP) $(UNI_UNIT_SERVICE_DEPS_STAMP) $(UNI_UNIT_SYMLINKS_KV_DEPS_STAMP))

# this stamp makes sure that everything is copied
$(call generate-stamp-rule,$(UNI_COPY_STAMP),$(UNI_UNITS),$(UNI_SYMLINKS) $(UNI_UNIT_DIRS))

# this removes the ACI rootfs directory
$(call generate-rm-dir-rule,$(UNI_REMOVE_ACIROOTFSDIR_STAMP),$(UNI_ACIROOTFSDIR))

STAGE1_INSTALL_FILES_$(UNI_FLAVOR) += $(UNI_INSTALL_FILES_TRIPLETS)
STAGE1_INSTALL_DIRS_$(UNI_FLAVOR) += $(UNI_INSTALL_DIRS_PAIRS)
STAGE1_INSTALL_SYMLINKS_$(UNI_FLAVOR) += $(UNI_SYMLINK_PAIRS)
STAGE1_SECONDARY_STAMPS_$(UNI_FLAVOR) += $(UNI_STAMP)

# Below dep.mk files are generated that may nuke the entire ACI rootfs
# directory. We have to do it that way, because the unit directory in
# the ACI rootfs may contain files that came from the initial ACI
# rootfs preparation. So we might need to recreate the tree before
# putting our units again.

# This filelist of all target unit files can be generated any time.
$(call generate-shallow-filelist,$(UNI_UNIT_TARGET_FILELIST),$(UNI_SRC_UNITDIR),.target)
# This dep.mk can be generated only after the filelist above was
# generated. Will trigger removing of the ACI rootfs directory in all
# ACI rootfses if target unit files change.
$(call generate-glob-deps,$(UNI_UNIT_TARGET_DEPS_STAMP),$(UNI_REMOVE_ACIROOTFSDIR_STAMP),$(UNI_UNIT_TARGET_DEPMK),.target,$(UNI_UNIT_TARGET_FILELIST),$(UNI_SRC_UNITDIR),normal)

# This filelist of all service unit files can be generated any time.
$(call generate-shallow-filelist,$(UNI_UNIT_SERVICE_FILELIST),$(UNI_SRC_UNITDIR),.service)
# This dep.mk can be generated only after the filelist above was
# generated. Will trigger removing of the ACI rootfs directory in all
# ACI rootfses if service unit files change.
$(call generate-glob-deps,$(UNI_UNIT_SERVICE_DEPS_STAMP),$(UNI_REMOVE_ACIROOTFSDIR_STAMP),$(UNI_UNIT_SERVICE_DEPMK),.service,$(UNI_UNIT_SERVICE_FILELIST),$(UNI_SRC_UNITDIR),normal)

# This dep.mk will force the removal of the ACI rootfs directory if
# any symlink to be created changes.
$(call generate-kv-deps,$(UNI_UNIT_SYMLINKS_KV_DEPS_STAMP),$(UNI_REMOVE_ACIROOTFSDIR_STAMP),$(UNI_UNIT_SYMLINKS_KV_DEPMK),UNI_SYMLINK_TARGETS UNI_SYMLINK_FILENAMES)

$(call undefine-namespaces,UNI)
