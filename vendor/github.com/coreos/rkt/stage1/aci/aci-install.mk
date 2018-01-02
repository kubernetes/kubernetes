# Here we create a set of directories inside the ACI rootfs directory,
# generate a manifest and copy it along with the os-release file to the
# ACI rootfs directory.
#
# AMI_FLAVOR - flavor for which directories and manifest should be
# created and copied along with the os-release file into the ACI
# rootfs directory.
#
# AMI_TMPDIR - a directory where this file can put its temporary files

# the ACI rootfs for this flavor
AMI_ACIROOTFSDIR := $(STAGE1_ACIROOTFSDIR_$(AMI_FLAVOR))
# dirs base and rests
AMI_ACI_DIRS_BASE := $(AMI_ACIROOTFSDIR)
AMI_ACI_DIRS_RESTS := \
	etc \
	opt/stage2 \
	rkt/iottymux \
	rkt/status \
	rkt/env
# all the directories we want to be created in the ACI rootfs
AMI_ACI_DIR_CHAINS := \
	$(foreach r,$(AMI_ACI_DIRS_RESTS), \
		$(call dir-chain,$(AMI_ACI_DIRS_BASE),$r))
# all final directories to be created
AMI_ACI_INSTALLED_DIRS := $(addprefix $(AMI_ACI_DIRS_BASE)/,$(AMI_ACI_DIRS_RESTS))
# os-release in /etc in the ACI rootfs
AMI_ACI_OS_RELEASE := $(AMI_ACI_DIRS_BASE)/etc/os-release
# manifest in the ACI directory
AMI_ACI_MANIFEST := $(STAGE1_ACIDIR_$(AMI_FLAVOR))/manifest
# generated manifest to be copied to the ACI directory
AMI_GEN_MANIFEST := $(AMI_TMPDIR)/aci-manifest-$(AMI_FLAVOR)
# a manifest template
AMI_SRC_MANIFEST := $(MK_SRCDIR)/aci-manifest.in
# a name for this flavor
AMI_NAME := coreos.com/rkt/stage1-$(AMI_FLAVOR)
# list of installed files and symlinks
AMI_INSTALLED_FILES := \
	$(AMI_ACI_OS_RELEASE) \
	$(AMI_ACI_MANIFEST) \
	$(AMI_ACI_DIRS_BASE)/etc/mtab

ifeq ($(AMI_FLAVOR),src)

# src flavor has a slightly different rule about its name - we append
# the systemd version to it too

AMI_NAME := $(AMI_NAME)-$(RKT_STAGE1_SYSTEMD_VER)

endif

# stage1 version, usually the same as rkt version, unless we override
# it with something else
AMI_STAGE1_VERSION := $(RKT_VERSION)

ifneq ($(RKT_STAGE1_VERSION_OVERRIDE),)

AMI_STAGE1_VERSION := $(RKT_STAGE1_VERSION_OVERRIDE)

endif

# escaped values of the ACI name, version and enter command, so
# they can be safely used in the replacement part of sed's s///
# command.
AMI_SED_NAME := $(call sed-replacement-escape,$(AMI_NAME))
AMI_SED_VERSION := $(call sed-replacement-escape,$(AMI_STAGE1_VERSION))
AMI_SED_ENTER := $(call sed-replacement-escape,$(STAGE1_ENTER_CMD_$(AMI_FLAVOR)))
AMI_SED_ARCH := $(call sed-replacement-escape,$(RKT_ACI_ARCH))
AMI_SED_STOP := $(call sed-replacement-escape,$(STAGE1_STOP_CMD_$(AMI_FLAVOR)))

# main stamp ensures everything is done
$(call setup-stamp-file,AMI_STAMP,$(AMI_FLAVOR)-main)

# stamp and dep file for invalidating the generated manifest if name,
# version or enter command changes for this flavor
$(call setup-stamp-file,AMI_MANIFEST_KV_DEPMK_STAMP,$(AMI_FLAVOR)-manifest-kv-dep)
$(call setup-dep-file,AMI_MANIFEST_KV_DEPMK,$(AMI_FLAVOR)-manifest-kv-dep)

# stamp and dep file for invalidating the ACI rootfs removing stamp if
# the list of installed files or directories change
$(call setup-stamp-file,AMI_INSTALLED_KV_DEPMK_STAMP,$(AMI_FLAVOR)-installed-kv-dep)
$(call setup-dep-file,AMI_INSTALLED_KV_DEPMK,$(AMI_FLAVOR)-installed-kv-dep)

# stamp for removing the ACI rootfs if installed files/directories change
$(call setup-stamp-file,AMI_RMDIR_STAMP,$(AMI_FLAVOR)-acirootfs-remove)

# main stamp rule - makes sure that os-release, manifest and specific
# directories are inside the ACI rootfs and deps file are generated
$(call generate-stamp-rule,$(AMI_STAMP),$(AMI_INSTALLED_FILES) $(AMI_MANIFEST_KV_DEPMK_STAMP) $(AMI_INSTALLED_KV_DEPMK_STAMP),$(AMI_ACI_INSTALLED_DIRS))

# this rule generates a manifest
$(call forward-vars,$(AMI_GEN_MANIFEST), \
	AMI_FLAVOR AMI_SED_NAME AMI_SED_VERSION AMI_SED_ENTER AMI_SED_ARCH AMI_SED_STOP)
$(AMI_GEN_MANIFEST): $(AMI_SRC_MANIFEST) | $(AMI_TMPDIR)
	$(VQ) \
	set -e; \
	$(call vb,vt,MANIFEST,$(AMI_FLAVOR)) \
	sed \
		-e 's/@RKT_STAGE1_NAME@/$(AMI_SED_NAME)/g' \
		-e 's/@RKT_STAGE1_VERSION@/$(AMI_SED_VERSION)/g' \
		-e 's/@RKT_STAGE1_ENTER@/$(AMI_SED_ENTER)/g' \
		-e 's/@RKT_STAGE1_ARCH@/$(AMI_SED_ARCH)/g' \
		-e 's/@RKT_STAGE1_STOP@/$(AMI_SED_STOP)/g' \
	"$<" >"$@.tmp"; \
	$(call bash-cond-rename,$@.tmp,$@)

# invalidate generated manifest if name, version, arch, enter or stop cmd changes
$(call generate-kv-deps,$(AMI_MANIFEST_KV_DEPMK_STAMP),$(AMI_GEN_MANIFEST),$(AMI_MANIFEST_KV_DEPMK),AMI_SED_NAME AMI_SED_VERSION AMI_SED_ARCH AMI_SED_STOP AMI_SED_ENTER)

# this removes the ACI rootfs dir
$(call generate-rm-dir-rule,$(AMI_RMDIR_STAMP),$(AMI_ACIROOTFSDIR))

# invalidate the ACI rootfs removing stamp if installed files change
$(call generate-kv-deps,$(AMI_INSTALLED_KV_DEPMK_STAMP),$(AMI_RMDIR_STAMP),$(AMI_INSTALLED_KV_DEPMK),AMI_INSTALLED_FILES AMI_ACI_INSTALLED_DIRS)

STAGE1_INSTALL_DIRS_$(AMI_FLAVOR) += $(addsuffix :0755,$(AMI_ACI_DIR_CHAINS))
STAGE1_INSTALL_FILES_$(AMI_FLAVOR) += \
	$(AMI_GEN_MANIFEST):$(AMI_ACI_MANIFEST):0644 \
	$(MK_SRCDIR)/os-release:$(AMI_ACI_OS_RELEASE):0644
STAGE1_INSTALL_SYMLINKS_$(AMI_FLAVOR) += \
	../proc/self/mounts:$(AMI_ACI_DIRS_BASE)/etc/mtab
STAGE1_SECONDARY_STAMPS_$(AMI_FLAVOR) += $(AMI_STAMP)
CLEAN_FILES += $(AMI_GEN_MANIFEST)

$(call undefine-namespaces,AMI)
