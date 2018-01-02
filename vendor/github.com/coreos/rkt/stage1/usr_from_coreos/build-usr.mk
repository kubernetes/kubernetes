# This file takes the following inputs and prepares the initial
# contents in a given ACI rootfs directory based on manifests in a
# given manifest directory.
#
# INPUTS:
#
# CBU_MANIFESTS_DIR - a directory with .manifest files for
# unsquashfs. These files are simply a list of files to be extracted
# from squashfs image.
#
# CBU_TMPDIR - a path to directory, which this Makefile can use for
# its own purposes.
#
# CBU_DIFF - a differentiator for stamps, filelists, etc. Required
# because this Makefile can be used by several flavors to build ACI
# rootfs (coreos, kvm). So it is used to differentiate build system
# files from each user/flavor. Needs to be unique across the entire
# build system.
#
# CBU_STAMP - a stamp which will tell that the basic contents of ACI
# rootfs are prepared.
#
# CBU_ACIROOTFSDIR - a directory of an ACI rootfs.
#
# CBU_FLAVOR - used for the flavor symlink in ACI rootfs.

$(call inc-one,coreos-common.mk)

# This will hold all the files taken from the squashfs file according
# to given manifests.
CBU_ROOTFS := $(CBU_TMPDIR)/rootfs
# This is just all manifests in CBU_MANIFESTS_DIR concatenated, sorted
# and uniquefied.
CBU_COMPLETE_MANIFEST := $(CBU_TMPDIR)/manifest.txt
# All manifest in the CBU_MANIFESTS_DIR
CBU_MANIFESTS := $(wildcard $(CBU_MANIFESTS_DIR)/*)
# A list of all files in the squashfs file without the likely
# "squashfs-root" prefix
CBU_SQUASHFS_FILES := $(CBU_TMPDIR)/squashfsfiles
# A list of files that appear in both CBU_COMPLETE_MANIFEST and
# CBU_SQUASHFS_FILES, should be the same as the list in
# CBU_COMPLETE_MANIFEST, otherwise we get an error.
CBU_COMMON_FILES := $(CBU_TMPDIR)/commonfiles

# Stamp telling when ACI rootfs was prepared.
$(call setup-stamp-file,CBU_ACI_ROOTFS_STAMP,$(CBU_DIFF)-acirootfs)
# Stamp telling when tmp rootfs was copied to ACI rootfs.
$(call setup-stamp-file,CBU_ROOTFS_COPY_STAMP,$(CBU_DIFF)-rootfs-copy)
# Stamp telling when squashfs file was unpacked using the complete
# manifest.
$(call setup-stamp-file,CBU_MKBASE_STAMP,$(CBU_DIFF)-mkbase)

# Stamp and dep file for generating dependencies on manifest files.
$(call setup-stamp-file,CBU_MANIFEST_DEPS_STAMP,$(CBU_DIFF)-manifest-deps)
$(call setup-dep-file,CBU_MANIFEST_DEPMK,$(CBU_DIFF)-manifest)

# Stamp and dep file for generating dependencies on tmp rootfs.
$(call setup-stamp-file,CBU_TMPROOTFS_DEPMK_STAMP,$(CBU_DIFF)-tmprootfs-deps)
$(call setup-dep-file,CBU_TMPROOTFS_DEPMK,$(CBU_DIFF)-tmprootfs)

# Filelist for stuff taken from squashfs - it is more detailed when
# compared to complete manifest.
$(call setup-filelist-file,CBU_DETAILED_FILELIST,$(CBU_DIFF)-acirootfs)
# Filelist for all manifest files.
$(call setup-filelist-file,CBU_ALL_MANIFESTS_FILELIST,$(CBU_DIFF)-manifests)

# Stamps for removing outdated contents of either ACI rootfs or tmp
# rootfs.
$(call setup-stamp-file,CBU_REMOVE_ACIROOTFSDIR_STAMP,$(CBU_DIFF)-remove-acirootfs)
$(call setup-stamp-file,CBU_REMOVE_TMPROOTFSDIR_STAMP,$(CBU_DIFF)-remove-tmprootfs)

# Stamp and dep file for generating dependencies on a list of
# symlinks.
$(call setup-stamp-file,CBU_ACIROOTFS_SYMLINKS_KV_DEPMK_STAMP,$(CBU_DIFF)-acirootfs-symlinks)
$(call setup-dep-file,CBU_ACIROOTFS_SYMLINKS_KV_DEPMK,$(CBU_DIFF)-acirootfs-symlinks-kv)

# Stamp and dep file for generating dependencies on a systemd version.
$(call setup-stamp-file,CBU_ACIROOTFS_SYSTEMD_VERSION_KV_DEPMK_STAMP,$(CBU_DIFF)-systemd-version)
$(call setup-dep-file,CBU_ACIROOTFS_SYSTEMD_VERSION_KV_DEPMK,$(CBU_DIFF)-systemd-version-kv)

# All stamps in this file that generate deps files
CBU_DEPS_STAMPS := \
	$(CBU_TMPROOTFS_DEPMK_STAMP) \
	$(CBU_MANIFEST_DEPS_STAMP) \
	$(CBU_ACIROOTFS_SYMLINKS_KV_DEPMK_STAMP) \
	$(CBU_ACIROOTFS_SYSTEMD_VERSION_KV_DEPMK_STAMP)

# All symlinks to be created in ACI rootfs
CBU_ACIROOTFS_SYMLINKS := \
	$(CBU_ACIROOTFSDIR)/flavor \
	$(CBU_ACIROOTFSDIR)/lib64 \
	$(CBU_ACIROOTFSDIR)/lib \
	$(CBU_ACIROOTFSDIR)/bin
CBU_SYSTEMD_VERSION_FILE := $(CBU_ACIROOTFSDIR)/systemd-version

CLEAN_FILES += \
	$(CBU_COMPLETE_MANIFEST) \
	$(CBU_SYSTEMD_VERSION_FILE) \
	$(CBU_SQUASHFS_FILES) \
	$(CBU_COMMON_FILES)
INSTALL_DIRS += \
	$(CBU_ROOTFS):0755
INSTALL_SYMLINKS += \
	$(CBU_FLAVOR):$(CBU_ACIROOTFSDIR)/flavor \
	usr/lib64:$(CBU_ACIROOTFSDIR)/lib64 \
	usr/lib:$(CBU_ACIROOTFSDIR)/lib \
	usr/bin:$(CBU_ACIROOTFSDIR)/bin


# The main stamp - makes sure that ACI rootfs directory is prepared
# with initial contents and all deps/clean files are generated.
$(call generate-stamp-rule,$(CBU_STAMP),$(CBU_ACI_ROOTFS_STAMP) $(CBU_DEPS_STAMPS))

# This stamp makes sure that ACI rootfs is fully populated - stuff is
# copied, symlinks and systemd-version file are created.
$(call generate-stamp-rule,$(CBU_ACI_ROOTFS_STAMP),$(CBU_ROOTFS_COPY_STAMP) $(CBU_SYSTEMD_VERSION_FILE),$(CBU_ACIROOTFS_SYMLINKS))

# This generates the systemd-version file in ACI rootfs.
$(call forward-vars,$(CBU_SYSTEMD_VERSION_FILE), \
	CCN_SYSTEMD_VERSION CBU_SYSTEMD_VERSION_FILE)
$(CBU_SYSTEMD_VERSION_FILE): $(CBU_ROOTFS_COPY_STAMP)
	$(VQ) \
	$(call vb,v2,GEN,$(call vsp,$(CBU_SYSTEMD_VERSION_FILE))) \
	echo "$(CCN_SYSTEMD_VERSION)" >"$(CBU_SYSTEMD_VERSION_FILE)"

# This depmk forces systemd-version file recreation if systemd version
# (in CCN_SYSTEMD_VERSION variable) changes.
$(call generate-kv-deps,$(CBU_ACIROOTFS_SYSTEMD_VERSION_KV_DEPMK_STAMP),$(CBU_SYSTEMD_VERSION_FILE),$(CBU_ACIROOTFS_SYSTEMD_VERSION_KV_DEPMK),CCN_SYSTEMD_VERSION)

# Create the symlinks after the tmp rootfs was copied to ACI rootfs
$(CBU_ACIROOTFS_SYMLINKS): $(CBU_ROOTFS_COPY_STAMP)

# This copies tmp rootfs to ACI rootfs
$(call generate-stamp-rule,$(CBU_ROOTFS_COPY_STAMP),$(CBU_MKBASE_STAMP),$(CBU_ACIROOTFSDIR), \
	$(call vb,v2,CP TREE,$(call vsp,$(CBU_ROOTFS)/.) => $(call vsp,$(CBU_ACIROOTFSDIR))) \
	cp -af "$(CBU_ROOTFS)/." "$(CBU_ACIROOTFSDIR)")

# This removes the ACI rootfs directory if it holds outdated initial
# contents (mostly happens when squashfs changes).
$(CBU_REMOVE_ACIROOTFSDIR_STAMP): $(CBU_MKBASE_STAMP)
$(call generate-rm-dir-rule,$(CBU_REMOVE_ACIROOTFSDIR_STAMP),$(CBU_ACIROOTFSDIR))

# This depmk forces the removal of ACI rootfs dir and its repopulation
# if any of the symlinks to be created by the stamp populating ACI
# rootfs dir changes.
$(call generate-kv-deps,$(CBU_ACIROOTFS_SYMLINKS_KV_DEPMK_STAMP),$(CBU_REMOVE_ACIROOTFSDIR_STAMP) $(CBU_ROOTFS_COPY_STAMP),$(CBU_ACIROOTFS_SYMLINKS_KV_DEPMK),CBU_ACIROOTFS_SYMLINKS)

# This depmk can be created only when detailed filelist is
# generated. It will invalidate ACI rootfs creation if contents of
# temporary rootfs change.
$(call generate-glob-deps,$(CBU_TMPROOTFS_DEPMK_STAMP),$(CBU_REMOVE_ACIROOTFSDIR_STAMP),$(CBU_TMPROOTFS_DEPMK),,$(CBU_DETAILED_FILELIST),$(CBU_ROOTFS))

# Generate clean file for files put in the ACI rootfs and in the
# temporary rootfs.
$(call generate-clean-mk-from-filelist, \
	$(CBU_STAMP), \
	$(CBU_DETAILED_FILELIST), \
	$(CBU_ACIROOTFSDIR) $(CBU_ROOTFS), \
	$(CBU_DIFF)-rootfs-cleanup)

# This unpacks squashfs image to a temporary rootfs.
$(call generate-stamp-rule,$(CBU_MKBASE_STAMP),$(CCN_SQUASHFS) $(CBU_COMPLETE_MANIFEST),$(CBU_ROOTFS), \
	$(call vb,vt,UNSQUASHFS,$(call vsp,$(CCN_SQUASHFS)) => $(call vsp,$(CBU_ROOTFS)/usr)) \
	CBU_SQROOT=$$$$(unsquashfs -ls "$(CCN_SQUASHFS)" --no-progress | tail --lines=1); \
	unsquashfs -ls "$(CCN_SQUASHFS)" | grep "^$$$${CBU_SQROOT}" | sed -e "s/$$$${CBU_SQROOT}\///g" | sort >"$(CBU_SQUASHFS_FILES)"; \
	comm -1 -2 "$(CBU_SQUASHFS_FILES)" "$(CBU_COMPLETE_MANIFEST)" >"$(CBU_COMMON_FILES)"; \
	if ! cmp --silent "$(CBU_COMMON_FILES)" "$(CBU_COMPLETE_MANIFEST)"; \
	then \
		echo -e "Files listed in $(CBU_COMPLETE_MANIFEST) are missing from $(CCN_SQUASHFS):\n$$$$(comm -1 -3 "$(CBU_SQUASHFS_FILES)" "$(CBU_COMPLETE_MANIFEST)")"; \
		exit 1; \
	fi; \
	unsquashfs -dest "$(CBU_ROOTFS)/usr" -ef "$(CBU_COMPLETE_MANIFEST)" "$(CCN_SQUASHFS)"$(call vl3, >/dev/null))

# If either squashfs file or the concatenated manifest file changes we
# need to unpack the squashfs file again. Clean the directory holding
# the old contents beforehand.
$(CBU_REMOVE_TMPROOTFSDIR_STAMP): $(CCN_SQUASHFS) $(CBU_COMPLETE_MANIFEST)
$(call generate-rm-dir-rule,$(CBU_REMOVE_TMPROOTFSDIR_STAMP),$(CBU_ROOTFS))

# This filelist can be generated only after the pxe image was
# unsquashed to a temporary rootfs.
$(CBU_DETAILED_FILELIST): $(CBU_MKBASE_STAMP)
$(call generate-deep-filelist,$(CBU_DETAILED_FILELIST),$(CBU_ROOTFS))

# This concatenates all manifests into one file.
$(CBU_COMPLETE_MANIFEST): $(CBU_MANIFESTS) | $(CBU_TMPDIR)
	$(VQ) \
	set -e; \
	$(call vb,v2,GEN,$(call vsp,$@)) \
	cat $^ | sort -u > "$@.tmp"; \
	$(call bash-cond-rename,$@.tmp,$@)

# This filelist can be generated anytime.
$(call generate-shallow-filelist,$(CBU_ALL_MANIFESTS_FILELIST),$(CBU_MANIFESTS_DIR),.manifest)

# This depmk can be created only when filelist of manifests is
# generated. If any of the file changes or is deleted or new one is
# added, it will invalidate CBU_COMPLETE_MANIFEST, which in turn will
# invalidate squashfs unpacking, which in turn will invalidate aci
# rootfs directory creation.
$(call generate-glob-deps,$(CBU_MANIFEST_DEPS_STAMP),$(CBU_COMPLETE_MANIFEST),$(CBU_MANIFEST_DEPMK),.manifest,$(CBU_ALL_MANIFESTS_FILELIST),$(CBU_MANIFESTS_DIR),normal)

$(call undefine-namespaces,CBU)
