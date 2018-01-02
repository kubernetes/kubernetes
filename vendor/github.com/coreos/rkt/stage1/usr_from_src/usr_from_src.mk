# This file prepares the initial ACI rootfs contents from systemd
# sources. The libraries required by systemd are taken from the build
# host.

# tmp dir for this flavor
$(call setup-tmp-dir,UFS_TMPDIR)
# directory for systemd stuff (srcdir, builddir, installdir)
UFS_SYSTEMDDIR := $(UFS_TMPDIR)/systemd
# systemd src dir
UFS_SYSTEMD_SRCDIR := $(UFS_SYSTEMDDIR)/src
# systemd build dir
UFS_SYSTEMD_BUILDDIR := $(UFS_SYSTEMDDIR)/build
# systemd install dir, also serves as a temporary rootfs
UFS_ROOTFSDIR := $(UFS_SYSTEMDDIR)/rootfs

# main stamp, makes sure that initial rootfs contents are prepared and
# clean/deps/filelist are generated
$(call setup-stamp-file,UFS_STAMP,systemd)
# stamp for generating initial ACI rootfs contents
$(call setup-stamp-file,UFS_ROOTFS_STAMP,systemd-rootfs)
# stamp for installing systemd into tmp rootfs
$(call setup-stamp-file,UFS_SYSTEMD_INSTALL_STAMP,systemd-install)
# stamp for building systemd
$(call setup-stamp-file,UFS_SYSTEMD_BUILD_STAMP,systemd-build)
# stamp for cloning the git repo and patching it
$(call setup-stamp-file,UFS_SYSTEMD_CLONE_AND_PATCH_STAMP,systemd-clone-and-patch)

# stamp for generating dep file on install/tmp rootfs dir
$(call setup-stamp-file,UFS_ROOTFS_DEPS_STAMP,systemd-rootfs-deps)
# dep.mk for install/tmp rootfs dir contents, will cause ACI rootfs
# removal if they change
$(call setup-dep-file,UFS_ROOTFS_DEPMK,systemd-rootfs)
# install/tmp rootfs dir filelist
$(call setup-filelist-file,UFS_ROOTFSDIR_FILELIST,systemd-rootfs)

# stamp for generating dep file on patches dir
$(call setup-stamp-file,UFS_PATCHES_DEPS_STAMP,systemd-patches-deps)
# dep.mk file for patches dir, will cause git repo reset and clean if
# they change
$(call setup-dep-file,UFS_PATCHES_DEPMK,systemd-patches)
# patches dir filelist
$(call setup-filelist-file,UFS_PATCHES_FILELIST,systemd-patches)

# stamp for removing builddir
$(call setup-stamp-file,UFS_SYSTEMD_RM_BUILDDIR_STAMP,systemd-rm-build)
# stamp for removing installdir
$(call setup-stamp-file,UFS_SYSTEMD_RM_ROOTFSDIR_STAMP,systemd-rm-rootfs)
# stamp for removing ACI rootfsdir
$(call setup-stamp-file,UFS_SYSTEMD_RM_ACIROOTFSDIR_STAMP,systemd-rm-acirootfs)

# We assume that the name passed to --stage1-systemd-version that
# matches a regexp '^v\d+$' (a name starts with a v followed by a
# number, like v211) is a name of a tag. Otherwise it is a branch.
# `expr string : regexp` returns a number of characters that matched
# the regexp, so if that number is equal to the string length then it
# means that the string matched the regexp.
UFS_SYSTEMD_TAG_MATCH := $(shell expr "$(RKT_STAGE1_SYSTEMD_VER)" : 'v[[:digit:]]\+')
UFS_SYSTEMD_TAG_LENGTH := $(shell expr length "$(RKT_STAGE1_SYSTEMD_VER)")
# patches dir
UFS_PATCHES_DIR := $(MK_SRCDIR)/patches/$(RKT_STAGE1_SYSTEMD_VER)
# output file for autogen.sh, used for the truncated verbosity
UFS_AG_OUT := $(UFS_TMPDIR)/ag_out
# systemd configure script
UFS_SYSTEMD_CONFIGURE := $(UFS_SYSTEMD_SRCDIR)/configure
UFS_SYSTEMD_CONFIGURE_AC := $(UFS_SYSTEMD_CONFIGURE).ac

S1_RF_USR_STAMPS += $(UFS_STAMP)
S1_RF_COPY_SO_DEPS := yes
INSTALL_DIRS += \
	$(UFS_SYSTEMDDIR):- \
	$(UFS_SYSTEMD_SRCDIR):- \
	$(UFS_SYSTEMD_BUILDDIR):- \
	$(UFS_ROOTFSDIR):0755
CLEAN_FILES += \
	$(S1_RF_ACIROOTFSDIR)/systemd-version \
	$(UFS_AG_OUT)
CLEAN_DIRS += \
	$(UFS_SYSTEMD_BUILDDIR) \
	$(UFS_ROOTFSDIR)
CLEAN_SYMLINKS += $(S1_RF_ACIROOTFSDIR)/flavor

$(call inc-one,bash.mk)
$(call inc-one,mount.mk)
$(call inc-one,libnss.mk)

# this makes sure everything is done - ACI rootfs is populated,
# clean/deps/filelist are generated
$(call generate-stamp-rule,$(UFS_STAMP),$(UFS_ROOTFS_STAMP) $(UFS_ROOTFS_DEPS_STAMP) $(UFS_PATCHES_DEPS_STAMP))

# this copies the temporary rootfs contents to ACI rootfs and adds
# some misc files
$(call generate-stamp-rule,$(UFS_ROOTFS_STAMP),$(UFS_SYSTEMD_INSTALL_STAMP),$(S1_RF_ACIROOTFSDIR), \
	$(call vb,v2,CP TREE,$(call vsp,$(UFS_ROOTFSDIR)/.) => $(call vsp,$(S1_RF_ACIROOTFSDIR))) \
	cp -af "$(UFS_ROOTFSDIR)/." "$(S1_RF_ACIROOTFSDIR)"; \
	$(call vb,v2,LN SF,src,$(call vsp,$(S1_RF_ACIROOTFSDIR)/flavor)) \
	ln -sf 'src' "$(S1_RF_ACIROOTFSDIR)/flavor"; \
	$(call vb,v2,GEN,$(call vsp,$(S1_RF_ACIROOTFSDIR)/systemd-version)) \
	echo "$(RKT_STAGE1_SYSTEMD_VER)" >"$(S1_RF_ACIROOTFSDIR)/systemd-version")

# this installs systemd into temporary rootfs
$(call generate-stamp-rule,$(UFS_SYSTEMD_INSTALL_STAMP),$(UFS_SYSTEMD_BUILD_STAMP),$(UFS_ROOTFSDIR), \
	$(call vb,v2,INSTALL,systemd) \
	DESTDIR="$(abspath $(UFS_ROOTFSDIR))" $$(MAKE) -C "$(UFS_SYSTEMD_BUILDDIR)" V=0 install-strip $(call vl2,>/dev/null))

# This filelist can be generated only after the installation of
# systemd to temporary rootfs was performed
$(UFS_ROOTFSDIR_FILELIST): $(UFS_SYSTEMD_INSTALL_STAMP)
$(call generate-deep-filelist,$(UFS_ROOTFSDIR_FILELIST),$(UFS_ROOTFSDIR))

# Generate dep.mk file which will cause the initial ACI rootfs to be
# recreated if any file in temporary rootfs changes.
$(call generate-glob-deps,$(UFS_ROOTFS_DEPS_STAMP),$(UFS_SYSTEMD_RM_ACIROOTFSDIR_STAMP),$(UFS_ROOTFS_DEPMK),,$(UFS_ROOTFSDIR_FILELIST),$(UFS_ROOTFSDIR))

# Generate a clean file for cleaning everything that systemd's "make
# install" put in a temporary rootfs directory and for the same files
# in ACI rootfs.
$(call generate-clean-mk-from-filelist, \
	$(UFS_STAMP), \
	$(UFS_ROOTFSDIR_FILELIST), \
	$(UFS_ROOTFSDIR) $(S1_RF_ACIROOTFSDIR), \
	systemd-rootfs-cleanup)

# this builds systemd
$(call generate-stamp-rule,$(UFS_SYSTEMD_BUILD_STAMP),$(UFS_SYSTEMD_CLONE_AND_PATCH_STAMP),$(UFS_SYSTEMD_BUILDDIR), \
	pushd "$(UFS_SYSTEMD_BUILDDIR)" $(call vl3,>/dev/null); \
	$(call vb,v2,CONFIGURE,systemd) \
	"$(abspath $(UFS_SYSTEMD_SRCDIR))/configure" \
		$(call vl3,--quiet) \
		--enable-seccomp \
		--enable-tmpfiles \
		--disable-dbus \
		--disable-kmod \
		--disable-blkid \
		--disable-selinux \
		--disable-pam \
		--disable-acl \
		--disable-smack \
		--disable-gcrypt \
		--disable-elfutils \
		--disable-libcryptsetup \
		--disable-qrencode \
		--disable-microhttpd \
		--disable-gnutls \
		--disable-binfmt \
		--disable-vconsole \
		--disable-quotacheck \
		--disable-randomseed \
		--disable-backlight \
		--disable-rfkill \
		--disable-logind \
		--disable-machined \
		--disable-timedated \
		--disable-timesyncd \
		--disable-localed \
		--disable-coredump \
		--disable-polkit \
		--disable-resolved \
		--disable-networkd \
		--disable-efi \
		--disable-myhostname \
		--disable-manpages \
		--disable-tests \
		--disable-blkid \
		--disable-hibernate \
		--disable-hwdb \
		--disable-importd \
		--disable-firstboot; \
	popd $(call vl3,>/dev/null); \
	$(call vb,v2,BUILD EXT,systemd) \
	$$(MAKE) -C "$(UFS_SYSTEMD_BUILDDIR)" all V=0 $(call vl2,>/dev/null))

# Generate a clean file for a build directory. This can be done only
# after building systemd was finished.
$(call generate-clean-mk-simple, \
	$(UFS_STAMP), \
	$(UFS_SYSTEMD_BUILDDIR), \
	$(UFS_SYSTEMD_BUILDDIR), \
	$(UFS_SYSTEMD_BUILD_STAMP), \
	builddir-cleanup)

# Generate a clean file for systemd's srcdir again. This can be done
# only after building systemd was finished. This is to take some files
# generated during build in the srcdir into account. Normally srcdir
# should be considered read only at this point, but apparently python
# likes to put compiled bytecode next to the src file.
$(call generate-clean-mk-simple, \
	$(UFS_STAMP), \
	$(UFS_SYSTEMD_SRCDIR), \
	$(UFS_SYSTEMD_SRCDIR), \
	$(UFS_SYSTEMD_BUILD_STAMP), \
	srcdir-cleanup-again)

# this stamp makes sure that systemd git repo was cloned and patched
$(call generate-stamp-rule,$(UFS_SYSTEMD_CLONE_AND_PATCH_STAMP),$(UFS_SYSTEMD_CONFIGURE))

# this patches the git repo and generates the configure script
$(call forward-vars,$(UFS_SYSTEMD_CONFIGURE), \
	UFS_PATCHES_DIR GIT UFS_SYSTEMD_SRCDIR UFS_AG_OUT)
$(UFS_SYSTEMD_CONFIGURE):
	$(VQ) \
	set -e; \
	shopt -s nullglob ; \
	if [ -d "$(UFS_PATCHES_DIR)" ]; then \
		for p in "$(abspath $(UFS_PATCHES_DIR))"/*.patch; do \
			$(call vb,v2,PATCH,$${p#$(MK_TOPLEVEL_ABS_SRCDIR)/}) \
			"$(GIT)" -C "$(UFS_SYSTEMD_SRCDIR)" am $(call vl3,--quiet) "$${p}"; \
		done; \
	fi; \
	pushd "$(UFS_SYSTEMD_SRCDIR)" $(call vl3,>/dev/null); \
	$(call vb,v1,BUILD EXT,systemd) \
	$(call vb,v2,AUTOGEN,systemd) \
	$(call v3,./autogen.sh) \
	$(call vl3, \
		if ! ./autogen.sh 2>$(UFS_AG_OUT) >/dev/null; then \
			cat $(UFS_AG_OUT) >&2; \
		fi); \
	popd $(call vl3,>/dev/null)

# Generate the clean file for systemd's srcdir. This can be done only
# after it was cloned, patched and configure script was generated.
$(call generate-clean-mk-simple, \
	$(UFS_STAMP), \
	$(UFS_SYSTEMD_SRCDIR), \
	$(UFS_SYSTEMD_SRCDIR), \
	$(UFS_SYSTEMD_CONFIGURE), \
	srcdir-cleanup)

# Generate a filelist of patches. Can happen anytime.
$(call generate-patches-filelist,$(UFS_PATCHES_FILELIST),$(UFS_PATCHES_DIR))

# Generate a dep.mk on those patches, so if patches change, the
# project should be reset and repatched, and configure script
# regenerated.
$(call generate-glob-deps,$(UFS_PATCHES_DEPS_STAMP),$(UFS_SYSTEMD_CONFIGURE_AC),$(UFS_PATCHES_DEPMK),.patch,$(UFS_PATCHES_FILELIST),$(UFS_PATCHES_DIR),normal)

# parameters for makelib/git.mk
GCL_REPOSITORY := $(RKT_STAGE1_SYSTEMD_SRC)
GCL_DIRECTORY := $(UFS_SYSTEMD_SRCDIR)
GCL_COMMITTISH := $(RKT_STAGE1_SYSTEMD_REV)
GCL_EXPECTED_FILE := $(notdir $(UFS_SYSTEMD_CONFIGURE_AC))
GCL_TARGET := $(UFS_SYSTEMD_CONFIGURE)

ifneq ($(UFS_SYSTEMD_TAG_MATCH),$(UFS_SYSTEMD_TAG_LENGTH))

# If the name is not a tag then we try to pull new changes from upstream.

GCL_DO_CHECK := yes

else

# The name is a tag, so we do not refresh the git repository.

GCL_DO_CHECK :=

endif

include makelib/git.mk

# Remove the build directory if there were some changes in sources
# (like different branch or repository, or a change in patches)
$(UFS_SYSTEMD_RM_BUILDDIR_STAMP): $(UFS_SYSTEMD_CLONE_AND_PATCH_STAMP)
$(generate-rm-dir-rule,$(UFS_SYSTEMD_RM_BUILDDIR_STAMP),$(UFS_SYSTEMD_BUILDDIR))

# Remove the install/tmp rootfs directory if there were some changes
# in build.
$(UFS_SYSTEMD_RM_ROOTFSDIR_STAMP): $(UFS_SYSTEMD_BUILD_STAMP)
$(generate-rm-dir-rule,$(UFS_SYSTEMD_RM_ROOTFSDIR_STAMP),$(UFS_ROOTFSDIR))

# Remove the ACI rootfs if something changes in install.
$(UFS_SYSTEMD_RM_ACIROOTFSDIR_STAMP): $(UFS_SYSTEMD_INSTALL_STAMP)
$(generate-rm-dir-rule,$(UFS_SYSTEMD_RM_ACIROOTFSDIR_STAMP),$(S1_RF_ACIROOTFSDIR))

$(call undefine-namespaces,UFS)
