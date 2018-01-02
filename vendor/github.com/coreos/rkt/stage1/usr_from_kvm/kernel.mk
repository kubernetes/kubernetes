# custom kernel compilation
KERNEL_VERSION := 4.9.2
KERNEL_TMPDIR := $(UFK_TMPDIR)/kernel
KERNEL_NAME := linux-$(KERNEL_VERSION)
KERNEL_TARBALL := $(KERNEL_NAME).tar.xz
KERNEL_TARGET_FILE := $(KERNEL_TMPDIR)/$(KERNEL_TARBALL)
KERNEL_SRCDIR := $(KERNEL_TMPDIR)/$(KERNEL_NAME)
KERNEL_BUILDDIR := $(abspath $(KERNEL_TMPDIR)/build-$(KERNEL_VERSION))
KERNEL_URL := https://www.kernel.org/pub/linux/kernel/v4.x/$(KERNEL_TARBALL)
KERNEL_MAKEFILE := $(KERNEL_SRCDIR)/Makefile
KERNEL_STUFFDIR := $(MK_SRCDIR)/kernel
KERNEL_SRC_CONFIG := $(KERNEL_STUFFDIR)/cutdown-config
KERNEL_PATCHESDIR := $(KERNEL_STUFFDIR)/patches
KERNEL_PATCHES := $(abspath $(KERNEL_PATCHESDIR)/*.patch)
KERNEL_BUILD_CONFIG := $(KERNEL_BUILDDIR)/.config
KERNEL_BZIMAGE := $(KERNEL_BUILDDIR)/arch/x86/boot/bzImage
KERNEL_ACI_BZIMAGE := $(S1_RF_ACIROOTFSDIR)/bzImage

$(call setup-stamp-file,KERNEL_STAMP,/build_kernel)
$(call setup-stamp-file,KERNEL_BZIMAGE_STAMP,/bzimage)
$(call setup-stamp-file,KERNEL_PATCH_STAMP,/patch_kernel)
$(call setup-stamp-file,KERNEL_DEPS_STAMP,/deps)
$(call setup-dep-file,KERNEL_PATCHES_DEPMK)
$(call setup-filelist-file,KERNEL_PATCHES_FILELIST,/patches)

CREATE_DIRS += $(KERNEL_TMPDIR) $(KERNEL_BUILDDIR)
CLEAN_DIRS += $(KERNEL_SRCDIR)
INSTALL_FILES += $(KERNEL_SRC_CONFIG):$(KERNEL_BUILD_CONFIG):-
S1_RF_INSTALL_FILES += $(KERNEL_BZIMAGE):$(KERNEL_ACI_BZIMAGE):-
S1_RF_SECONDARY_STAMPS += $(KERNEL_STAMP)
CLEAN_FILES += $(KERNEL_TARGET_FILE)

$(call generate-stamp-rule,$(KERNEL_STAMP),$(KERNEL_ACI_BZIMAGE) $(KERNEL_DEPS_STAMP))

# $(KERNEL_ACI_BZIMAGE) has a dependency on $(KERNEL_BZIMAGE), which
# is actually provided by $(KERNEL_BZIMAGE_STAMP)
$(KERNEL_BZIMAGE): $(KERNEL_BZIMAGE_STAMP)

# This stamp is to make sure that building linux kernel has finished.
$(call generate-stamp-rule,$(KERNEL_BZIMAGE_STAMP),$(KERNEL_BUILD_CONFIG) $(KERNEL_PATCH_STAMP),, \
	$(call vb,vt,BUILD EXT,bzImage) \
	$$(MAKE) $(call vl2,--silent) -C "$(KERNEL_SRCDIR)" O="$(KERNEL_BUILDDIR)" V=0 bzImage $(call vl2,>/dev/null))

# Generate clean file of a builddir. Can happen only after the
# building finished.
$(call generate-clean-mk-simple,\
	$(KERNEL_STAMP), \
	$(KERNEL_BUILDDIR), \
	$(KERNEL_BUILDDIR), \
	$(KERNEL_BZIMAGE_STAMP), \
	builddir-cleanup)

$(call generate-stamp-rule,$(KERNEL_PATCH_STAMP),$(KERNEL_MAKEFILE),, \
	shopt -s nullglob; \
	for p in $(KERNEL_PATCHES); do \
		$(call vb,v2,PATCH,$$$${p#$(MK_TOPLEVEL_ABS_SRCDIR)/}) \
		patch $(call vl3,--silent )--directory="$(KERNEL_SRCDIR)" --strip=1 --forward <"$$$${p}"; \
	done)

# Generate clean file of a srcdir. Can happen after the sources were
# patched.
$(call generate-clean-mk-simple,\
	$(KERNEL_STAMP), \
	$(KERNEL_SRCDIR), \
	$(KERNEL_SRCDIR), \
	$(KERNEL_PATCH_STAMP), \
	srcdir-cleanup)

# Generate a filelist of patches. Can happen anytime.
$(call generate-patches-filelist,$(KERNEL_PATCHES_FILELIST),$(KERNEL_PATCHESDIR))

# Generate a dep.mk on those patches, so if patches change, sources
# are removed, extracted again and repatched.
$(call generate-glob-deps,$(KERNEL_DEPS_STAMP),$(KERNEL_MAKEFILE),$(KERNEL_PATCHES_DEPMK),.patch,$(KERNEL_PATCHES_FILELIST),$(KERNEL_PATCHESDIR),normal)

$(call forward-vars,$(KERNEL_MAKEFILE), \
	KERNEL_SRCDIR KERNEL_TMPDIR)
$(KERNEL_MAKEFILE): $(KERNEL_TARGET_FILE)
	$(VQ) \
	set -e; \
	rm -rf "$(KERNEL_SRCDIR)"; \
	$(call vb,vt,UNTAR,$(call vsp,$<) => $(call vsp,$(KERNEL_TMPDIR))) \
	tar --extract --xz --touch --file="$<" --directory="$(KERNEL_TMPDIR)"

$(call forward-vars,$(KERNEL_TARGET_FILE), \
	KERNEL_URL)
$(KERNEL_TARGET_FILE): | $(KERNEL_TMPDIR)
	$(VQ) \
	$(call vb,vt,WGET,$(KERNEL_URL) => $(call vsp,$@)) \
	wget $(call vl3,--quiet) --tries=20 --output-document="$@" "$(KERNEL_URL)"

$(call undefine-namespaces,KERNEL)
