$(call setup-stamp-file,LKVM_STAMP)
LKVM_TMPDIR := $(UFK_TMPDIR)/lkvm
LKVM_SRCDIR := $(LKVM_TMPDIR)/src
LKVM_BINARY := $(LKVM_SRCDIR)/lkvm-static
LKVM_ACI_BINARY := $(HV_ACIROOTFSDIR)/lkvm
LKVM_GIT := https://kernel.googlesource.com/pub/scm/linux/kernel/git/will/kvmtool
# just last published version (for reproducible builds), not for any other reason
LKVM_VERSION := cfae4d64482ed745214e3c62dd84b79c2ae0f325

LKVM_STUFFDIR := $(MK_SRCDIR)/lkvm
LKVM_PATCHESDIR := $(LKVM_STUFFDIR)/patches
LKVM_PATCHES := $(abspath $(LKVM_PATCHESDIR)/*.patch)

$(call setup-stamp-file,LKVM_BUILD_STAMP,/build)
$(call setup-stamp-file,LKVM_PATCH_STAMP,/patch_lkvm)
$(call setup-stamp-file,LKVM_DEPS_STAMP,/deps)
$(call setup-dep-file,LKVM_PATCHES_DEPMK)
$(call setup-filelist-file,LKVM_PATCHES_FILELIST,/patches)

S1_RF_SECONDARY_STAMPS += $(LKVM_STAMP)
S1_RF_INSTALL_FILES += $(LKVM_BINARY):$(LKVM_ACI_BINARY):-
INSTALL_DIRS += \
	$(LKVM_SRCDIR):- \
	$(LKVM_TMPDIR):-

$(call generate-stamp-rule,$(LKVM_STAMP),$(LKVM_ACI_BINARY) $(LKVM_DEPS_STAMP))

$(LKVM_BINARY): $(LKVM_BUILD_STAMP)

$(call generate-stamp-rule,$(LKVM_BUILD_STAMP),$(LKVM_PATCH_STAMP),, \
	$(call vb,vt,BUILD EXT,lkvm) \
	$$(MAKE) $(call vl2,--silent) -C "$(LKVM_SRCDIR)" V= lkvm-static $(call vl2,>/dev/null))

# Generate clean file for lkvm directory (this is both srcdir and
# builddir). Can happen after build finished.
$(call generate-clean-mk-simple, \
	$(LKVM_STAMP), \
	$(LKVM_SRCDIR), \
	$(LKVM_SRCDIR), \
	$(LKVM_BUILD_STAMP), \
	cleanup)

$(call generate-stamp-rule,$(LKVM_PATCH_STAMP),,, \
	shopt -s nullglob; \
	for p in $(LKVM_PATCHES); do \
		$(call vb,v2,PATCH,$$$${p#$(MK_TOPLEVEL_ABS_SRCDIR)/}) \
		patch $(call vl3,--silent) --directory="$(LKVM_SRCDIR)" --strip=1 --forward <"$$$${p}"; \
	done)

# Generate a filelist of patches. Can happen anytime.
$(call generate-patches-filelist,$(LKVM_PATCHES_FILELIST),$(LKVM_PATCHESDIR))

# Generate dep.mk on patches, so if they change, the project has to be
# reset to original checkout and patches reapplied.
$(call generate-glob-deps,$(LKVM_DEPS_STAMP),$(LKVM_SRCDIR)/Makefile,$(LKVM_PATCHES_DEPMK),.patch,$(LKVM_PATCHES_FILELIST),$(LKVM_PATCHESDIR),normal)

# parameters for makelib/git.mk
GCL_REPOSITORY := $(LKVM_GIT)
GCL_DIRECTORY := $(LKVM_SRCDIR)
GCL_COMMITTISH := $(LKVM_VERSION)
GCL_EXPECTED_FILE := Makefile
GCL_TARGET := $(LKVM_PATCH_STAMP)
GCL_DO_CHECK :=

include makelib/git.mk

$(call undefine-namespaces,LKVM)
