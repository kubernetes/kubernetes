# Download or reuse the local image to get the squashfs file
# containing CoreOS. The path to the file is saved in CCN_SQUASHFS
# variable.

ifeq ($(_CCN_INCLUDED_),)

_CCN_INCLUDED_ := x

$(call setup-tmp-dir,CCN_TMPDIR)

# systemd version in coreos image
CCN_SYSTEMD_VERSION := v231
# coreos image version
CCN_IMG_RELEASE := 1235.0.0
# coreos image URL
CCN_IMG_URL := https://alpha.release.core-os.net/$(RKT_STAGE1_COREOS_BOARD)/$(CCN_IMG_RELEASE)/coreos_production_pxe_image.cpio.gz
# path to downloaded pxe image
CCN_DOWNLOADED_PXE := $(CCN_TMPDIR)/pxe.img
CLEAN_FILES += \
	$(CCN_DOWNLOADED_PXE) \
	$(CCN_DOWNLOADED_PXE).$(firstword $(shell echo -n $(CCN_IMG_URL) | md5sum)).sig

ifneq ($(RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER),)

CCN_SYSTEMD_VERSION := $(RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER)

endif

# stamp and depmk for invalidating the usr.squashfs file
$(call setup-stamp-file,CCN_INVALIDATE_SQUASHFS_DEPMK_STAMP,invalidate-squashfs)
$(call setup-depmk-file,CCN_INVALIDATE_SQUASHFS_DEPMK,invalidate-squashfs)
# stamp and depmk for invalidating the downloaded pxe.img file
$(call setup-stamp-file,CCN_INVALIDATE_DOWNLOADED_PXE_IMG_DEPMK_STAMP,invalidate-dl-pxe-img)
$(call setup-depmk-file,CCN_INVALIDATE_DOWNLOADED_PXE_IMG_DEPMK,invalidate-dl-pxe-img)

CCN_SQUASHFS_BASE := usr.squashfs
# path to the squashfs file extracted from pxe.img
CCN_SQUASHFS := $(CCN_TMPDIR)/$(CCN_SQUASHFS_BASE)
# path to script performing the downloading and verifying pxe.img
CCN_CACHE_SH := $(MK_SRCDIR)/cache.sh

ifneq ($(RKT_LOCAL_COREOS_PXE_IMAGE_PATH),)

# We are using local pxe.img

CCN_PXE := $(abspath $(RKT_LOCAL_COREOS_PXE_IMAGE_PATH))

else

# We are using downloaded pxe.img

CCN_PXE := $(CCN_DOWNLOADED_PXE)

endif

# This depmk forces the squashfs file recreation if the path to the
# pxe.img file (in the CCN_PXE variable) changes.
$(call generate-kv-deps,$(CCN_INVALIDATE_SQUASHFS_DEPMK_STAMP),$(CCN_SQUASHFS),$(CCN_INVALIDATE_SQUASHFS_DEPMK),CCN_PXE)

# This depmk forces the pxe.img file redownloading if the url to the
# pxe.img file (in the CCN_IMG_URL variable) changes.
$(call generate-kv-deps,$(CCN_INVALIDATE_DOWNLOADED_PXE_IMG_DEPMK_STAMP),$(CCN_DOWNLOADED_PXE_IMG),$(CCN_INVALIDATE_DOWNLOADED_PXE_IMG_DEPMK),CCN_IMG_URL)

CLEAN_FILES += $(CCN_SQUASHFS)

# This extracts the pxe image to get the squashfs file
$(call forward-vars,$(CCN_SQUASHFS), \
	CCN_TMPDIR CCN_PXE CCN_SQUASHFS_BASE)
$(CCN_SQUASHFS): $(CCN_PXE) | $(CCN_TMPDIR)
	$(VQ) \
	$(call vb,vt,EXTRACT,$(call vsp,$(CCN_PXE)) => $(call vsp,$@)) \
	cd "$(CCN_TMPDIR)" && gzip --to-stdout --decompress "$(CCN_PXE)" | cpio $(call vl3,--quiet )--unconditional --extract "$(CCN_SQUASHFS_BASE)"

# This downloads the pxe image
$(call forward-vars,$(CCN_DOWNLOADED_PXE), \
	CCN_TMPDIR CCN_IMG_URL BASH_SHELL CCN_CACHE_SH)
$(CCN_DOWNLOADED_PXE): $(CCN_CACHE_SH) | $(CCN_TMPDIR)
	$(VQ) \
	ITMP="$(CCN_TMPDIR)" IMG_URL="$(CCN_IMG_URL)" V="$(V)" $(BASH_SHELL) $(CCN_CACHE_SH)

# Excluding CCN_SQUASHFS because other will want to know where we
# placed the squashfs file, CCN_SYSTEMD_VERSION might be needed to
# create systemd-version file in ACI rootfs directory.
$(call undefine-namespaces,CCN,CCN_SQUASHFS CCN_SYSTEMD_VERSION)

endif
