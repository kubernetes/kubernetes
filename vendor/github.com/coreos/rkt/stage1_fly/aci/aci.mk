$(call setup-stamp-file,FLY_ACI_STAMP,aci-manifest)
$(call setup-tmp-dir,FLY_ACI_TMPDIR_BASE)

FLY_ACI_TMPDIR := $(FLY_ACI_TMPDIR_BASE)/fly
# a manifest template
FLY_ACI_SRC_MANIFEST := $(MK_SRCDIR)/aci-manifest.in
# generated manifest to be copied to the ACI directory
FLY_ACI_GEN_MANIFEST := $(FLY_ACI_TMPDIR)/manifest
# manifest in the ACI directory
FLY_ACI_MANIFEST := $(FLY_ACIDIR)/manifest
# escaped values of the ACI name, version and enter command, so
# they can be safely used in the replacement part of sed's s///
# command.
FLY_ACI_VERSION := $(call sed-replacement-escape,$(RKT_VERSION))
FLY_ACI_ARCH := $(call sed-replacement-escape,$(RKT_ACI_ARCH))
# stamp and dep file for invalidating the generated manifest if name,
# version or enter command changes for this flavor
$(call setup-stamp-file,FLY_ACI_MANIFEST_KV_DEPMK_STAMP,$manifest-kv-dep)
$(call setup-dep-file,FLY_ACI_MANIFEST_KV_DEPMK,manifest-kv-dep)
FLY_ACI_DIRS := \
	$(FLY_ACIROOTFSDIR)/rkt \
	$(FLY_ACIROOTFSDIR)/rkt/status \
	$(FLY_ACIROOTFSDIR)/opt \
	$(FLY_ACIROOTFSDIR)/opt/stage2

# main stamp rule - makes sure manifest and deps files are generated
$(call generate-stamp-rule,$(FLY_ACI_STAMP),$(FLY_ACI_MANIFEST) $(FLY_ACI_MANIFEST_KV_DEPMK_STAMP))

# invalidate generated manifest if version or arch changes
$(call generate-kv-deps,$(FLY_ACI_MANIFEST_KV_DEPMK_STAMP),$(FLY_ACI_GEN_MANIFEST),$(FLY_ACI_MANIFEST_KV_DEPMK),FLY_ACI_VERSION FLY_ACI_ARCH)

# this rule generates a manifest
$(call forward-vars,$(FLY_ACI_GEN_MANIFEST), \
	FLY_ACI_VERSION FLY_ACI_ARCH)
$(FLY_ACI_GEN_MANIFEST): $(FLY_ACI_SRC_MANIFEST) | $(FLY_ACI_TMPDIR) $(FLY_ACI_DIRS) $(FLY_ACIROOTFSDIR)/flavor
	$(VQ) \
	set -e; \
	$(call vb,vt,MANIFEST,fly) \
	sed \
		-e 's/@RKT_STAGE1_VERSION@/$(FLY_ACI_VERSION)/g' \
		-e 's/@RKT_STAGE1_ARCH@/$(FLY_ACI_ARCH)/g' \
	"$<" >"$@.tmp"; \
	$(call bash-cond-rename,$@.tmp,$@)

INSTALL_DIRS += \
	$(FLY_ACI_TMPDIR):- \
	$(foreach d,$(FLY_ACI_DIRS),$d:-)
INSTALL_SYMLINKS += \
	fly:$(FLY_ACIROOTFSDIR)/flavor
FLY_STAMPS += $(FLY_ACI_STAMP)
INSTALL_FILES += \
	$(FLY_ACI_GEN_MANIFEST):$(FLY_ACI_MANIFEST):0644
CLEAN_FILES += $(FLY_ACI_GEN_MANIFEST)

$(call undefine-namespaces,FLY_ACI _FLY_ACI)
