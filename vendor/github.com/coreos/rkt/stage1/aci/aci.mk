$(call setup-tmp-dir,AMM_TMPDIR)

$(foreach flavor,$(STAGE1_FLAVORS), \
	$(eval AMI_FLAVOR := $(flavor)) \
	$(eval AMI_TMPDIR := $(AMM_TMPDIR)/$(flavor)) \
	$(eval INSTALL_DIRS += $(AMI_TMPDIR):-) \
	$(call inc-one,aci-install.mk))

$(call undefine-namespaces,AMM)
