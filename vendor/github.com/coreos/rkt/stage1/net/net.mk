# Installs configuration files in the ACI rootfs for each flavor

$(foreach flavor,$(STAGE1_FLAVORS), \
	$(eval NMI_FLAVOR := $(flavor)) \
	$(call inc-one,net-install.mk))
