# Installs unit files in the ACI rootfs for each flavor

$(foreach flavor,$(STAGE1_FLAVORS), \
	$(eval UNI_FLAVOR := $(flavor)) \
	$(call inc-one,units-install.mk))
