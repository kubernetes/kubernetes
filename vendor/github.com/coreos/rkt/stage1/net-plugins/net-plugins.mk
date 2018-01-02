# This file has two parts:
#
# 1. Build the plugins
#
# 2. Install plugins in the ACI rootfs for each flavor


# 1.

# plugin names - taken from github.com/containernetworking/cni/plugins
NPM_PLUGIN_NAMES := \
	main/ptp \
	main/bridge \
	main/macvlan \
	main/ipvlan \
	ipam/host-local \
	ipam/dhcp \
	meta/flannel \
	meta/tuning

# both lists below have the same number of elements
# array of path to built plugins
NPM_BUILT_PLUGINS :=
# array of stamps used to build the plugins
NPM_BUILT_STAMPS :=

# Generates a build rule for a given plugin name
# 1 - plugin name (like main/ptp or ipam/dhcp)
define NPM_GENERATE_BUILD_PLUGIN_RULE
# base name of a plugin
NPM_BASE := $$(notdir $1)
# path to the built plugin
NPM_PLUGIN := $$(TARGET_TOOLSDIR)/$$(NPM_BASE)

# stamp used to build a plugin
$$(call setup-stamp-file,NPM_STAMP,$$(NPM_BASE))

# variables for makelib/build_go_bin.mk
BGB_STAMP := $$(NPM_STAMP)
BGB_BINARY := $$(NPM_PLUGIN)
BGB_PKG_IN_REPO := vendor/github.com/containernetworking/cni/plugins/$1
include makelib/build_go_bin.mk

$$(NPM_PLUGIN): | $(TARGET_TOOLSDIR)

$$(call generate-stamp-rule,$$(NPM_STAMP))

CLEAN_FILES += $$(NPM_PLUGIN)
NPM_BUILT_PLUGINS += $$(NPM_PLUGIN)
NPM_BUILT_STAMPS += $$(NPM_STAMP)
endef

$(foreach p,$(NPM_PLUGIN_NAMES), \
        $(eval $(call NPM_GENERATE_BUILD_PLUGIN_RULE,$p)))


# 2.

$(foreach flavor,$(STAGE1_FLAVORS), \
	$(eval NPI_FLAVOR := $(flavor)) \
	$(eval NPI_BUILT_PLUGINS := $(NPM_BUILT_PLUGINS)) \
	$(eval NPI_BUILT_STAMPS := $(NPM_BUILT_STAMPS)) \
	$(call inc-one,net-plugins-install.mk))

$(call undefine-namespaces,NPM)
