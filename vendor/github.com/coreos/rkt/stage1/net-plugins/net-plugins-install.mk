# This file puts all built plugins in the ACI rootfs for a given
# flavor.
#
# Inputs:
#
# NPI_FLAVOR - flavor for which we will install net plugins
#
# NPI_BUILT_PLUGINS - list of built plugins
#
# NPI_BUILT_STAMPS - list of stamps for built plugins


# stamp telling if all plugins for given flavor are installed
$(call setup-stamp-file,NPI_STAMP,$(NPI_FLAVOR))

# stamp and dep file for invalidating the directory-removing stamp
$(call setup-stamp-file,NPI_BUILT_PLUGINS_KV_DEPMK_STAMP,$(NPI_FLAVOR)-built-plugins-kv-dep)
$(call setup-dep-file,NPI_BUILT_PLUGINS_KV_DEPMK,$(NPI_FLAVOR)-built-plugins-kv-dep)

# stamp for removing the directory with installed plugins in the ACI
# rootfs
$(call setup-stamp-file,NPI_RMDIR_STAMP,$(NPI_FLAVOR)-rmdir)

# base and rest variables for dir-chain
NPI_PLUGINSDIR_BASE := $(STAGE1_ACIROOTFSDIR_$(NPI_FLAVOR))/usr/lib
NPI_PLUGINSDIR_REST := rkt/plugins/net
# plugins directory inside the ACI rootfs
NPI_PLUGINSDIR := $(NPI_PLUGINSDIR_BASE)/$(NPI_PLUGINSDIR_REST)
# list of plugins in the ACI rootfs
NPI_ACI_PLUGINS := $(addprefix $(NPI_PLUGINSDIR)/,$(notdir $(NPI_BUILT_PLUGINS)))
# list of install files triplets (src:dest:mode) for plugins
NPI_INSTALL_FILES_TRIPLETS := $(call install-file-triplets,$(NPI_BUILT_PLUGINS),$(NPI_ACI_PLUGINS),-)
# list of install dir pairs (dir:mode) for plugins dir
NPI_INSTALL_DIRS_PAIRS := $(foreach d,$(call dir-chain,$(NPI_PLUGINSDIR_BASE),$(NPI_PLUGINSDIR_REST)),$d:0755)

# main stamp which makes sure that all the plugins are installed in
# the plugins directory in the ACI rootfs
$(call generate-stamp-rule,$(NPI_STAMP),$(NPI_ACI_PLUGINS))

# this removes the plugins directory
$(call generate-rm-dir-rule,$(NPI_RMDIR_STAMP),$(NPI_PLUGINSDIR))

# invalidate the directory-removing stamp when a list of built plugins
# changes
$(call generate-kv-deps,$(NPI_BUILT_PLUGINS_KV_DEPMK_STAMP),$(NPI_RMDIR_STAMP),$(NPI_BUILT_PLUGINS_KV_DEPMK),NPI_BUILT_PLUGINS)

STAGE1_INSTALL_FILES_$(NPI_FLAVOR) += $(NPI_INSTALL_FILES_TRIPLETS)
STAGE1_INSTALL_DIRS_$(NPI_FLAVOR) += $(NPI_INSTALL_DIRS_PAIRS)
STAGE1_SECONDARY_STAMPS_$(NPI_FLAVOR) += $(NPI_STAMP)

# pairs of plugin in the ACI rootfs and the stamp used to build the
# original
NPI_PLUGIN_STAMP_PAIRS := $(join $(addsuffix :,$(NPI_ACI_PLUGINS)),$(NPI_BUILT_STAMPS))

# make a plugin in the ACI rootfs to depend also on its build stamp
$(foreach p,$(NPI_PLUGIN_STAMP_PAIRS), \
	$(eval NPI_DEP_LIST := $(subst :, ,$p)) \
	$(eval NPI_DEP_PLUGIN := $(word 1,$(NPI_DEP_LIST))) \
	$(eval NPI_DEP_STAMP := $(word 2,$(NPI_DEP_LIST))) \
	$(call add-dependency,$(NPI_DEP_PLUGIN),$(NPI_DEP_STAMP)))

$(call undefine-namespaces,NPI)
