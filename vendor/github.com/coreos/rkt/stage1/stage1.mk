# Prepare some variables, this might look strange so here goes: let's
# assume we are going to build two flavors - coreos and kvm. The
# following code is going to initialize some variables to some initial
# values. The name of the variable depends on the flavor:
# STAGE1_COPY_SO_DEPS_coreos, STAGE1_USR_STAMPS_coreos,
# STAGE1_APP_STAMPS_coreos and so on for the coreos flavor. For the
# kvm flavor the names would be STAGE1_COPY_SO_DEPS_kvm,
# STAGE1_USR_STAMPS_kvm, STAGE1_APP_STAMPS_kvm and so on.
#
# STAGE1_COPY_SO_DEPS_$(flavor) tells whether a given flavor wants to
# get missing libraries from build host before building an ACI.
#
# STAGE1_USR_STAMPS_$(flavor) is a list of stamps that say whether the
# initial /usr contents of ACI are prepared.
#
# STAGE1_SECONDARY_STAMPS_$(flavor) is a list of stamps that say
# whether the additional stuff was copied to the ACI rootfs (init,
# enter, units, net config and so on). Usually the dependencies of
# these stamps depend in turn on the stamps from
# STAGE1_USR_STAMPS_$(flavor) - which means that the additional stuff
# is copied to the ACI rootfs only after the initial /usr contents are
# already there.
#
# STAGE1_ACIDIR_$(flavor) tells where is the ACI directory for a given
# flavor. Note that this is the toplevel directory (where manifest is
# stored). Most of the time you will want the
# STAGE1_ACIROOTFSDIR_$(flavor) variable.
#
# STAGE1_ACIROOTFSDIR_$(flavor) tells where is the ACI rootfs
# directory for a given flavor.
#
# STAGE1_ACI_IMAGE_$(flavor) tells where in the build directory the
# final ACI is going to be built.
#
# STAGE1_INSTALL_FILES_$(flavor) - a list of files that can be
# installed only after all the stamps in STAGE1_USR_STAMPS_$(flavor)
# are created.
#
# STAGE1_INSTALL_SYMLINKS_$(flavor) - a list of symlinks that can be
# installed only after all the stamps in STAGE1_USR_STAMPS_$(flavor)
# are created.
#
# STAGE1_INSTALL_DIRS_$(flavor) - a list of directories that can be
# installed only after all the stamps in STAGE1_USR_STAMPS_$(flavor)
# are created.
#
# STAGE1_CREATE_DIRS_$(flavor) - a list of files that can be installed
# only after all the stamps in STAGE1_USR_STAMPS_$(flavor) are
# created. (This is less restrictive than INSTALL_DIRS counterpart,
# because an entry in CREATE_DIRS does not require the parent
# directory to exist). Please consider this as deprecated in favor of
# INSTALL_DIRS + dir-chain function.
#
# STAGE1_ENTER_CMD_$(flavor) - an enter command in stage1 to be used
# for the "rkt enter" command.
#
# STAGE1_STOP_CMD_$(flavor) - a stop command in stage1 to be used
# for the "rkt stop" command.
# 
# STAGE1_IMAGES_$(flavor) - list of stage1 images we want to build for
# each flavor. Need to be parsed here. If empty, single $(STAGE1_ACI_IMAGE_$f)
# will be built. Variables $(STAGE1_ACI_IMAGE_$f-$(image)) will be used
# only for aci build rules.

STAGE1_FLAVORS := $(call commas-to-spaces,$(RKT_STAGE1_ALL_FLAVORS))
STAGE1_BUILT_FLAVORS := $(call commas-to-spaces,$(RKT_STAGE1_FLAVORS))
# filter out the fly flavor - it is special
STAGE1_FLAVORS := $(filter-out fly,$(STAGE1_FLAVORS))
STAGE1_BUILT_FLAVORS := $(filter-out fly,$(STAGE1_BUILT_FLAVORS))

# hypervisors for kvm stage1
STAGE1_BUILT_KVM_HV := $(call commas-to-spaces,$(RKT_STAGE1_KVM_HV))

$(foreach f,$(STAGE1_FLAVORS), \
	$(eval STAGE1_COPY_SO_DEPS_$f :=) \
	$(eval STAGE1_USR_STAMPS_$f :=) \
	$(eval STAGE1_SECONDARY_STAMPS_$f :=) \
	$(eval STAGE1_ACIDIR_$f := $(BUILDDIR)/aci-for-$f-flavor) \
	$(eval STAGE1_ACIROOTFSDIR_$f := $(STAGE1_ACIDIR_$f)/rootfs) \
	$(eval STAGE1_ACI_IMAGE_$f := $(TARGET_BINDIR)/stage1-$f.aci) \
	$(eval STAGE1_INSTALL_FILES_$f :=) \
	$(eval STAGE1_INSTALL_SYMLINKS_$f :=) \
	$(eval STAGE1_INSTALL_DIRS_$f :=) \
	$(eval STAGE1_CREATE_DIRS_$f :=) \
	$(eval STAGE1_ENTER_CMD_$f :=) \
	$(eval STAGE1_STOP_CMD_$f :=))

# Main stamp that tells whether all the ACIs have been built.
$(call setup-stamp-file,_STAGE1_BUILT_ACI_STAMP_,built_aci)

# List of all the ACIs that the build system will build
_STAGE1_ALL_ACI_ := $(foreach f,$(STAGE1_FLAVORS),$(STAGE1_ACI_IMAGE_$f))
_STAGE1_BUILT_ACI_ := $(foreach f,$(STAGE1_BUILT_FLAVORS),$(STAGE1_ACI_IMAGE_$f))

# Assign rules for selected kvm hypervisors, fill STAGE1_IMAGES_kvm variable 
# needed by _STAGE1_BUILT_ACI_ variable and _STAGE1_ACI_RULE_ function
$(foreach h,$(STAGE1_BUILT_KVM_HV), \
	$(eval STAGE1_ACI_IMAGE_kvm-$h := $(TARGET_BINDIR)/stage1-kvm-$h.aci) \
	$(eval STAGE1_IMAGES_kvm += $h))

# Replace stage1-flavor.aci with stage1-flavor-image1.aci,stage1-flavor-image2.aci,...
# for every flavor with more than one image
$(foreach f,$(STAGE1_BUILT_FLAVORS), \
	$(if $(STAGE1_IMAGES_$f), \
		$(eval _STAGE1_BUILT_ACI_ := \
		$(subst $(STAGE1_ACI_IMAGE_$f), $(foreach i,$(STAGE1_IMAGES_$f),$(STAGE1_ACI_IMAGE_$f-$i)), $(_STAGE1_BUILT_ACI_)))))

# The rootfs.mk file takes care of preparing the initial /usr contents
# of the ACI rootfs of a specific flavor. Basically fills
# STAGE1_USR_STAMPS_$(flavor) variables. Might add something to the
# STAGE1_SECONDARY_STAMPS_$(flavor) or the STAGE1_INSTALL_* variables
# too.
$(foreach flavor,$(STAGE1_FLAVORS), \
	$(eval S1_RF_FLAVOR := $(flavor)) \
	$(call inc-one,rootfs.mk))

# secondary-stuff.mk takes care of putting additional stuff into
# the ACI rootfs for each flavor. Basically fills the
# STAGE1_SECONDARY_STAMPS_$(flavor) and the STAGE1_INSTALL_*
# variables.
$(call inc-one,secondary-stuff.mk)

TOPLEVEL_STAMPS += $(_STAGE1_BUILT_ACI_STAMP_)
CLEAN_FILES += $(_STAGE1_ALL_ACI_)

$(call generate-stamp-rule,$(_STAGE1_BUILT_ACI_STAMP_),$(_STAGE1_BUILT_ACI_))

# A rule template for building an ACI. To build the ACI we need to
# have the /usr contents prepared and the additional stuff in place as
# well. If a flavor wants to have missing libraries copied, it is done
# here too. If we want to build more than one image for single flavor 
# (with differences in rootfs - i.e. hypervisor binary for kvm flavor), 
# rootfs common part has to be stored in STAGE1_ACIROOTFSDIR_$1, and 
# additional files in STAGE1_ACIROOTFSDIR_$1_$(image), for each $(image)
# entry from $2.
# 1 - flavor
# 2 - images
define _STAGE1_ACI_RULE_

STAGE1_ALL_STAMPS_$1 := $$(STAGE1_USR_STAMPS_$1) $$(STAGE1_SECONDARY_STAMPS_$1)

$$(STAGE1_SECONDARY_STAMPS_$1): $$(STAGE1_USR_STAMPS_$1)

ifeq ($$(STAGE1_COPY_SO_DEPS_$1),)

# The flavor needs no libraries from the host

$$(STAGE1_ACI_IMAGE_$1): $$(STAGE1_ALL_STAMPS_$1)

# Same dependencies has to be fulfilled for every image in given flavor
$(foreach i,$2, \
$$(eval $$(STAGE1_ACI_IMAGE_$1-$i): $$(STAGE1_ALL_STAMPS_$1)))

else

# The flavor requires libraries from the host, so depend on the stamp
# generated by find-so-deps.mk, which in turn will depend on the usr
# and the secondary stamps.

STAGE1_FSD_FLAVOR := $1

$$(call inc-one,find-so-deps.mk)

$$(STAGE1_ACI_IMAGE_$1): $$(STAGE1_FSD_STAMP)

endif

# The actual rule that builds the ACI. Additional dependencies are
# above.

# Forward variables to rules that build acis. If more than one image,
# forward to every rule.
$$(call forward-vars,$$(STAGE1_ACI_IMAGE_$1), \
	ACTOOL STAGE1_ACIDIR_$1)
$(foreach i,$2, \
$$(call forward-vars,$$(STAGE1_ACI_IMAGE_$1-$i), \
	ACTOOL STAGE1_ACIDIR_$1))

# Prepare targets to build images. Also copy common rootfs to each image directory
# and ensure that all of them will be removed
$(if $(strip $2), \
        $(eval IMG := $(foreach h,$2,$1-$h)) \
        $(foreach s,$(IMG),
                $(call setup-stamp-file,STAGE1_COPY_IMAGE_FILES_$s,/copy-files-$s) \
		$$(STAGE1_COPY_IMAGE_FILES_$s): $$(STAGE1_ALL_STAMPS_$1)
			$(VQ) $(call vb,vt,COPY FILES,$$(call vsp,$s)) cp -rn $(STAGE1_ACIDIR_$1)/rootfs $(STAGE1_ACIDIR_$1)/manifest $(STAGE1_ACIDIR_$1)/$s
		$(eval CLEAN_DIRS += $(STAGE1_ACIDIR_$1)/$s)
		$(call generate-clean-mk-simple,
		        $(STAGE1_ACI_IMAGE_$s),
		        $(STAGE1_ACIDIR_$1)/$s,
		        $(STAGE1_ACIDIR_$1)/$s,
		        $(STAGE1_COPY_IMAGE_FILES_$s),
		        copy-$s)), \
        $(eval IMG := $1))

# Build images. Common files are stored in STAGE1_ACIDIR_$1,
# image specific files in STAGE1_ACIDIR_$1/flavor-image/...
# Add dependency for copying image-specific files, if any ($2 is not empty)
$(foreach s,$(IMG),
$$(STAGE1_ACI_IMAGE_$s): $$(ACTOOL_STAMP) | $$(TARGET_BINDIR) $$(if $(strip $2),$$(STAGE1_COPY_IMAGE_FILES_$s),)
	$(eval ACI_OUTPUT := "$(if $(filter 1,$(words $2)),$(STAGE1_ACI_IMAGE_$1),$(STAGE1_ACI_IMAGE_$s))")
	$(VQ) \
	$(call vb,vt,ACTOOL,$$(call vsp,$(ACI_OUTPUT))) \
	"$$(ACTOOL)" build --overwrite --owner-root "$$(STAGE1_ACIDIR_$1)$(if $(strip $2),/$s,)" $(ACI_OUTPUT)
)

endef

$(foreach f,$(STAGE1_FLAVORS), \
	$(eval $(call _STAGE1_ACI_RULE_,$f,$(STAGE1_IMAGES_$f))))


# The following piece of wizardry takes the variables
# STAGE1_INSTALL_FILES_$(flavor), STAGE1_INSTALL_SYMLINKS_$(flavor),
# STAGE1_INSTALL_DIRS_$(flavor) and STAGE1_CREATE_DIRS_$(flavor),
# makes their contents to depend on STAGE1_USR_STAMPS_$(flavor) and
# appends their contents to, respectively, INSTALL_FILES,
# INSTALL_SYMLINKS, INSTALL_DIRS and CREATE_DIRS. This is to make sure
# that all the additional stuff will be installed only after the
# initial /usr contents in the ACI rootfs are prepared.


# Three fields:
# 1 - base variable name (e.g. INSTALL_FILES)
# 2 - whether it has to be split to get the created entry (elements in
# INSTALL_FILES have to be split, because the created entry is between
# source entry and mode, like src_file:created_entry:mode, elements in
# the CREATE_DIRS variable do not need to be split)
# 3 - if it has to be split, then which item is the created entry
# (index starting from 1)
_STAGE1_FILE_VARS_ := \
	INSTALL_FILES:y:2 \
	INSTALL_SYMLINKS:y:2 \
	INSTALL_DIRS:y:1 \
	CREATE_DIRS:n

# Generates dependency of a created entry on
# STAGE1_USR_STAMPS_$(flavor)
# 1 - item(s)
# 2 - flavor
define _STAGE1_GEN_DEP_
$(call add-dependency,$1,$(STAGE1_USR_STAMPS_$2))
endef

# The actual fixup - appending and "dependencing".
$(foreach v,$(_STAGE1_FILE_VARS_), \
	$(eval _S1_FX_VAR_LIST_ := $(subst :, ,$v)) \
	$(eval _S1_FX_NAME_ := $(word 1,$(_S1_FX_VAR_LIST_))) \
	$(eval _S1_FX_SPLIT_ := $(word 2,$(_S1_FX_VAR_LIST_))) \
	$(eval _S1_FX_IDX_ := $(word 3,$(_S1_FX_VAR_LIST_))) \
	$(foreach f,$(STAGE1_FLAVORS), \
		$(eval _S1_FX_VAR_NAME_ := STAGE1_$(_S1_FX_NAME_)_$f) \
		$(eval $(_S1_FX_NAME_) += $($(_S1_FX_VAR_NAME_))) \
		$(eval $(foreach i,$($(_S1_FX_VAR_NAME_)), \
			$(if $(filter y,$(_S1_FX_SPLIT_)), \
				$(eval _S1_FX_LIST_ := $(subst :, ,$i)) \
				$(eval _S1_FX_ITEM_ := $(word $(_S1_FX_IDX_),$(_S1_FX_LIST_))) \
				$(call _STAGE1_GEN_DEP_,$(_S1_FX_ITEM_),$f), \
				$(call _STAGE1_GEN_DEP_,$i,$f))))) \
	$(call undefine-namespaces,_S1_FX))

$(call undefine-namespaces,STAGE1 _STAGE1)
