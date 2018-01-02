# inputs cleared after including this file:
# BGB_STAMP
# BGB_BINARY
# BGB_GO_FLAGS
# BGB_PKG_IN_REPO
# BGB_ADDITIONAL_GO_ENV

# inputs left alone:
# DEPSDIR
# DEPSGENTOOL
# GOPATH
# GO_ENV
# MAKEFILE_LIST
# REPO_PATH

_BGB_PATH_ := $(lastword $(MAKEFILE_LIST))

$(call setup-stamp-file,_BGB_GO_DEPMK_STAMP_,$(BGB_BINARY)/bgb-go-depmk)
$(call setup-stamp-file,_BGB_KV_DEPMK_STAMP_,$(BGB_BINARY)/bgb-kv-depmk)

# the gopath symlink creation rule should be generated only once, even
# if we include this file multiple times.
ifeq ($(_BGB_RKT_SYMLINK_STAMP_),)

# the symlink stamp wasn't yet generated, do it now.

_BGB_RKT_BASE_SYMLINK_ := src/$(REPO_PATH)
_BGB_RKT_SYMLINK_NAME_ := $(GOPATH_TO_CREATE)/$(_BGB_RKT_BASE_SYMLINK_)

$(call setup-custom-stamp-file,_BGB_RKT_SYMLINK_STAMP_,$(_BGB_PATH_)/rkt-symlink)

$(call generate-stamp-rule,$(_BGB_RKT_SYMLINK_STAMP_),,$(_BGB_RKT_SYMLINK_NAME_))

INSTALL_SYMLINKS += $(MK_TOPLEVEL_ABS_SRCDIR):$(_BGB_RKT_SYMLINK_NAME_)
CREATE_DIRS += $(call dir-chain,$(GOPATH_TO_CREATE),$(call to-dir,$(_BGB_RKT_BASE_SYMLINK_)))

endif

_BGB_PKG_NAME_ := $(REPO_PATH)/$(BGB_PKG_IN_REPO)

$(call setup-dep-file,_BGB_GO_DEPMK,$(_BGB_PKG_NAME_))
$(call setup-dep-file,_BGB_KV_DEPMK,$(_BGB_PKG_NAME_)/kv)

$(call forward-vars,$(BGB_BINARY), \
	BGB_ADDITIONAL_GO_ENV GO_ENV GO BGB_GO_FLAGS _BGB_PKG_NAME_)

ifeq ($(INCREMENTAL_BUILD),yes)

$(BGB_BINARY): $(_BGB_PATH_) $(_BGB_RKT_SYMLINK_STAMP_)
	$(VQ) \
	$(call vb,vt,GO,$(call vsg,$(_BGB_PKG_NAME_))) \
	$(GO_ENV) $(BGB_ADDITIONAL_GO_ENV) GOBIN=$(dir $(@)) "$(GO)" install -pkgdir $(GOPATH)/pkg $(call v3,-v -x) $(BGB_GO_FLAGS) "$(_BGB_PKG_NAME_)"

else

$(BGB_BINARY): $(_BGB_PATH_) $(_BGB_RKT_SYMLINK_STAMP_)
	$(VQ) \
	$(call vb,vt,GO,$(call vsg,$(_BGB_PKG_NAME_))) \
	$(GO_ENV) $(BGB_ADDITIONAL_GO_ENV) "$(GO)" build $(call v3,-v -x) -o "$@" $(BGB_GO_FLAGS) "$(_BGB_PKG_NAME_)"

endif

$(call generate-go-deps,$(_BGB_GO_DEPMK_STAMP_),$(BGB_BINARY),$(_BGB_GO_DEPMK),$(BGB_PKG_IN_REPO))
$(call generate-kv-deps,$(_BGB_KV_DEPMK_STAMP_),$(BGB_BINARY),$(_BGB_KV_DEPMK),BGB_GO_FLAGS)

$(BGB_STAMP): $(BGB_BINARY) $(_BGB_GO_DEPMK_STAMP_) $(_BGB_KV_DEPMK_STAMP_)

# _BGB_RKT_SYMLINK_STAMP_ is deliberately not cleared - it needs to
# stay defined to make sure that the gopath symlink rule is generated
# only once.
$(call undefine-namespaces,BGB _BGB,_BGB_RKT_SYMLINK_STAMP_)
