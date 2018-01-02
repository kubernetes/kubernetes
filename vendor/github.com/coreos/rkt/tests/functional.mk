$(call setup-stamp-file,FTST_FUNCTIONAL_TESTS_STAMP,/functional-tests)
$(call setup-tmp-dir,FTST_TMPDIR)

FTST_TEST_TMP := $(FTST_TMPDIR)/test-tmp
FTST_IMAGE_DIR := $(FTST_TMPDIR)/image
FTST_IMAGE_ROOTFSDIR := $(FTST_IMAGE_DIR)/rootfs
FTST_IMAGE := $(FTST_TMPDIR)/rkt-inspect.aci
FTST_IMAGE_MANIFEST_SRC := $(MK_SRCDIR)/image/manifest
FTST_IMAGE_MANIFEST := $(FTST_IMAGE_DIR)/manifest
FTST_IMAGE_TEST_DIRS := \
	$(FTST_IMAGE_ROOTFSDIR)/dir1 \
	$(FTST_IMAGE_ROOTFSDIR)/dir2 \
	$(FTST_IMAGE_ROOTFSDIR)/bin \
	$(FTST_IMAGE_ROOTFSDIR)/etc
FTST_ACE_MAIN_IMAGE_DIR := $(FTST_TMPDIR)/ace-main
FTST_ACE_MAIN_IMAGE := $(FTST_TMPDIR)/rkt-ace-validator-main.aci
FTST_ACE_MAIN_IMAGE_MANIFEST_SRC := vendor/github.com/appc/spec/ace/image_manifest_main.json.in
FTST_ACE_MAIN_IMAGE_MANIFEST := $(FTST_ACE_MAIN_IMAGE_DIR)/manifest
FTST_ACE_SIDEKICK_IMAGE_DIR := $(FTST_TMPDIR)/ace-sidekick
FTST_ACE_SIDEKICK_IMAGE := $(FTST_TMPDIR)/rkt-ace-validator-sidekick.aci
FTST_ACE_SIDEKICK_IMAGE_MANIFEST_SRC := vendor/github.com/appc/spec/ace/image_manifest_sidekick.json.in
FTST_ACE_SIDEKICK_IMAGE_MANIFEST := $(FTST_ACE_SIDEKICK_IMAGE_DIR)/manifest
FTST_INSPECT_BINARY := $(FTST_TMPDIR)/inspect
FTST_ACI_INSPECT := $(FTST_IMAGE_ROOTFSDIR)/inspect
FTST_ACE_BINARY := $(FTST_TMPDIR)/ace
FTST_ECHO_SERVER_BINARY := $(FTST_TMPDIR)/echo-socket-activated
FTST_ACI_ECHO_SERVER := $(FTST_IMAGE_ROOTFSDIR)/echo-socket-activated
FTST_CNI_PROXY_BINARY := $(FTST_TMPDIR)/cniproxy
FTST_EMPTY_IMAGE_DIR := $(FTST_TMPDIR)/empty-image
FTST_EMPTY_IMAGE_ROOTFSDIR := $(FTST_EMPTY_IMAGE_DIR)/rootfs
FTST_EMPTY_IMAGE := $(FTST_TMPDIR)/rkt-empty.aci
FTST_EMPTY_IMAGE_MANIFEST_SRC := $(MK_SRCDIR)/empty-image/manifest
FTST_EMPTY_IMAGE_MANIFEST := $(FTST_EMPTY_IMAGE_DIR)/manifest
FTST_RKT_PATH := $(FTST_TMPDIR)/rkt
FTST_STAGE1_FLAVOR_FILENAMES := $(foreach f,$(call commas-to-spaces,$(RKT_STAGE1_ALL_FLAVORS)),stage1-$f.aci)
FTST_STAGE1_ALL_FLAVOR_SYMLINKS := $(foreach f,$(FTST_STAGE1_FLAVOR_FILENAMES),$(FTST_TMPDIR)/$f)

$(call inc-one,stub-stage1/stub-stage1.mk)

TOPLEVEL_CHECK_STAMPS += $(FTST_FUNCTIONAL_TESTS_STAMP)
TOPLEVEL_FUNCTIONAL_CHECK_STAMPS += $(FTST_FUNCTIONAL_TESTS_STAMP)
INSTALL_FILES += \
	$(RKT_BINARY):$(FTST_RKT_PATH):- \
	$(FTST_IMAGE_MANIFEST_SRC):$(FTST_IMAGE_MANIFEST):- \
	$(FTST_INSPECT_BINARY):$(FTST_ACI_INSPECT):- \
	$(FTST_EMPTY_IMAGE_MANIFEST_SRC):$(FTST_EMPTY_IMAGE_MANIFEST):- \
	$(FTST_ECHO_SERVER_BINARY):$(FTST_ACI_ECHO_SERVER):-
CREATE_DIRS += \
	$(FTST_IMAGE_DIR) \
	$(FTST_EMPTY_IMAGE_DIR) \
	$(FTST_EMPTY_IMAGE_ROOTFSDIR) \
	$(FTST_IMAGE_TEST_DIRS) \
	$(FTST_TEST_TMP)
INSTALL_DIRS += \
	$(FTST_IMAGE_ROOTFSDIR):0755
INSTALL_SYMLINKS += \
	$(foreach f,$(FTST_STAGE1_FLAVOR_FILENAMES),$(TARGET_BINDIR)/$f:$(FTST_TMPDIR)/$f)
CLEAN_FILES += \
	$(FTST_IMAGE) \
	$(FTST_ECHO_SERVER_BINARY) \
	$(FTST_CNI_PROXY_BINARY) \
	$(FTST_INSPECT_BINARY) \
	$(FTST_EMPTY_IMAGE) \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/file \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dir2 \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dotdot_dir2 \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_notexists \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_root \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_rel_dir2 \
	$(FTST_IMAGE_ROOTFSDIR)/dir1/link_invalid \
	$(FTST_IMAGE_ROOTFSDIR)/dir2/file \
	$(FTST_IMAGE_ROOTFSDIR)/etc/group \
	$(FTST_IMAGE_ROOTFSDIR)/etc/passwd \
	$(FTST_ACE_BINARY)
CLEAN_DIRS += \
	$(FTST_IMAGE_ROOTFSDIR)/dir1 \
	$(FTST_IMAGE_ROOTFSDIR)/dir2 \
	$(FTST_IMAGE_ROOTFSDIR)/bin \
	$(FTST_IMAGE_ROOTFSDIR)/etc
CLEAN_SYMLINKS += \
	$(FTST_IMAGE_ROOTFSDIR)/inspect-link \
	$(FTST_IMAGE_ROOTFSDIR)/bin/inspect-link-bin
FTST_FUNCTIONAL_TESTS_TIMEOUT := 60m

$(call forward-vars,$(FTST_FUNCTIONAL_TESTS_STAMP), \
	FTST_RKT_PATH ACTOOL FTST_IMAGE FTST_EMPTY_IMAGE FTST_TEST_TMP ABS_GO \
	FTST_INSPECT_BINARY GO_ENV GO_TEST_FUNC_ARGS REPO_PATH FTST_CNI_PROXY_BINARY \
	FTST_ACE_MAIN_IMAGE FTST_ACE_SIDEKICK_IMAGE RKT_STAGE1_DEFAULT_FLAVOR FTST_SS1_IMAGE \
	FTST_FUNCTIONAL_TESTS_TIMEOUT)
$(FTST_FUNCTIONAL_TESTS_STAMP): $(FTST_IMAGE) $(FTST_EMPTY_IMAGE) $(ACTOOL_STAMP) $(RKT_STAMP) $(FTST_ACE_MAIN_IMAGE) $(FTST_ACE_SIDEKICK_IMAGE) $(FTST_SS1_STAMP) | $(FTST_TEST_TMP) $(FTST_RKT_PATH) $(FTST_STAGE1_ALL_FLAVOR_SYMLINKS)
	$(VQ) \
	$(call vb,vt,GO TEST,$(REPO_PATH)/tests) \
	sudo -E RKT_STAGE1_DEFAULT_FLAVOR="$(RKT_STAGE1_DEFAULT_FLAVOR)" \
		RKT="$(FTST_RKT_PATH)" \
		ACTOOL="$(ACTOOL)" \
		RKT_INSPECT_IMAGE="$(FTST_IMAGE)" \
		RKT_EMPTY_IMAGE="$(FTST_EMPTY_IMAGE)" \
		RKT_ACE_MAIN_IMAGE=$(FTST_ACE_MAIN_IMAGE) \
		RKT_ACE_SIDEKICK_IMAGE=$(FTST_ACE_SIDEKICK_IMAGE) \
		RKT_CNI_PROXY=$(FTST_CNI_PROXY_BINARY) \
		FUNCTIONAL_TMP="$(FTST_TEST_TMP)" \
		INSPECT_BINARY="$(FTST_INSPECT_BINARY)" \
		STUB_STAGE1="$(FTST_SS1_IMAGE)" \
		$(GO_ENV) "$(ABS_GO)" \
		test -tags $(RKT_STAGE1_DEFAULT_FLAVOR) -timeout $(FTST_FUNCTIONAL_TESTS_TIMEOUT) -v $(GO_TEST_FUNC_ARGS) $(REPO_PATH)/tests

$(call forward-vars,$(FTST_IMAGE), \
	FTST_IMAGE_ROOTFSDIR ACTOOL FTST_IMAGE_DIR)
$(FTST_IMAGE): $(FTST_IMAGE_MANIFEST) $(FTST_ACI_INSPECT) $(FTST_ACI_ECHO_SERVER) | $(FTST_IMAGE_TEST_DIRS)
	$(VQ) \
	set -e; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/file)) \
	echo -n dir1 >$(FTST_IMAGE_ROOTFSDIR)/dir1/file; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dir2)) \
	ln -sf /dir2 $(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dir2; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dotdot_dir2)) \
	ln -sf /../dir2 $(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_dotdot_dir2; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/link_rel_dir2)) \
	ln -sf ../dir2 $(FTST_IMAGE_ROOTFSDIR)/dir1/link_rel_dir2; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_root)) \
	ln -sf / $(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_root; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_notexists)) \
	ln -sf /notexists $(FTST_IMAGE_ROOTFSDIR)/dir1/link_abs_notexists; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/dir2/file)) \
	echo -n dir2 >$(FTST_IMAGE_ROOTFSDIR)/dir2/file; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/etc/group)) \
	echo -n group1:x:100:user1 >$(FTST_IMAGE_ROOTFSDIR)/etc/group; \
	$(call vb,v2,GEN,$(call vsp,$(FTST_IMAGE_ROOTFSDIR)/etc/passwd)) \
	echo -n user1:x:1000:100::/: >$(FTST_IMAGE_ROOTFSDIR)/etc/passwd; \
	$(call vb,v2,LN SF,/inspect $(call vsp,$(FTST_IMAGE_ROOTFSDIR)/inspect-link)) \
	ln -sf /inspect $(FTST_IMAGE_ROOTFSDIR)/inspect-link; \
	$(call vb,v2,LN SF,/inspect $(call vsp,$(FTST_IMAGE_ROOTFSDIR)/bin/inspect-link-bin)) \
	ln -sf /inspect $(FTST_IMAGE_ROOTFSDIR)/bin/inspect-link-bin; \
	$(call vb,vt,ACTOOL,$(call vsp,$@)) \
	"$(ACTOOL)" build --overwrite --owner-root "$(FTST_IMAGE_DIR)" "$@"

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FTST_FUNCTIONAL_TESTS_STAMP)
BGB_BINARY := $(FTST_INSPECT_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/inspect
BGB_GO_FLAGS := -a -installsuffix cgo
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0

include makelib/build_go_bin.mk

BGB_STAMP := $(FTST_FUNCTIONAL_TESTS_STAMP)
BGB_BINARY := $(FTST_ACE_BINARY)
BGB_PKG_IN_REPO := vendor/github.com/appc/spec/ace
BGB_GO_FLAGS := -a -installsuffix cgo
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0

include makelib/build_go_bin.mk

$(call forward-vars,$(FTST_EMPTY_IMAGE), \
	ACTOOL FTST_EMPTY_IMAGE_DIR)
$(FTST_EMPTY_IMAGE): $(FTST_EMPTY_IMAGE_MANIFEST) | $(FTST_EMPTY_IMAGE_ROOTFSDIR)
	$(VQ) \
	$(call vb,vt,ACTOOL,$(call vsp,$@)) \
	"$(ACTOOL)" build --overwrite "$(FTST_EMPTY_IMAGE_DIR)" "$@"

# variables for makelib/build_go_bin.mk
BGB_STAMP := $(FTST_FUNCTIONAL_TESTS_STAMP)
BGB_BINARY := $(FTST_ECHO_SERVER_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/echo-socket-activated
BGB_GO_FLAGS := -a -installsuffix cgo
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0

include makelib/build_go_bin.mk


BGB_STAMP := $(FTST_FUNCTIONAL_TESTS_STAMP)
BGB_BINARY := $(FTST_CNI_PROXY_BINARY)
BGB_PKG_IN_REPO := $(call go-pkg-from-dir)/cniproxy
BGB_GO_FLAGS := -a -installsuffix cgo
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0

include makelib/build_go_bin.mk


# 1 - image
# 2 - aci directory
# 3 - ace validator
# 4 - manifest.json.in
define FTST_GENERATE_ACE_IMAGE

$$(call forward-vars,$2/manifest,ABS_GO)
$2/manifest: $4 | $2
	$$(VQ) \
	$$(call vb,v2,GEN,$$(call vsp,$$@)) \
	GOARCH="$$$$($$(ABS_GO) env GOARCH)"; \
	sed -e "s/@GOOS@/linux/" -e "s/@GOARCH@/$$$$GOARCH/" <$$< >$$@

$$(call forward-vars,$1,ACTOOL)
$1: $2/manifest $2/rootfs/ace-validator | $2/rootfs/opt/acvalidator
	$$(VQ) \
	$$(call vb,vt,ACTOOL,$$(call vsp,$$@)) \
	"$$(ACTOOL)" build --overwrite "$2" "$1"

CREATE_DIRS += $2 $$(call dir-chain,$2,rootfs/opt/acvalidator)
INSTALL_FILES += $3:$2/rootfs/ace-validator:-
CLEAN_FILES += $1 $2/manifest
endef

$(eval $(call FTST_GENERATE_ACE_IMAGE,$(FTST_ACE_MAIN_IMAGE),$(FTST_ACE_MAIN_IMAGE_DIR),$(FTST_ACE_BINARY),$(FTST_ACE_MAIN_IMAGE_MANIFEST_SRC)))
$(eval $(call FTST_GENERATE_ACE_IMAGE,$(FTST_ACE_SIDEKICK_IMAGE),$(FTST_ACE_SIDEKICK_IMAGE_DIR),$(FTST_ACE_BINARY),$(FTST_ACE_SIDEKICK_IMAGE_MANIFEST_SRC)))

$(call undefine-namespaces,FTST)
