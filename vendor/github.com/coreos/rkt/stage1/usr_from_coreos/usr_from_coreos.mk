$(call setup-tmp-dir,UFC_TMPDIR)

# This directory will be used by the build-usr.mk
UFC_CBUDIR := $(UFC_TMPDIR)/cbu

$(call setup-stamp-file,UFC_CBU_STAMP,cbu)

INSTALL_DIRS += $(UFC_CBUDIR):-
S1_RF_USR_STAMPS += $(UFC_CBU_STAMP)

# Input variables for building the ACI rootfs from CoreOS image
# (build-usr.mk).
CBU_MANIFESTS_DIR := $(MK_SRCDIR)/manifest-$(RKT_STAGE1_COREOS_BOARD).d
CBU_TMPDIR := $(UFC_CBUDIR)
CBU_DIFF := for-usr-from-coreos-mk
CBU_STAMP := $(UFC_CBU_STAMP)
CBU_ACIROOTFSDIR := $(S1_RF_ACIROOTFSDIR)
CBU_FLAVOR := coreos

$(call inc-one,build-usr.mk)

$(call undefine-namespaces,UFC)
