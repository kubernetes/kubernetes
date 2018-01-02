$(call setup-stamp-file,UFKF_STAMP)
UFKF_DIR := $(MK_SRCDIR)/files
UFKF_BASE := $(S1_RF_ACIROOTFSDIR)
UFKF_REST := var/run
UFKF_DIR_CHAIN := $(call dir-chain,$(UFKF_BASE),$(UFKF_REST))
UFKF_VAR_RUN := $(UFKF_BASE)/$(UFKF_REST)

UFKF_ACI_FILES := \
	$(S1_RF_ACIROOTFSDIR)/etc/passwd \
	$(S1_RF_ACIROOTFSDIR)/etc/shadow \
	$(S1_RF_ACIROOTFSDIR)/usr/lib64/systemd/system/sshd.socket \
	$(S1_RF_ACIROOTFSDIR)/etc/group \
	$(S1_RF_ACIROOTFSDIR)/etc/ssh/sshd_config \
	$(S1_RF_ACIROOTFSDIR)/usr/lib64/systemd/system/sshd-prep.service \
	$(S1_RF_ACIROOTFSDIR)/usr/lib64/systemd/system/sshd@.service

UFKF_SRC_FILES := $(addprefix $(UFKF_DIR)/,$(notdir $(UFKF_ACI_FILES)))

S1_RF_SECONDARY_STAMPS += $(UFKF_STAMP)
S1_RF_INSTALL_FILES += $(call install-file-triplets,$(UFKF_SRC_FILES),$(UFKF_ACI_FILES),0644)
S1_RF_INSTALL_DIRS += \
	$(addsuffix :0755,$(UFKF_DIR_CHAIN) $(sort $(call to-dir,$(UFKF_ACI_FILES))))

$(call generate-stamp-rule,$(UFKF_STAMP),$(UFKF_ACI_FILES),$(UFKF_VAR_RUN))

$(call undefine-namespaces,UFKF)
