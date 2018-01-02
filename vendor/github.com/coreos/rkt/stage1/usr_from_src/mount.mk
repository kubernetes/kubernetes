$(call setup-stamp-file,UFSM_STAMP)
UFSM_MOUNT_ON_ACI := $(S1_RF_ACIROOTFSDIR)/usr/bin/mount
UFSM_UMOUNT_ON_ACI := $(S1_RF_ACIROOTFSDIR)/usr/bin/umount

S1_RF_SECONDARY_STAMPS += $(UFSM_STAMP)
S1_RF_INSTALL_FILES += /bin/mount:$(UFSM_MOUNT_ON_ACI):-
S1_RF_INSTALL_FILES += /bin/umount:$(UFSM_UMOUNT_ON_ACI):-
S1_RF_INSTALL_DIRS += $(S1_RF_ACIROOTFSDIR)/usr/bin:-
S1_RF_INSTALL_SYMLINKS += usr/bin:$(S1_RF_ACIROOTFSDIR)/bin

$(call generate-stamp-rule,$(UFSM_STAMP),$(UFSM_MOUNT_ON_ACI),$(S1_RF_ACIROOTFSDIR)/bin)
# TODO(krzesimir): add a stamp for umount

$(call undefine-namespaces,UFSM)
