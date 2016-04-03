$(call setup-stamp-file,UFSB_STAMP)
UFSB_BASH_ON_ACI := $(S1_RF_ACIROOTFSDIR)/usr/bin/$(notdir $(BASH_SHELL))

S1_RF_SECONDARY_STAMPS += $(UFSB_STAMP)
S1_RF_INSTALL_FILES += $(BASH_SHELL):$(UFSB_BASH_ON_ACI):-
S1_RF_INSTALL_DIRS += $(S1_RF_ACIROOTFSDIR)/usr/bin:-
S1_RF_INSTALL_SYMLINKS += usr/bin:$(S1_RF_ACIROOTFSDIR)/bin

$(call generate-stamp-rule,$(UFSB_STAMP),$(UFSB_BASH_ON_ACI),$(S1_RF_ACIROOTFSDIR)/bin)

$(call undefine-namespaces,UFSB)
