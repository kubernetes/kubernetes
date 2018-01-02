$(call setup-stamp-file,UFH_STAMP)

S1_RF_USR_STAMPS += $(UFH_STAMP)

$(call generate-stamp-rule,$(UFH_STAMP),,$(S1_RF_ACIROOTFSDIR), \
	$(call vb,v2,LN SF,usr/bin $(S1_RF_ACIROOTFSDIR)/bin) \
	ln -sf usr/bin "$(S1_RF_ACIROOTFSDIR)/bin" && \
	$(call vb,v2,LN SF,host $(S1_RF_ACIROOTFSDIR)/flavor) \
	ln -sf 'host' "$(S1_RF_ACIROOTFSDIR)/flavor" && \
	mkdir -p "$(S1_RF_ACIROOTFSDIR)/usr/lib" && \
	ln -sf usr/lib "$(S1_RF_ACIROOTFSDIR)/lib" && \
	ln -sf usr/lib "$(S1_RF_ACIROOTFSDIR)/lib64"&& \
	ln -sf lib "$(S1_RF_ACIROOTFSDIR)/usr/lib64")

CLEAN_SYMLINKS += \
	$(S1_RF_ACIROOTFSDIR)/bin \
	$(S1_RF_ACIROOTFSDIR)/flavor \
	$(S1_RF_ACIROOTFSDIR)/lib \
	$(S1_RF_ACIROOTFSDIR)/lib64 \
	$(S1_RF_ACIROOTFSDIR)/usr/lib64

CLEAN_DIRS += $(S1_RF_ACIROOTFSDIR)/usr/lib

$(call undefine-namespaces,UFH)
