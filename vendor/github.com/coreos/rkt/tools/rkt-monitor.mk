$(call setup-stamp-file,RKT_MONITOR_STAMP)

# variables for makelib/build_go_bin.mk
RKT_MONITOR := $(TARGET_BINDIR)/rkt-monitor
BGB_STAMP := $(RKT_MONITOR_STAMP)
BGB_PKG_IN_REPO := tests/rkt-monitor
BGB_BINARY := $(RKT_MONITOR)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)

CLEAN_FILES += $(RKT_MONITOR)

$(call generate-stamp-rule,$(RKT_MONITOR_STAMP))

$(RKT_MONITOR): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

# RKT_MONITOR_STAMP deliberately not cleared

RKT_MONITOR_STAMPS += $(RKT_MONITOR_STAMP)
