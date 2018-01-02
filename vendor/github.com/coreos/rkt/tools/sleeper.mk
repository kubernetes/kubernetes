$(call setup-stamp-file,SLEEPER_STAMP)

# variables for makelib/build_go_bin.mk
SLEEPER := $(TARGET_BINDIR)/sleeper
BGB_STAMP := $(SLEEPER_STAMP)
BGB_PKG_IN_REPO := tests/rkt-monitor/sleeper
BGB_BINARY := $(SLEEPER)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)
BGB_GO_FLAGS := -tags netgo -ldflags '-w'
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0 GOOS=linux

CLEAN_FILES += $(SLEEPER)

$(call generate-stamp-rule,$(SLEEPER_STAMP))

$(SLEEPER): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

# SLEEPER_STAMP deliberately not cleared

RKT_MONITOR_STAMPS += $(SLEEPER_STAMP)
