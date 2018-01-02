$(call setup-stamp-file,CPU_STRESSER_STAMP)

# variables for makelib/build_go_bin.mk
CPU_STRESSER := $(TARGET_BINDIR)/cpu-stresser
BGB_STAMP := $(CPU_STRESSER_STAMP)
BGB_PKG_IN_REPO := tests/rkt-monitor/cpu-stresser
BGB_BINARY := $(CPU_STRESSER)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)
BGB_GO_FLAGS := -tags netgo -ldflags '-w'
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0 GOOS=linux

CLEAN_FILES += $(CPU_STRESSER)

$(call generate-stamp-rule,$(CPU_STRESSER_STAMP))

$(CPU_STRESSER): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

# CPU_STRESSER_STAMP deliberately not cleared

RKT_MONITOR_STAMPS += $(CPU_STRESSER_STAMP)
