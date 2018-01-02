$(call setup-stamp-file,MEM_STRESSER_STAMP)

# variables for makelib/build_go_bin.mk
MEM_STRESSER := $(TARGET_BINDIR)/mem-stresser
BGB_STAMP := $(MEM_STRESSER_STAMP)
BGB_PKG_IN_REPO := tests/rkt-monitor/mem-stresser
BGB_BINARY := $(MEM_STRESSER)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)
BGB_GO_FLAGS := -tags netgo -ldflags '-w'
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0 GOOS=linux

CLEAN_FILES += $(MEM_STRESSER)

$(call generate-stamp-rule,$(MEM_STRESSER_STAMP))

$(MEM_STRESSER): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

# MEM_STRESSER_STAMP deliberately not cleared

RKT_MONITOR_STAMPS += $(MEM_STRESSER_STAMP)
