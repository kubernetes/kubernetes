$(call setup-stamp-file,LOG_STRESSER_STAMP)

# variables for makelib/build_go_bin.mk
LOG_STRESSER := $(TARGET_BINDIR)/log-stresser
BGB_STAMP := $(LOG_STRESSER_STAMP)
BGB_PKG_IN_REPO := tests/rkt-monitor/log-stresser
BGB_BINARY := $(LOG_STRESSER)
BGB_ADDITIONAL_GO_ENV := GOARCH=$(GOARCH_FOR_BUILD)
BGB_GO_FLAGS := -tags netgo -ldflags '-w'
BGB_ADDITIONAL_GO_ENV := CGO_ENABLED=0 GOOS=linux

CLEAN_FILES += $(LOG_STRESSER)

$(call generate-stamp-rule,$(LOG_STRESSER_STAMP))

$(LOG_STRESSER): $(MK_PATH) | $(BINDIR)

include makelib/build_go_bin.mk

# LOG_STRESSER_STAMP deliberately not cleared

RKT_MONITOR_STAMPS += $(LOG_STRESSER_STAMP)
