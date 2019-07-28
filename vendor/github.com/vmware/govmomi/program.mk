ifneq (,$(strip $(GOOS)))
ifeq (,$(strip $(GOARCH)))
GOARCH := $(shell go env | grep GOARCH | awk -F= '{print $$2}' | tr -d '"')
endif
endif

ifneq (,$(strip $(GOARCH)))
ifeq (,$(strip $(GOOS)))
GOOS := $(shell go env | grep GOOS | awk -F= '{print $$2}' | tr -d '"')
endif
endif

ifeq (2,$(words $(GOOS) $(GOARCH)))
PROGRAM := $(PROGRAM)_$(GOOS)_$(GOARCH)
endif

ifeq (windows,$(GOOS))
PROGRAM := $(PROGRAM).exe
endif

all: $(PROGRAM)

TAGS += netgo
ifeq (,$(strip $(findstring -w,$(LDFLAGS))))
LDFLAGS += -w
endif
BUILD_ARGS := -tags '$(TAGS)' -ldflags '$(LDFLAGS)' -v

$(PROGRAM):
	CGO_ENABLED=0 go build -a $(BUILD_ARGS) -o $@

install:
	CGO_ENABLED=0 go install $(BUILD_ARGS)

ifneq (,$(strip $(BUILD_OS)))
ifneq (,$(strip $(BUILD_ARCH)))
GOOS_GOARCH_TARGETS := $(foreach a,$(BUILD_ARCH),$(patsubst %,%_$a,$(BUILD_OS)))
XBUILD := $(addprefix $(PROGRAM)_,$(GOOS_GOARCH_TARGETS))
$(XBUILD):
	GOOS=$(word 2,$(subst _, ,$@)) GOARCH=$(word 3,$(subst _, ,$@)) $(MAKE) --output-sync=target
build-all: $(XBUILD)
endif
endif

clean:
	@rm -f $(PROGRAM) $(XBUILD)

.PHONY: build-all install clean
