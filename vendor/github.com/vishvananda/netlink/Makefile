DIRS := \
	. \
	nl

DEPS = \
	github.com/vishvananda/netns \
	golang.org/x/sys/unix

uniq = $(if $1,$(firstword $1) $(call uniq,$(filter-out $(firstword $1),$1)))
testdirs = $(call uniq,$(foreach d,$(1),$(dir $(wildcard $(d)/*_test.go))))
goroot = $(addprefix ../../../,$(1))
unroot = $(subst ../../../,,$(1))
fmt = $(addprefix fmt-,$(1))

all: test

$(call goroot,$(DEPS)):
	go get $(call unroot,$@)

.PHONY: $(call testdirs,$(DIRS))
$(call testdirs,$(DIRS)):
	go test -test.exec sudo -test.parallel 4 -timeout 60s -test.v github.com/vishvananda/netlink/$@

$(call fmt,$(call testdirs,$(DIRS))):
	! gofmt -l $(subst fmt-,,$@)/*.go | grep -q .

.PHONY: fmt
fmt: $(call fmt,$(call testdirs,$(DIRS)))

test: fmt $(call goroot,$(DEPS)) $(call testdirs,$(DIRS))
