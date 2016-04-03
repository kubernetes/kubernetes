# path to rkt source code distro
ORG_PATH := github.com/coreos
REPO_PATH := $(ORG_PATH)/rkt

override BUILDDIR := $(abspath $(BUILDDIR))

MAKETOOLSDIR := $(MK_TOPLEVEL_SRCDIR)/tools
STAMPSDIR := $(BUILDDIR)/stamps
TOOLSDIR := $(BUILDDIR)/tools
BINDIR := $(BUILDDIR)/bin
GOPATH_TO_CREATE := $(BUILDDIR)/gopath
GOPATH := $(GOPATH_TO_CREATE)/src/github.com/coreos/rkt/Godeps/_workspace:$(GOPATH_TO_CREATE)
DEPSDIR := $(BUILDDIR)/deps
FILELISTDIR := $(BUILDDIR)/filelists
MAINTEMPDIR := $(BUILDDIR)/tmp
CLEANDIR := $(BUILDDIR)/clean

ACTOOL := $(BINDIR)/actool
DEPSGENTOOL := $(TOOLSDIR)/depsgen
FILELISTGENTOOL := $(TOOLSDIR)/filelistgen
CLEANGENTOOL := $(TOOLSDIR)/cleangen
QUICKRMTOOL := $(TOOLSDIR)/quickrm

# test-related variables
GO_TEST_PACKAGES ?= ./...
GO_TEST_FUNC_ARGS ?=

GO_ENV := $(strip GOPATH="$(GOPATH)" $(if $(strip $(GOROOT)),GOROOT=$(strip $(GOROOT))))

CREATE_DIRS += \
	$(BUILDDIR) \
	$(STAMPSDIR) \
	$(TOOLSDIR) \
	$(BINDIR) \
	$(GOPATH_TO_CREATE) \
	$(DEPSDIR) \
	$(FILELISTDIR) \
	$(MAINTEMPDIR) \
	$(CLEANDIR)
