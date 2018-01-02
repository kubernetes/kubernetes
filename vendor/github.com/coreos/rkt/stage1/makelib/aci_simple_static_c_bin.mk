# inputs cleared after including this file:
#
# ASSCB_EXTRA_HEADERS - additional local C headers used
#
# ASSCB_EXTRA_SOURCES - additional local C sources used
#
# ASSCB_FLAVORS - which flavors want this app, set to STAGE1_FLAVORS
# if empty
#
# ASSCB_EXTRA_CFLAGS - additional cflags

# There are two parts:
#
# 1. Building the binary - it is done only once.
#
# 2. Installing the binary into the ACI rootfs directory - this is
# done once for each flavor.

# Initial stuff

# The path of this file. This file is included (or at least it should
# be) with a standard include directive instead of our inc-one (or
# inc-many), so the MK_PATH, MK_FILENAME and MK_SRCDIR are set to
# values specific to the parent file (that is - including this one).
_ASSCB_PATH_ := $(lastword $(MAKEFILE_LIST))
# Name of a binary, deduced upon filename of a parent Makefile.
_ASSCB_NAME_ := $(patsubst %.mk,%,$(MK_FILENAME))
# Path to built binary. Not the one in the ACI rootfs.
_ASSCB_BINARY_ := $(TARGET_TOOLSDIR)/$(_ASSCB_NAME_)

ifeq ($(ASSCB_FLAVORS),)

ASSCB_FLAVORS := $(STAGE1_FLAVORS)

endif

# 1.

# variables for build_static_c_bin.mk
BSCB_BINARY := $(_ASSCB_BINARY_)
BSCB_HEADERS := $(foreach h,$(ASSCB_EXTRA_HEADERS),$(MK_SRCDIR)/$h)
BSCB_SOURCES := $(MK_SRCDIR)/$(_ASSCB_NAME_).c $(foreach h,$(ASSCB_EXTRA_SOURCES),$(MK_SRCDIR)/$h)
BSCB_ADDITIONAL_CFLAGS := -Wall -Os $(ASSCB_EXTRA_CFLAGS)

CLEAN_FILES += $(_ASSCB_BINARY_)

include makelib/build_static_c_bin.mk

$(_ASSCB_BINARY_): $(_ASSCB_PATH_) $(MK_PATH) | $(TARGET_TOOLSDIR)

# 2.

AIB_FLAVORS := $(ASSCB_FLAVORS)
AIB_BINARY := $(_ASSCB_BINARY_)
include stage1/makelib/aci_install_bin.mk

$(call undefine-namespaces,ASSCB _ASSCB)
