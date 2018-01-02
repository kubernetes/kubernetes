# This file gets the list of standard library locations and tells make
# to create their counterparts inside the ACI rootfs directory.
#
# Inputs:
#
# SLD_FLAVOR - a flavor which wants the standard library directories
# to be created

ifeq ($(SLD_INCLUDED),)

SLD_INCLUDED := x
SLD_LOCATIONS := $(shell ld --verbose | grep SEARCH_DIR | sed -e 's/SEARCH_DIR("=*\([^"]*\)");*/\1/g')
SLD_LOCATIONS += $(foreach l,$(SLD_LOCATIONS),$l/systemd)

endif

ifneq ($(SLD_FLAVOR),)

ifeq ($(SLDKEEP_INCLUDED_$(SLD_FLAVOR)),)

SLDKEEP_INCLUDED_$(SLD_FLAVOR) := x
SLD_ACIROOTFSDIR := $(STAGE1_ACIROOTFSDIR_$(SLD_FLAVOR))

INSTALL_DIRS += $(foreach l,$(SLD_LOCATIONS),$(SLD_ACIROOTFSDIR)$l:0755)

endif

endif

# SLD_LOCATIONS is deliberately not cleared, we will use this variable
# to know standard library directories.
# SLD_INCLUDED and SLDKEEP_* variables are not cleared to avoid
# potential problems with including this file more than once.
$(call undefine-namespaces,SLD,SLD_LOCATIONS SLD_INCLUDED)
