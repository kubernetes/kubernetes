# This file provides some functions for controlling the verbosity of make
# rules.

#
# Translate english word arguments to a numeric level.
#

ifeq ($V,silent)
override V := 0
endif

ifeq ($V,quiet)
override V := 0
endif

ifeq ($V,info)
override V := 1
endif

ifeq ($V,all)
override V := 2
endif

ifeq ($V,raw)
override V := 3
endif

# if V is empty or bogus, default to info verbosity
ifeq ($(filter $V,0 1 2 3),)
override V := 1
endif

#
# basic functions used by all predicates
#

# returns a non-empty value if $V is one of the $1 verbosity levels
define veq
$(filter $V,$1)
endef

#
# predicates
#

# expands to $1 if $V is 0
define v0
$(if $(call veq,0),$1)
endef

# expands to $1 if $V is 1
define v1
$(if $(call veq,1),$1)
endef

# expands to $1 if $V is 2
define v2
$(if $(call veq,2),$1)
endef

# expands to $1 if $V is 3
define v3
$(if $(call veq,3),$1)
endef

# expands to $1 if $V is greater than 0
define vg0
$(if $(call veq,1 2 3),$1)
endef

# expands to $1 if $V is greater than 1
define vg1
$(if $(call veq,2 3),$1)
endef

# expands to $1 if $V is greater than 2
define vg2
$(if $(call veq,3),$1)
endef

# expands to $1 if $V is lower than 1
define vl1
$(if $(call veq,0),$1)
endef

# expands to $1 if $V is lower than 2
define vl2
$(if $(call veq,0 1),$1)
endef

# expands to $1 if $V is lower than 3
define vl3
$(if $(call veq,0 1 2),$1)
endef

# expands to $1 if $V is either 1 or 2 (truncated verbosity)
define vt
$(if $(call veq,1 2),$1)
endef

#
# functions and variables used inside make rules
#

# expands to a bash snippet printing shortcut and parameters if
# predicate returns a non-empty value
# 1 - predicate (vlX, vgX, vX, or vt for X in <0,3>)
# 2 - shortcut (max 12 characters, like CC, DEPSGLOB, etc)
# 3 - other parameters, printed after shortcut
#
# Example: $(call vb,vl2,STUFF,foo/bar) will print the following text
# if the verbosity level is lower than 2:
#
#   STUFF      foo/bar
define vb
$(call $1,printf '  %-12s %s\n' '$2' "$3";)
endef

# This variable should be used as the first thing in the recipe to
# stop echoing the recipe for lower verbosity levels. It also sets up
# the shell to fail on the first failed command - this is usually
# needed anyway, as lower verbosity levels often add additional
# commands to the recipe to print nice messages.
VQ = $(call vl3,@set -e;)

# This shortens the paths by removing the absolute src directory from
# them. So it truncates /home/foo/projects/rkt/<builddir>/tmp/... to
# <builddir>/tmp/...
define vsp
$(subst $(MK_TOPLEVEL_ABS_SRCDIR)/,,$1)
endef

# This shortens the paths by removing the vendor part. So it
# truncates
# github.com/coreos/rkt/vendor/github.com/appc/spec/schema
# to <VENDOR>/github.com/appc/spec/schema.
define vsg
$(subst $(REPO_PATH)/vendor,<VENDOR>,$1)
endef
