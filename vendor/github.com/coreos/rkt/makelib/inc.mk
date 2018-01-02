# PUBLIC VARIABLES

# Directory containing an mk file.
MK_SRCDIR := # will be set later in file
# Basename of an mk file.
MK_FILENAME := # will be set later in file
# Path of an mk file.
MK_PATH := # will be set later in file
# Directory containing a toplevel Makefile.
MK_TOPLEVEL_SRCDIR :=

# PUBLIC FUNCTIONS

# Includes one file. Sets MK_SRCDIR, MK_FILENAME and MK_PATH variables
# for the scope of included files and restores them then to old
# values. The passed include file must be relative to current
# MK_SRCDIR.
# Example: $(call inc-one,foo/foo.mk)
# Inside foo/foo.mk, MK_SRCDIR will be "inc" (without quotes),
# MK_FILENAME will be "foo.mk" (also without quotes) and MK_PATH -
# "foo/foo.mk" (without quotes, of course).
define inc-one
$(eval $(call _UTIL_MK_INC_ONE_,$1))
endef
# Includes many files. See inc-one docs.
# Example $(call inc-many,foo/foo.mk bar/bar.mk baz/baz.mk)
define inc-many
$(eval $(foreach f,$1,$(call inc-one,$f)))
endef
# Gets directory part of filename, strips trailing slash.
# Example: DIR := $(call to-dir,foo/bar/baz.mk)
define to-dir
$(call _UTIL_MK_TO_DIR_,$1)
endef


# PRIVATE DETAILS

$(if $(_UTIL_MK_INCLUDED_),$(error $(_UTIL_MK_) can be included only once))
# Guard variable used for checking against including this file
# multiple times.
_UTIL_MK_INCLUDED_ := 1

# Gets a directory containing this file. Path is relative to current
# working directory (which is often a directory where toplevel
# Makefile sits). Should be called before including anything in file.
# Example: MK_SRCDIR := $(call _UTIL_MK_GET_THIS_DIR_)
define _UTIL_MK_GET_THIS_DIR_
$(call _UTIL_MK_TO_DIR_,$(call _UTIL_MK_MAKEFILE_))
endef
# Gets a filename of this file. Should be called before including
# anything in file.
# Example: MK_FILENAME := $(call _UTIL_MK_GET_THIS_FILENAME_)
define _UTIL_MK_GET_THIS_FILENAME_
$(notdir $(call _UTIL_MK_MAKEFILE_))
endef
# Gets a path of this file. Should be called before including anything
# in file.
# Example: MK_PATH := $(call _UTIL_MK_GET_THIS_PATH_)
define _UTIL_MK_GET_THIS_PATH_
$(call _UTIL_MK_MAKEFILE_)
endef

# Gets the name of this file. It's important that nothing in this file
# gets included before this line.
_UTIL_MK_ := $(lastword $(MAKEFILE_LIST))
# Gets a name of the file. Filters out name of this file from the list.
# Example: FILE := $(call _UTIL_MK_MAKEFILE)
define _UTIL_MK_MAKEFILE_
$(lastword $(filter-out $(_UTIL_MK_),$(MAKEFILE_LIST)))
endef
# See docs of to-dir.
define _UTIL_MK_TO_DIR_
$(patsubst %/,%,$(dir $1))
endef

# Stacks for saving MK_SRCDIR variables.
_UTIL_MK_SRCDIR_STACK_ :=
_UTIL_MK_SRCDIR_DO_POP_ :=

# Stacks for saving MK_FILENAME variables.
_UTIL_MK_FILENAME_STACK_ :=
_UTIL_MK_FILENAME_DO_POP_ :=

# Stacks for saving MK_PATH variables.
_UTIL_MK_PATH_STACK_ :=
_UTIL_MK_PATH_DO_POP_ :=

# Bails out if variable of passed name is undefined.
# Example: $(call _UTIL_MK_ENSURE_DEFINED_,FOO_VAR)
define _UTIL_MK_ENSURE_DEFINED_
$(eval $(if $(filter undefined,$(origin $1)),$(error $1 variable is not defined)))
endef

# Pushes value onto given stack. Stack must be defined.
# Example: $(call _UTIL_MK_PUSH_,MY_STACK,$(SOME_VALUE))
define _UTIL_MK_PUSH_
$(eval $(call _UTIL_MK_ENSURE_DEFINED_,$1)) \
$(eval $1 := $(strip $2 $($1)))
endef

# Pops value from given stack. Stack must be defined.
# Example: $(call _UTIL_MK_POP_,MY_STACK)
define _UTIL_MK_POP_
$(eval $(call _UTIL_MK_ENSURE_DEFINED_,$1)) \
$(eval $1 := $(wordlist 2,$(words $($1)),$($1)))
endef

# Gets value from top of stack. Stack must be defined.
# Example: VALUE := $(call _UTIL_MK_TOP,MY_STACK)
define _UTIL_MK_TOP_
$(eval $(call _UTIL_MK_ENSURE_DEFINED_,$1)) \
$(firstword $($1))
endef

# Potentially saves a value if it is not empty. Requires two stacks to
# do that - one for maybe storing the value and one for saying whether
# the value was saved. Both stacks must be defined.
# Example: $(call _UTIL_MK_SAVE_VALUE,IS_VALUE_SAVED_STACK,VALUE_STACK,VALUE)
define _UTIL_MK_SAVE_VALUE_
$(eval _UTIL_MK_TMP_DO_POP_STACK_NAME_ := $1) \
$(eval _UTIL_MK_TMP_VALUE_STACK_NAME_ := $2) \
$(eval _UTIL_MK_TMP_VALUE_NAME_ := $3) \
 \
$(eval $(if $(strip $($(_UTIL_MK_TMP_VALUE_NAME_))), \
        $(eval $(call _UTIL_MK_PUSH_,$(_UTIL_MK_TMP_DO_POP_STACK_NAME_),1)) \
        $(eval $(call _UTIL_MK_PUSH_,$(_UTIL_MK_TMP_VALUE_STACK_NAME_),$($(_UTIL_MK_TMP_VALUE_NAME_)))), \
 \
        $(eval $(call _UTIL_MK_PUSH_,$(_UTIL_MK_TMP_DO_POP_STACK_NAME_),0)))) \
 \
$(eval _UTIL_MK_TMP_DO_POP_STACK_NAME_ :=) \
$(eval _UTIL_MK_TMP_VALUE_STACK_NAME_ :=) \
$(eval _UTIL_MK_TMP_VALUE_NAME_ :=)
endef

# Potentially restores a value if it was not empty when saving it. If
# it was, then the variable is set to empty string. See
# _UTIL_MK_SAVE_VALUE_ docs for details.
# Example: $(call _UTIL_MK_LOAD_VALUE,IS_VALUE_SAVED_STACK,VALUE_STACK,VALUE)
define _UTIL_MK_LOAD_VALUE_
$(eval _UTIL_MK_TMP_DO_POP_STACK_NAME_ := $1) \
$(eval _UTIL_MK_TMP_VALUE_STACK_NAME_ := $2) \
$(eval _UTIL_MK_TMP_VALUE_NAME_ := $3) \
$(eval $(if $(filter 0,$(call _UTIL_MK_TOP_,$(_UTIL_MK_TMP_DO_POP_STACK_NAME_))), \
        $(eval $(_UTIL_MK_TMP_VALUE_NAME_) :=), \
 \
        $(eval $(_UTIL_MK_TMP_VALUE_NAME_) := $(call _UTIL_MK_TOP_,$(_UTIL_MK_TMP_VALUE_STACK_NAME_))) \
        $(eval $(call _UTIL_MK_POP_,$(_UTIL_MK_TMP_VALUE_STACK_NAME_))))) \
$(eval $(call _UTIL_MK_POP_,$(_UTIL_MK_TMP_DO_POP_STACK_NAME_))) \
$(eval _UTIL_MK_TMP_DO_POP_STACK_NAME_ :=) \
$(eval _UTIL_MK_TMP_VALUE_STACK_NAME_ :=) \
$(eval _UTIL_MK_TMP_VALUE_NAME_ :=)
endef

# Expands include file path with MK_SRCDIR (or . if it is
# empty). Produces spurious leading/trailing whitespace, so
# _UTIL_MK_EXPAND_INC_FILE_ should be used instead.
# Example: INC_FILE := $(strip $(call _UTIL_MK_EXPAND_INC_FILE_UNSTRIPPED_,bar.mk))
define _UTIL_MK_EXPAND_INC_FILE_UNSTRIPPED_
$(eval $(if $(MK_SRCDIR), \
        $(eval _UTIL_MK_TMP_SRCDIR_ := $(MK_SRCDIR)), \
 \
        $(eval _UTIL_MK_TMP_SRCDIR_ := .))) \
$(eval _UTIL_MK_EXPANDED_INC_FILE_ := $(_UTIL_MK_TMP_SRCDIR_)/$1) \
$(strip $(_UTIL_MK_EXPANDED_INC_FILE_)) \
$(eval _UTIL_MK_TMP_SRCDIR_ :=) \
$(eval _UTIL_MK_EXPANDED_INC_FILE_ :=)
endef

# Same as _UTIL_MK_EXPAND_INC_FILE_UNSTRIPPED_, but without
# unnecessary whitespace.
# Example: INC_FILE := $(call _UTIL_MK_EXPAND_INC_FILE_,bar.mk)
define _UTIL_MK_EXPAND_INC_FILE_
$(strip $(call _UTIL_MK_EXPAND_INC_FILE_UNSTRIPPED_,$1))
endef

define _UTIL_MK_CANONICALIZE_
$(strip \
	$(eval _UTIL_MK_TMP_PATH_ := $(abspath $1)) \
	$(eval _UTIL_MK_TMP_PATH_ := $(patsubst $(MK_TOPLEVEL_ABS_SRCDIR)/%,%,$(_UTIL_MK_TMP_PATH_))) \
	$(eval _UTIL_MK_TMP_PATH_ := $(MK_TOPLEVEL_SRCDIR)/$(_UTIL_MK_TMP_PATH_)) \
	$(_UTIL_MK_TMP_PATH_) \
	$(eval _UTIL_MK_TMP_PATH_ :=))
endef

# See docs of inc-one.
define _UTIL_MK_INC_ONE_
$(eval _UTIL_MK_INC_FILE_ := $(strip $1)) \
$(eval $(if $(filter-out 1,$(words $(_UTIL_MK_INC_FILE_))),$(error Expected one file to include, got '$(_UTIL_MK_INC_FILE_)'))) \
 \
$(eval _UTIL_MK_INC_FILE_ := $(call _UTIL_MK_EXPAND_INC_FILE_,$(_UTIL_MK_INC_FILE_))) \
$(eval _UTIL_MK_INC_FILE_ := $(call _UTIL_MK_CANONICALIZE_,$(_UTIL_MK_INC_FILE_)))
$(eval $(call _UTIL_MK_SAVE_VALUE_,_UTIL_MK_SRCDIR_DO_POP_,_UTIL_MK_SRCDIR_STACK_,MK_SRCDIR)) \
$(eval $(call _UTIL_MK_SAVE_VALUE_,_UTIL_MK_FILENAME_DO_POP_,_UTIL_MK_FILENAME_STACK_,MK_FILENAME)) \
$(eval $(call _UTIL_MK_SAVE_VALUE_,_UTIL_MK_PATH_DO_POP_,_UTIL_MK_PATH_STACK_,MK_PATH)) \
 \
$(eval MK_SRCDIR := $(call _UTIL_MK_TO_DIR_,$(_UTIL_MK_INC_FILE_))) \
$(eval MK_FILENAME := $(notdir $(_UTIL_MK_INC_FILE_))) \
$(eval MK_PATH := $(_UTIL_MK_INC_FILE_)) \
$(eval include $(_UTIL_MK_INC_FILE_)) \
 \
$(eval $(call _UTIL_MK_LOAD_VALUE_,_UTIL_MK_SRCDIR_DO_POP_,_UTIL_MK_SRCDIR_STACK_,MK_SRCDIR)) \
$(eval $(call _UTIL_MK_LOAD_VALUE_,_UTIL_MK_FILENAME_DO_POP_,_UTIL_MK_FILENAME_STACK_,MK_FILENAME)) \
$(eval $(call _UTIL_MK_LOAD_VALUE_,_UTIL_MK_PATH_DO_POP_,_UTIL_MK_PATH_STACK_,MK_PATH))
endef

# Initial setup of exported variables.
MK_SRCDIR := $(call _UTIL_MK_GET_THIS_DIR_)
MK_FILENAME := $(call _UTIL_MK_GET_THIS_FILENAME_)
MK_PATH := $(call _UTIL_MK_GET_THIS_PATH_)
MK_TOPLEVEL_SRCDIR := $(MK_SRCDIR)
MK_TOPLEVEL_ABS_SRCDIR := $(abspath $(MK_TOPLEVEL_SRCDIR))
