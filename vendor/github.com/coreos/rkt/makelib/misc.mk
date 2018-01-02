# check if we have undefine feature (only in make >=3.82)
ifneq ($(filter undefine,$(.FEATURES)),)

# we have undefine
define undef
$(eval undefine $1)
endef

else

# no undefine available, simply set the variable to empty value
define undef
$(eval $1 :=)
endef

endif

# 1 - a list of variables to undefine
#
# Simply uses undefine directive on all passed variables.
#
# It does not check if variables are in any way special (like being
# special make variables or else).
#
# Example: $(call undefine-variables-unchecked,VAR1 VAR2 VAR3)
define undefine-variables-unchecked
$(strip \
	$(foreach v,$1, \
		$(call undef,$v)))
endef

# 1 - a list of variable namespaces
# 2 - a list of excluded variables
#
# Undefines all variables in all given namespaces (which basically
# means variables with names prefixed with <namespace>_) except for
# ones listed in a given exclusions list.
#
# It does not check if variables are in any way special (like being
# special make variables or else).
#
# It is a bit of make-golf to avoid using variables. See
# undefine-namespaces below, which has clearer code, is doing exactly
# the same, but calls undefine-variables instead (which changes its
# behaviour wrt. the origin of the variables).
#
# Example: $(call undefine-namespaces-unchecked,NS1 NS2 NS3,N1_KEEP_THIS N3_THIS_TOO)
define undefine-namespaces-unchecked
$(strip \
	$(foreach ,x x, \
		$(call undefine-variables-unchecked, \
			$(filter-out $2, \
				$(filter $(foreach n,$1,$n_%),$(.VARIABLES))))))
endef

# 1 - a list of variables to undefine
#
# Undefines those variables from a given list, which have origin
# "file". If the origin of a variable is different, it is left
# untouched.
#
# This function will bail out if any of the variables starts with a
# dot or MAKE.
#
# Example: $(call undefine-variables,VAR1 VAR2 VAR3)
define undefine-variables
$(strip \
	$(foreach p,.% MAKE%, \
		$(eval _MISC_UV_FORBIDDEN_ := $(strip $(filter $p,$1))) \
		$(if $(_MISC_UV_FORBIDDEN_), \
			$(eval _MISC_UV_ERROR_ += Trying to undefine $(_MISC_UV_FORBIDDEN_) variables which match the forbidden pattern $p.))) \
	$(if $(_MISC_UV_ERROR_), \
		$(error $(_MISC_UV_ERROR_))) \
	$(foreach v,$1, \
		$(if $(filter-out file,$(origin $v)), \
			$(eval _MISC_UV_EXCLUDES_ += $v))) \
	$(eval _MISC_UV_VARS_ := $(filter-out $(_MISC_UV_EXCLUDES_), $1)) \
	$(call undefine-variables-unchecked,$(_MISC_UV_VARS_)) \
	$(call undefine-namespaces-unchecked,_MISC_UV))
endef

# 1 - a list of variable namespaces
# 2 - a list of excluded variables
#
# Undefines those variables in all given namespaces (which basically
# means variables with names prefixed with <namespace>_), which have
# origin "file". If the origin of the variable is different or the
# variable is a part of exclusions list, it is left untouched.
#
# This function will bail out if any of the variables starts with a
# dot or MAKE.
#
# The function performs the action twice - sometimes defined variables
# are not listed in .VARIABLES list initially, but they do show up
# there after first iteration, so we can remove them then. It is
# likely a make bug.
#
# Example: $(call undefine-namespaces,NS1 NS2 NS3,N1_KEEP_THIS N3_THIS_TOO)
define undefine-namespaces
$(strip \
	$(foreach ,x x, \
		$(eval _MISC_UN_VARS_ := $(filter $(foreach n,$1,$n_%),$(.VARIABLES))) \
		$(eval _MISC_UN_VARS_ := $(filter-out $2,$(_MISC_UN_VARS_))) \
		$(call undefine-variables,$(_MISC_UN_VARS_)) \
		$(call undefine-namespaces-unchecked,_MISC_UN)))
endef

define multi-subst
$(strip \
	$(eval _MISC_MS_TMP_ := $(strip $3)) \
	$(eval $(foreach s,$1, \
		$(eval _MISC_MS_TMP_ := $(subst $s,$2,$(_MISC_MS_TMP_))))) \
	$(_MISC_MS_TMP_) \
	$(call undefine-namespaces,_MISC_MS))
endef

# When updating replaced chars here, remember to update them in
# libdepsgen.pm in escape_path sub.
define escape-for-file
$(call multi-subst,- / . : +,_,$1)
endef

define path-to-file-with-suffix
$(call escape-for-file,$1).$2
endef

define stamp-file
$(STAMPSDIR)/$(call path-to-file-with-suffix,$1,stamp)
endef

# Generates a stamp filename and assigns it to passed variable
# name. Generates a stamp's dependency on stamps directory. Adds stamp
# to CLEAN_FILES. Optional second parameter is for adding a suffix to
# stamp.
# Example: $(call setup-custom-stamp-file,FOO_STAMP,/some_suffix)
define setup-custom-stamp-file
$(strip \
	$(eval $1 := $(call stamp-file,$2)) \
	$(eval $($1): | $$(call to-dir,$($1))) \
	$(eval CLEAN_FILES += $($1)))
endef

# Generates a stamp filename and assigns it to passed variable
# name. Generates a stamp's dependency on stamps directory. Adds stamp
# to CLEAN_FILES. Optional second parameter is for adding a suffix to
# stamp.
# Example: $(call setup-stamp-file,FOO_STAMP,/some_suffix)
define setup-stamp-file
$(eval $(call setup-custom-stamp-file,$1,$(MK_PATH)$2))
endef

define dep-file
$(DEPSDIR)/$(call path-to-file-with-suffix,$1,dep.mk)
endef

define setup-custom-dep-file
$(strip \
	$(eval $1 := $(call dep-file,$2)) \
	$(eval $($1): | $$(call to-dir,$($1))) \
	$(eval CLEAN_FILES += $($1)))
endef

define setup-dep-file
$(eval $(call setup-custom-dep-file,$1,$(MK_PATH)$2))
endef

# Returns all not-excluded directories inside $REPO_PATH that have
# nonzero files matching given "go list -f {{.ITEM1}} {{.ITEM2}}...".
# 1 - where to look for files (./... to look for all files inside the project)
# 2 - a list of "go list -f {{.ITEM}}" items (GoFiles, TestGoFiles, etc)
# 3 - space-separated list of excluded directories
# Example: $(call go-find-directories,./...,TestGoFiles XTestGoFiles,tests)
define go-find-directories
$(strip \
	$(eval _MISC_GFD_ESCAPED_SRCDIR := $(MK_TOPLEVEL_ABS_SRCDIR)) \
	$(eval _MISC_GFD_ESCAPED_SRCDIR := $(subst .,\.,$(_MISC_GFD_ESCAPED_SRCDIR))) \
	$(eval _MISC_GFD_ESCAPED_SRCDIR := $(subst /,\/,$(_MISC_GFD_ESCAPED_SRCDIR))) \
	$(eval _MISC_GFD_SPACE_ :=) \
	$(eval _MISC_GFD_SPACE_ +=) \
	$(eval _MISC_GFD_GO_LIST_ITEMS_ := $(foreach i,$2,{{.$i}})) \
	$(eval _MISC_GFD_FILES_ := $(shell $(GO_ENV) "$(GO)" list -f '{{.ImportPath}} $(_MISC_GFD_GO_LIST_ITEMS_)' $1 | \
		grep '\[[^]]' | \
		grep -v '/vendor' | \
		sed -e 's/.*$(_MISC_GFD_ESCAPED_SRCDIR)\///' -e 's/[[:space:]]*\[.*\]$$//' \
		$(if $3,| grep --invert-match '^\($(subst $(_MISC_GFD_SPACE_),\|,$3)\)'))) \
	$(_MISC_GFD_FILES_) \
	$(call undefine-namespaces,_MISC_GFD))
endef

# Escapes all single quotes in $1 (by replacing all ' with '"'"')
define sq_escape
$(subst ','"'"',$1)
endef
#'

# Returns 1 if both parameters are equal, otherwise returns empty
# string.
# Example: is_a_equal_to_b := $(if $(call equal,a,b),yes,no)
define equal
$(strip \
        $(eval _MISC_EQ_OP1_ := $(call sq_escape,$1)) \
        $(eval _MISC_EQ_OP2_ := $(call sq_escape,$2)) \
        $(eval _MISC_EQ_TMP_ := $(shell expr '$(_MISC_EQ_OP1_)' = '$(_MISC_EQ_OP2_)')) \
        $(filter $(_MISC_EQ_TMP_),1) \
        $(call undefine-namespaces,_MISC_EQ))
endef

# Returns a string with all backslashes and double quotes escaped and
# wrapped in another double quotes. Useful for passing a string as a
# single parameter. In general the following should print the same:
# str := "aaa"
# $(info $(str))
# $(shell echo $(call escape-and-wrap,$(str)))
define escape-and-wrap
"$(subst ",\",$(subst \,\\,$1))"
endef
# "
# the double quotes in comment above remove highlighting confusion

# Forwards given variables to a given rule.
# 1 - a rule target
# 2 - a list of variables to forward
#
# Example: $(call forward-vars,$(MY_TARGET),VAR1 VAR2 VAR3)
#
# The effect is basically:
# $(MY_TARGET): VAR1 := $(VAR1)
# $(MY_TARGET): VAR2 := $(VAR2)
# $(MY_TARGET): VAR3 := $(VAR3)
define forward-vars
$(strip \
	$(foreach v,$2, \
		$(eval $1: $v := $($v))))
endef

# Returns a colon (:) if passed value is empty. Useful for avoiding
# shell errors about an empty body.
# 1 - bash code
define colon-if-empty
$(if $(strip $1),$1,:)
endef

# Used by generate-simple-rule, see its docs.
define simple-rule-template
$1: $2 $(if $(strip $3),| $3)
	$(call colon-if-empty,$4)
endef

# Generates a simple rule - without variable forwarding and with only
# a single-command recipe.
# 1 - targets
# 2 - reqs
# 3 - order-only reqs
# 4 - recipe
define generate-simple-rule
$(eval $(call simple-rule-template,$1,$2,$3,$4))
endef

# Generates a rule with a "set -e".
# 1 - target (stamp file)
# 2 - reqs
# 3 - order-only reqs
# 4 - recipe placed after 'set -e'
define generate-strict-rule
$(call generate-simple-rule,$1,$2,$3,$(VQ)set -e; $(call colon-if-empty,$4))
endef

# Generates a rule for creating a stamp with additional actions to be
# performed before the actual stamp creation.
# 1 - target (stamp file)
# 2 - reqs
# 3 - order-only reqs
# 4 - recipe placed between 'set -e' and 'touch "$@"'
define generate-stamp-rule
$(call generate-strict-rule,$1,$2,$3,$(call colon-if-empty,$4); $$(call vb,v2,STAMP,$$(call vsp,$$@))touch "$$@")
endef

# 1 - from
# 2 - to
define bash-cond-rename
if cmp --silent "$1" "$2"; then rm -f "$1"; else mv "$1" "$2"; fi
endef

# Generates a rule for generating a depmk for a given go binary from a
# given package. It also tries to include the depmk.
#
# This function (and the other generate-*-deps) are stamp-based. It
# generates no rule for actual depmk. Instead it generates a rule for
# creating a stamp, which will also generate the depmk. This is to
# avoid generating depmk at make startup, when it parses the
# Makefile. At startup, make tries to rebuild all the files it tries
# to include if there is a rule for the file. We do not want that -
# that would override a depmk with a fresh one, so no file
# additions/deletions made before running make would be detected.
#
# 1 - a stamp file
# 2 - a binary name
# 3 - depmk name
# 4 - a package name
define generate-go-deps
$(strip \
	$(if $(call equal,$2,$(DEPSGENTOOL)), \
		$(eval _MISC_GGD_DEP_ := $(DEPSGENTOOL)), \
		$(eval _MISC_GGD_DEP_ := $(DEPSGENTOOL_STAMP))) \
	$(eval -include $3) \
	$(eval $(call generate-stamp-rule,$1,$(_MISC_GGD_DEP_),$(DEPSDIR),$(call vb,v2,DEPS GO,$(call vsg,$(REPO_PATH)/$4) => $(call vsp,$3))$(GO_ENV) "$(DEPSGENTOOL)" go --repo "$(REPO_PATH)" --module "$4" --target '$2 $1' >"$3.tmp"; $(call bash-cond-rename,$3.tmp,$3))) \
	$(call undefine-namespaces,_MISC_GGD))
endef

# Generates a rule for generating a key-value depmk for a given target
# with given variable names to store. It also tries to include the
# depmk.
# 1 - a stamp file
# 2 - a target
# 3 - depmk name
# 4 - a list of variable names to store
define generate-kv-deps
$(strip \
	$(if $(call equal,$2,$(DEPSGENTOOL)), \
		$(eval _MISC_GKD_DEP_ := $(DEPSGENTOOL)), \
		$(eval _MISC_GKD_DEP_ := $(DEPSGENTOOL_STAMP))) \
	$(foreach v,$4, \
		$(eval _MISC_GKD_KV_ += $v $(call escape-and-wrap,$($v)))) \
	$(eval -include $3) \
	$(eval $(call generate-stamp-rule,$1,$(_MISC_GKD_DEP_),$(DEPSDIR),$(call vb,v2,DEPS KV,$4 => $(call vsp,$3))"$(DEPSGENTOOL)" kv --target '$2 $1' $(_MISC_GKD_KV_) >"$3.tmp"; $(call bash-cond-rename,$3.tmp,$3))) \
	$(call undefine-namespaces,_MISC_GKD))
endef

define filelist-file
$(FILELISTDIR)/$(call path-to-file-with-suffix,$1,filelist)
endef

define setup-custom-filelist-file
$(eval $1 := $(call filelist-file,$2)) \
$(eval $($1): | $$(call to-dir,$($1))) \
$(eval CLEAN_FILES += $($1))
endef

define setup-filelist-file
$(eval $(call setup-custom-filelist-file,$1,$(MK_PATH)$2))
endef

# 1 - filelist
define generate-empty-filelist
$(eval $(call generate-strict-rule,$1,$(FILELISTGENTOOL_STAMP),$(FILELISTDIR),$(call vb,v2,FILELIST,<nothing> => $(call vsp,$1))"$(FILELISTGENTOOL)" --empty >"$1.tmp"; $(call bash-cond-rename,$1.tmp,$1)))
endef

# 1 - filelist
# 2 - directory
define generate-deep-filelist
$(eval $(call generate-strict-rule,$1,$(FILELISTGENTOOL_STAMP),$(FILELISTDIR),$(call vb,v2,FILELIST,$(call vsp,$2) => $(call vsp,$1))"$(FILELISTGENTOOL)" --directory="$2" >"$1.tmp"; $(call bash-cond-rename,$1.tmp,$1)))
endef

# 1 - filelist
# 2 - a directory
# 3 - a suffix
define generate-shallow-filelist
$(eval $(call generate-strict-rule,$1,$(FILELISTGENTOOL_STAMP),$(FILELISTDIR),$(call vb,v2,FILELIST,$(call vsp,$2/*$3) => $(call vsp,$1))"$(FILELISTGENTOOL)" --directory="$2" --suffix="$3" >"$1.tmp"; $(call bash-cond-rename,$1.tmp,$1)))
endef

# This is used for the truncated output - it takes a list of
# directories, shortens them and puts them in a comma-separated list
# between parens: (dir1,dir2,...,dirN)
define nice-dirs-output
$(strip \
	$(eval _MISC_NDO_COMMA_ := ,) \
	$(eval _MISC_NDO_SPACE_ := ) \
	$(eval _MISC_NDO_SPACE_ += ) \
	$(eval _MISC_NDO_DIRS_ := ($(subst $(_MISC_NDO_SPACE_),$(_MISC_NDO_COMMA_),$(call vsp,$1)))) \
	$(_MISC_NDO_DIRS_) \
	$(call undefine-namespaces,_MISC_NDO))
endef

# Generates a rule for generating a glob depmk for a given target
# based on a given filelist. This is up to you to ensure that every
# file in a filelist ends with a given suffix.
# 1 - a stamp file
# 2 - a target
# 3 - depmk name
# 4 - a suffix
# 5 - a filelist
# 6 - a list of directories to map the files from filelist to
# 7 - glob mode, can be all, dot-files, normal (empty means all)
define generate-glob-deps
$(strip \
	$(if $(call equal,$2,$(DEPSGENTOOL)), \
		$(eval _MISC_GLDF_DEP_ := $(DEPSGENTOOL)), \
		$(eval _MISC_GLDF_DEP_ := $(DEPSGENTOOL_STAMP))) \
	$(if $(strip $7), \
		$(eval _MISC_GLDF_GLOB_ := --glob-mode="$(strip $7)")) \
	$(eval -include $3) \
	$(eval _MISC_GLDF_DIRS_ := $(call nice-dirs-output,$6)) \
	$(eval $(call generate-stamp-rule,$1,$(_MISC_GLDF_DEP_) $5,$(DEPSDIR),$(call vb,v2,DEPS GLOB,$(_MISC_GLDF_DIRS_) $(call vsp,$5) => $(call vsp,$3))"$(DEPSGENTOOL)" glob --target "$2 $1" --suffix="$4" --filelist="$5" $(foreach m,$6,--map-to="$m") $(_MISC_GLDF_GLOB_)>"$3.tmp"; $(call bash-cond-rename,$3.tmp,$3))) \
	$(call undefine-namespaces,_MISC_GLDF))
endef

# Returns a list of directories starting from a subdirectory of given
# base up to the full path made of the given base and a rest. So, if
# base is "base" and rest is "r/e/s/t" then the returned list is
# "base/r base/r/e base/r/e/s base/r/e/s/t".
#
# Useful for getting a list of directories to be removed.
# 1 - a base
# 2 - a dir (or dirs) in base
#
# Example: CREATE_DIRS += $(call dir-chain,$(BASE),src/foo/bar/baz)
define dir-chain
$(strip \
	$(eval _MISC_DC_DIRS_ := $(subst /, ,$2)) \
	$(eval _MISC_DC_PATHS_ :=) \
	$(eval _MISC_DC_LIST_ :=) \
	$(eval $(foreach d,$(_MISC_DC_DIRS_), \
		$(eval _MISC_DC_LAST_ := $(lastword $(_MISC_DC_PATHS_))) \
		$(eval $(if $(_MISC_DC_LAST_), \
			$(eval _MISC_DC_PATHS_ += $(_MISC_DC_LAST_)/$d), \
			$(eval _MISC_DC_PATHS_ := $d))))) \
	$(eval $(foreach d,$(_MISC_DC_PATHS_), \
		$(eval _MISC_DC_LIST_ += $1/$d))) \
	$(_MISC_DC_LIST_) \
	$(call undefine-namespaces,_MISC_DC))
endef

# 1 - variable for dirname
define setup-tmp-dir
$(strip \
	$(eval _MISC_TMP_DIR_ := $(MAINTEMPDIR)/$(subst .mk,,$(MK_FILENAME)))
	$(eval CREATE_DIRS += $(_MISC_TMP_DIR_)) \
	$(eval $1 := $(_MISC_TMP_DIR_)) \
	$(call undefine-namespaces,_MISC_TMP))
endef

define clean-file
$(CLEANDIR)/$(call path-to-file-with-suffix,$1,clean.mk)
endef

define setup-custom-clean-file
$(eval $1 := $(call clean-file,$2)) \
$(eval $($1): | $$(call to-dir,$($1))) \
$(eval CLEAN_FILES += $($1))
endef

define setup-clean-file
$(eval $(call setup-custom-clean-file,$1,$(MK_PATH)$2))
endef

# 1 - stamp file
# 2 - cleanmk file
# 3 - filelist name
# 4 - a list of directory mappings
define generate-clean-mk
$(strip \
	$(eval -include $2) \
	$(eval _MISC_GCM_DIRS_ := $(call nice-dirs-output,$4)) \
	$(eval $(call generate-stamp-rule,$1,$(CLEANGENTOOL_STAMP) $3,$(CLEANDIR),$(call vb,v2,CLEANFILE,$(_MISC_GCM_DIRS_) $(call vsp,$3) => $(call vsp,$2))"$(CLEANGENTOOL)" --filelist="$3" $(foreach m,$4,--map-to="$m") >"$2.tmp"; $(call bash-cond-rename,$2.tmp,$2))) \
	$(call undefine-namespaces,_MISC_GCM))
endef

define sed-replacement-escape
$(strip $(shell echo $1 | sed -e 's/[\/&]/\\&/g'))
endef

define add-dependency-template
$1: $2
endef

# 1 - a target
# 2 - a dependency (or a prerequisite in makese)
define add-dependency
$(eval $(call add-dependency-template,$1,$2))
endef

# 1 - stamp file, which will depend on the generated clean stamp
# 2 - file list
# 3 - a list of directory mappings
# 4 - descriptor
define generate-clean-mk-from-filelist
$(strip \
	$(eval _MISC_GCMFF_MAIN_STAMP_ := $(strip $1)) \
	$(eval _MISC_GCMFF_FILELIST_ := $(strip $2)) \
	$(eval _MISC_GCMFF_DIR_MAPS_ := $(strip $3)) \
	$(eval _MISC_GCMFF_DESCRIPTOR_ := $(strip $4)) \
	\
	$(call setup-stamp-file,_MISC_GCMFF_CLEAN_STAMP_,$(_MISC_GCMFF_DESCRIPTOR_)-gcmff-generated-clean-stamp) \
	$(call setup-clean-file,_MISC_GCMFF_CLEANMK_,$(_MISC_GCMFF_DESCRIPTOR_)-gcmff-generated-cleanmk) \
	\
	$(call add-dependency,$(_MISC_GCMFF_MAIN_STAMP_),$(_MISC_GCMFF_CLEAN_STAMP_)) \
	$(call generate-clean-mk,$(_MISC_GCMFF_CLEAN_STAMP_),$(_MISC_GCMFF_CLEANMK_),$(_MISC_GCMFF_FILELIST_),$(_MISC_GCMFF_DIR_MAPS_)) \
	\
	$(call undefine-namespaces,_MISC_GCMFF))
endef

# 1 - stamp file, which will depend on the generated clean stamp
# 2 - source directory
# 3 - a list of directory mappings
# 4 - filelist deps
# 5 - descriptor
define generate-clean-mk-simple
$(strip \
	$(eval _MISC_GCMS_MAIN_STAMP_ := $(strip $1)) \
	$(eval _MISC_GCMS_SRCDIR_ := $(strip $2)) \
	$(eval _MISC_GCMS_DIR_MAPS_ := $(strip $3)) \
	$(eval _MISC_GCMS_DEPS_ := $(strip $4)) \
	$(eval _MISC_GCMS_DESCRIPTOR_ := $(strip $5)) \
	\
	$(call setup-filelist-file,_MISC_GCMS_FILELIST_,$(_MISC_GCMS_DESCRIPTOR_)-gcms-generated-filelist) \
	$(call add-dependency,$(_MISC_GCMS_FILELIST_),$(_MISC_GCMS_DEPS_)) \
	$(call generate-deep-filelist,$(_MISC_GCMS_FILELIST_),$(_MISC_GCMS_SRCDIR_)) \
	\
	$(call generate-clean-mk-from-filelist, \
		$(_MISC_GCMS_MAIN_STAMP_), \
		$(_MISC_GCMS_FILELIST_), \
		$(_MISC_GCMS_DIR_MAPS_), \
		$(_MISC_GCMS_DESCRIPTOR_)) \
	\
	$(call undefine-namespaces,_MISC_GCMS))
endef

# Formats given lists of source and destination files for the
# INSTALL_FILES variable.
#
# 1 - list of src files
# 2 - list of target files
# 3 - mode
define install-file-triplets
$(strip $(join $(addsuffix :,$1),$(addsuffix :$3,$2)))
endef

define commas-to-spaces
$(strip \
	$(eval _MISC_CTS_COMMA_ := ,) \
	$(eval _MISC_CTS_SPACE_ := ) \
	$(eval _MISC_CTS_SPACE_ += ) \
	$(subst $(_MISC_CTS_COMMA_),$(_MISC_CTS_SPACE_),$1) \
	$(call undefine-namespaces,_MISC_CTS))
endef

# Generates a rule for given stamp which removes given directory and
# adds a dependency to the directory on a given stamp. That way, the
# directory will be removed before it is created if the stamp does not
# exist or is invalidated. Additional dependencies for the stamp can
# be specified by using usual make syntax ($(stamp): $(dep)).
#
# 1 - stamp
# 2 - directory to remove
define generate-rm-dir-rule
$(strip \
	$(call add-dependency,$2,$1) \
	$(call generate-stamp-rule,$1,,, \
		$(call vb,v2,RM RF,$(call vsp,$2))rm -rf $2))
endef

define go-pkg-from-dir
$(subst $(MK_TOPLEVEL_SRCDIR)/,,$(MK_SRCDIR))
endef

# Generate a filelist for patches. Usually, when generating filelists,
# we require the directory to exist. In this case, the patches
# directory may not exist and it is fine. We generate an empty
# filelist.
#
# 1 - patches filelist
# 2 - patches dir
define generate-patches-filelist
$(strip \
	$(eval _MISC_GPF_FILELIST := $(strip $1)) \
	$(eval _MISC_GPF_DIR := $(strip $2)) \
	$(eval $(if $(shell test -d "$(_MISC_GPF_DIR)" && echo yes), \
		$(call generate-shallow-filelist,$(_MISC_GPF_FILELIST),$(_MISC_GPF_DIR),.patch), \
		$(call generate-empty-filelist,$(_MISC_GPF_FILELIST)))) \
	$(call undefine-namespaces,_MISC_GPF))
endef

# Recursive wildcard - recursively generate a list of files
#
# 1 - root directory
# 2 - filter pattern
define rwildcard
$(foreach d, $(wildcard $1/*), \
	$(filter $(subst *, %, $2), $d) $(call rwildcard, $d, $2))
endef

# Recursively list all files in a given path
# This just shells out to `find`.
#
# 1 - the path (file or directory)
define rlist-files
$(strip \
	$(eval _MISC_RLIST_OP1_ := $(call sq_escape,$1)) \
	$(eval _MISC_RLIST_FILES_ := $(shell find $(_MISC_RLIST_OP1_) -type f)) \
	$(_MISC_RLIST_FILES_))
endef
