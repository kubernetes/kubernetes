# Creates a rule creating directories.
# Example: $(eval $(call _FILE_OPS_CREATE_DIRS_RULE_,dir1 dir1/dir2 dir1/dir2/dir3))
define _FILE_OPS_CREATE_DIRS_RULE_
CLEAN_DIRS += $1
$1:
	$(VQ) \
	set -e; \
	if [[ ! -e "$$@" ]]; then \
		$(call vb,v2,MKDIR,$$(call vsp,$$@)) \
		mkdir -p "$$@"; \
	fi; \
	$$(call _FILE_OPS_BAIL_OUT_IF_NOT_DIR_,$$@)
endef

# Creates a rule for installing directory. Depends on parent
# directory.
# Example: $(eval $(call _FILE_OPS_CREATE_INSTALL_DIR_RULE_,dir:0644))
define _FILE_OPS_CREATE_INSTALL_DIR_RULE_
$$(eval $$(call _FILE_OPS_SPLIT_2_,$1,_FILE_OPS_CIDR_DIR_,_FILE_OPS_CIDR_MODE_))
CLEAN_DIRS += $$(_FILE_OPS_CIDR_DIR_)
$$(call forward-vars,$$(_FILE_OPS_CIDR_DIR_), \
	INSTALL _FILE_OPS_CIDR_MODE_)
# TODO: Create a proper dependency on parent directory
# $$(_FILE_OPS_CIDR_DIR_): | $$(call to-dir,$$(_FILE_OPS_CIDR_DIR_))
$$(_FILE_OPS_CIDR_DIR_):
	$(VQ) \
	set -e; \
	if [[ ! -e "$$@" ]]; then \
		$(call vb,v2,MKDIR,$$(call vsp,$$@)) \
		$$(INSTALL) $$(call _FILE_OPS_DASH_M_,$$(_FILE_OPS_CIDR_MODE_)) -d "$$@"; \
	fi;\
	$$(call _FILE_OPS_BAIL_OUT_IF_NOT_DIR_,$$@)
$$(call undefine-namespaces,_FILE_OPS_CIDR)
endef

# Creates a rule for installing a file. Depends on source file and
# parent directory. Pass - as a third parameter for dest to inherit
# mode from src.
# Example: $(eval $(call _FILE_OPS_CREATE_INSTALL_FILE_RULE_,src,dest,0755))
define _FILE_OPS_CREATE_INSTALL_FILE_RULE_
$$(eval $$(call _FILE_OPS_SPLIT_3_,$1,_FILE_OPS_CIFR_SRC_,_FILE_OPS_CIFR_DEST_,_FILE_OPS_CIFR_MODE_))
CLEAN_FILES += $$(_FILE_OPS_CIFR_DEST_)
$$(call forward-vars,$$(_FILE_OPS_CIFR_DEST_), \
	INSTALL _FILE_OPS_CIFR_MODE_)
$$(_FILE_OPS_CIFR_DEST_): $$(_FILE_OPS_CIFR_SRC_) | $$(call to-dir,$$(_FILE_OPS_CIFR_DEST_))
	$(VQ) \
	$(call vb,v2,CP,$$(call vsp,$$<) => $$(call vsp,$$@)) \
	$$(INSTALL) $$(call _FILE_OPS_DASH_M_,$$(_FILE_OPS_CIFR_MODE_)) "$$<" "$$@"
$$(call undefine-namespaces,_FILE_OPS_CIFR)
endef

# Creates a rule for installing a symlink. Depends on parent
# directory.
# Example: $(eval $(call _FILE_OPS_CREATE_INSTALL_FILE_RULE_,src,dest,0755))
define _FILE_OPS_CREATE_INSTALL_SYMLINK_RULE_
$$(eval $$(call _FILE_OPS_SPLIT_2_,$1,_FILE_OPS_CISR_TARGET_,_FILE_OPS_CISR_LINK_NAME_))
CLEAN_SYMLINKS += $$(_FILE_OPS_CISR_LINK_NAME_)
$$(call forward-vars,$$(_FILE_OPS_CISR_LINK_NAME_), \
	_FILE_OPS_CISR_TARGET_)
$$(_FILE_OPS_CISR_LINK_NAME_): | $$(call to-dir,$$(_FILE_OPS_CISR_LINK_NAME_))
	$(VQ) \
	set -e; \
	if [ -h "$$@" ]; then \
		tgt=$$$$(readlink "$$@"); \
		if [ "$$$${tgt}" != "$$(_FILE_OPS_CISR_TARGET_)" ]; then \
			echo "'$$@' is a symlink pointing to '$$$${tgt}' instead of '$$(_FILE_OPS_CISR_TARGET_)', bailing out" >&2; \
			exit 1; \
		fi; \
	elif [ -e "$$@" ]; then \
		echo "$$@ already exists and is not a symlink, bailing out" >&2; \
		exit 1; \
	else \
		$(call vb,v2,LN S,$$(_FILE_OPS_CISR_TARGET_) $$(call vsp,$$@)) \
		ln -s "$$(_FILE_OPS_CISR_TARGET_)" "$$@"; \
	fi
$$(call undefine-namespaces,_FILE_OPS_CISR)
endef

# Print an error if name is not a directory. To be used inside rules.
# Example $(call _FILE_OPS_BAIL_OUT_IF_NOT_DIR_,dir)
define _FILE_OPS_BAIL_OUT_IF_NOT_DIR_
if [[ ! -d "$1" ]]; then echo "$1 is not a directory, bailing out" >&2; exit 1; fi
endef

# Returns -m <foo> if foo is not a dash. Used for install invocations.
# Example: $(call _FILE_OPS_DASH_M_:0755)
define _FILE_OPS_DASH_M_
$(if $(filter-out -,$1),-m $1)
endef

define _FILE_OPS_SPLIT_2_COMMON_
$(eval _FILE_OPS_S_SPLITTED_ := $(subst :, ,$1)) \
$(eval $2 := $(word 1,$(_FILE_OPS_S_SPLITTED_))) \
$(eval $3 := $(word 2,$(_FILE_OPS_S_SPLITTED_)))
endef

define _FILE_OPS_SPLIT_2_
$(eval $(call _FILE_OPS_SPLIT_2_COMMON_,$1,$2,$3)) \
$(call undefine-namespaces,_FILE_OPS_S)
endef

define _FILE_OPS_SPLIT_3_
$(eval $(call _FILE_OPS_SPLIT_2_COMMON_,$1,$2,$3)) \
$(eval $4 := $(word 3,$(_FILE_OPS_S_SPLITTED_))) \
$(call undefine-namespaces,_FILE_OPS_S)
endef

# Special dir for storing lists of removed stuff. Sometimes the lists
# are too long for bash, so they need to be stored in files.
_FILE_OPS_DIR_ := $(BUILDDIR)/file_ops
_FILE_OPS_FILES_ := $(_FILE_OPS_DIR_)/files
_FILE_OPS_SYMLINKS_ := $(_FILE_OPS_DIR_)/symlinks
_FILE_OPS_DIRS_ := $(_FILE_OPS_DIR_)/dirs

CREATE_DIRS += $(_FILE_OPS_DIR_)
CLEAN_FILES += $(_FILE_OPS_FILES_) $(_FILE_OPS_SYMLINKS_) $(_FILE_OPS_DIRS_)

# generate rule for mkdir
$(eval $(call _FILE_OPS_CREATE_DIRS_RULE_,$(sort $(CREATE_DIRS))))

# generate rules for installing directories
$(foreach d,$(sort $(INSTALL_DIRS)), \
        $(eval $(call _FILE_OPS_CREATE_INSTALL_DIR_RULE_,$d)))

# generate rules for installing files
$(foreach f,$(sort $(INSTALL_FILES)), \
        $(eval $(call _FILE_OPS_CREATE_INSTALL_FILE_RULE_,$f)))

# generate rules for creating symlinks
$(foreach s,$(sort $(INSTALL_SYMLINKS)), \
        $(eval $(call _FILE_OPS_CREATE_INSTALL_SYMLINK_RULE_,$s)))

# $(file ...) function was introduced in GNU Make 4.0, but it received
# no entry in .FEATURES variable. So instead, we check the major
# version of GNU Make. If it is 3, then we will use some slower path
# for removing files. If it is 4, we will use the faster one.

_FILE_OPS_M_VERSION_ := $(strip $(shell $(MAKE) --version | grep '^GNU Make' | grep -o ' [[:digit:]]\+'))

# _FILE_OPS_WRITE_VAR_TO_FILE_ function
# 1 - variable
# 2 - filename

ifeq ($(_FILE_OPS_M_VERSION_),3)

# Ew.
define _FILE_OPS_WRITE_VAR_TO_FILE_
$(strip \
	$(eval list := $1) \
	$(eval file := $2) \
	$(eval count := $(words $(list))) \
	$(eval chunksize := 50) \
	$(eval fullchunknum := $(shell expr '$(count)' '/' '$(chunksize)')) \
	$(eval format := $(strip $(shell printf '%%s %0.s' {1..$(chunksize)}))) \
	$(shell printf '%d\n' '$(count)' >'$(file)') \
	$(foreach i,$(shell seq 1 '$(fullchunknum)'), \
		$(eval s := $(shell expr '(' '$i' - 1 ')' '*' '$(chunksize)' + 1)) \
		$(eval e := $(shell expr '$i' '*' '$(chunksize)')) \
		$(eval l := $(wordlist $s,$e,$(list))) \
		$(shell printf '$(format) ' $l >>'$(file)')) \
	$(eval rest := $(shell expr '$(count)' % '$(chunksize)')) \
	$(eval nonzero := $(shell expr '$(rest)' '>' 0)) \
	$(if $(filter 1,$(nonzero)), \
		$(eval format := $(strip $(shell printf '%%s %0.s' {1..$(rest)}))) \
		$(eval s := $(shell expr '$(count)' - '$(rest)' + 1)) \
		$(eval e := $(count)) \
		$(eval l := $(wordlist $s,$e,$(list))) \
		$(shell printf '$(format)' $l >>'$(file)')) \
	$(shell printf '\n' >>'$(file)'))
endef

else ifeq ($(_FILE_OPS_M_VERSION_),4)

define _FILE_OPS_WRITE_VAR_TO_FILE_
$(strip \
	$(eval \
		$(file >$2,$(words $1)) \
		$(file >>$2,$1)))
endef

else

$(error Unsupported major version of make: $(_FILE_OPS_M_VERSION_))

endif

$(call forward-vars,_file_ops_mk_clean_, \
	_FILE_OPS_FILES_ _FILE_OPS_SYMLINKS_ _FILE_OPS_DIRS_ QUICKRMTOOL)
_file_ops_mk_clean_: $(QUICKRMTOOL_STAMP) | $(_FILE_OPS_DIR_)
	$(info writing files)
	$(call _FILE_OPS_WRITE_VAR_TO_FILE_,$(CLEAN_FILES),$(_FILE_OPS_FILES_))
	$(info writing symlinks)
	$(call _FILE_OPS_WRITE_VAR_TO_FILE_,$(CLEAN_SYMLINKS),$(_FILE_OPS_SYMLINKS_))
	$(info writing dirs)
	$(call _FILE_OPS_WRITE_VAR_TO_FILE_,$(CLEAN_DIRS),$(_FILE_OPS_DIRS_))
	set -e; \
	echo "Removing everything"; \
	"$(QUICKRMTOOL)" --files="$(_FILE_OPS_FILES_)" --symlinks="$(_FILE_OPS_SYMLINKS_)" --dirs="$(_FILE_OPS_DIRS_)"

clean: _file_ops_mk_clean_

_FILE_OPS_ALL_DIRS_ := \
	$(CREATE_DIRS) \
	$(foreach d,$(INSTALL_DIRS),$(firstword $(subst :, ,$d))) \
	$(foreach s,$(INSTALL_SYMLINKS),$(lastword $(subst :, ,$s)))

.PHONY: $(_FILE_OPS_ALL_DIRS_) _file_ops_mk_clean_

# Excluding _FILE_OPS_BAIL_OUT_IF_NOT_DIR_, _FILE_OPS_DASH_M_ and
# _FILE_OPS_WRITE_VAR_TO_FILE_ because they are used inside
# recipes. Undefining them here would mean that inside recipes they
# would return empty value.
$(call undefine-namespaces,_FILE_OPS,_FILE_OPS_BAIL_OUT_IF_NOT_DIR_ \
	_FILE_OPS_DASH_M_ _FILE_OPS_WRITE_VAR_TO_FILE_)
