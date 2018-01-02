# This file does a shallow clone of a given repository into a given
# directory and does a checkout to a given branch. The end result is a
# pristine state of a repository.
#
# Inputs:
#
# GCL_REPOSITORY - a git repository to clone
#
# GCL_DIRECTORY - a directory where the repository is supposed to be cloned
#
# GCL_COMMITTISH - a committish in a repository we want to check
# out. A committish is basically either a SHA, or a tag or a branch.
# Be careful with using SHAs - shallow fetches of a specified SHA are
# quite a new feature on the server side and not all git providers
# have this implemented or enabled.
#
# GCL_EXPECTED_FILE - a file relative to GCL_DIRECTORY that is
# expected to exist after the repository is cloned
#
# GCL_TARGET - the target that will depend on GCL_EXPECTED_FILE
# directly or indirectly, depending on whether we want to fetch the
# changes from git repo or not, see GCL_DO_CHECK. Also will depend on
# some deps stamps.
#
# GCL_DO_CHECK - whether we should try to fetch the new changes from
# the repository and invalidate the GCL_TARGET in case of new changes
# availability; should not be used for SHA or tags.

_GCL_GIT_ := "$(GIT)" -C "$(GCL_DIRECTORY)"
_GCL_FULL_PATH_ := $(GCL_DIRECTORY)/$(GCL_EXPECTED_FILE)

# init, set remote, fetch, reset --hard and clean -ffdx
$(call forward-vars,$(_GCL_FULL_PATH_), \
	_GCL_GIT_ GCL_REPOSITORY GCL_DIRECTORY GCL_COMMITTISH)
$(_GCL_FULL_PATH_): | $(GCL_DIRECTORY)
	$(VQ) \
	set -e; \
	$(_GCL_GIT_) init $(call vl3,--quiet); \
	if ! $(_GCL_GIT_) remote | grep --silent origin; \
	then \
		$(_GCL_GIT_) remote add origin "$(GCL_REPOSITORY)"; \
	fi; \
	saved_committish=''; \
	if $(_GCL_GIT_) config --list | grep --silent '^rkt\.committish='; then \
		saved_committish="$$($(_GCL_GIT_) config rkt.committish)"; \
	fi; \
	if [ "$${saved_committish}" != "$(GCL_COMMITTISH)" ]; then \
		$(call vb,vt,GIT CLONE,$(GCL_REPOSITORY) ($(GCL_COMMITTISH)) => $(call vsp,$(GCL_DIRECTORY))) \
		$(_GCL_GIT_) fetch $(call vl3,--quiet) --depth=1 origin $(GCL_COMMITTISH); \
		$(_GCL_GIT_) config rkt.committish "$(GCL_COMMITTISH)"; \
		$(_GCL_GIT_) config rkt.fetch-head $$($(_GCL_GIT_) rev-parse FETCH_HEAD); \
	fi; \
	rev="$$($(_GCL_GIT_) config rkt.fetch-head)"; \
	$(call vb,vt,GIT RESET,$(call vsp,$(GCL_DIRECTORY)) => $(GCL_COMMITTISH)) \
	$(_GCL_GIT_) reset --hard $(call vl3,--quiet) "$${rev}"; \
	$(call vb,vt,GIT CLEAN,$(call vsp,$(GCL_DIRECTORY))) \
	$(_GCL_GIT_) clean -ffdx $(call vl3,--quiet); \
	human_rev="$$($(_GCL_GIT_) describe --always)"; \
	$(call vb,vt,GIT DESCRIBE,$(GCL_REPOSITORY) ($(GCL_COMMITTISH)) => "$${human_rev}") \
	touch "$@"

# remove the GCL_DIRECTORY if GCL_REPOSITORY changes
# also invalidate the _GCL_FULL_PATH_ as make seems not to check for
# nonexistence of a file after its prerequisites are remade
$(call setup-stamp-file,_GCL_RM_DIR_STAMP,gcl-$(GCL_DIRECTORY)-rm-dir)
$(call setup-stamp-file,_GCL_KV_DEPMK_STAMP,gcl-$(GCL_DIRECTORY)-kv-depmk)
$(call setup-dep-file,_GCL_KV_DEPMK,gcl-$(GCL_DIRECTORY)-kv)

$(call generate-rm-dir-rule,$(_GCL_RM_DIR_STAMP),$(GCL_DIRECTORY))
$(call generate-kv-deps,$(_GCL_KV_DEPMK_STAMP),$(_GCL_RM_DIR_STAMP) $(_GCL_FULL_PATH_),$(_GCL_KV_DEPMK),GCL_REPOSITORY)

$(GCL_TARGET): $(_GCL_KV_DEPMK_STAMP)

# invalidate the _GCL_FULL_PATH_ if GCL_COMMITTISH changed
$(call setup-stamp-file,_GCL_KV_COMMITTISH_DEPMK_STAMP,gcl-$(GCL_DIRECTORY)-kv-committish-depmk)
$(call setup-dep-file,_GCL_KV_COMMITTISH_DEPMK,gcl-$(GCL_DIRECTORY)-kv-committish)

$(call generate-kv-deps,$(_GCL_KV_COMMITTISH_DEPMK_STAMP),$(_GCL_FULL_PATH_),$(_GCL_KV_COMMITTISH_DEPMK),GCL_COMMITTISH)

$(GCL_TARGET): $(_GCL_KV_COMMITTISH_DEPMK_STAMP)

# perform updates if wanted
ifneq ($(GCL_DO_CHECK),)

GR_TARGET := $(GCL_TARGET)
GR_SRCDIR := $(GCL_DIRECTORY)
GR_BRANCH := $(GCL_COMMITTISH)
GR_PREREQS := $(_GCL_FULL_PATH_)

include makelib/git-refresh.mk

else

$(GCL_TARGET): $(_GCL_FULL_PATH_)

endif

$(call undefine-namespaces,GCL _GCL)
