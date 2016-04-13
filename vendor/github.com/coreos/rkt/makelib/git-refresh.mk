# This file refreshes the given repository. If new changes are
# available in a given branch on remote repo, this does the hard reset
# and complete clean of the directory. Treat this file rather as an
# implementation detail of makelib/git.mk
#
# Inputs:
#
# GR_TARGET - target that will be invalidated if new changes are
# available.
#
# GR_PREREQS - stuff needed to be done before we can do the check
# (something like initial clone).
#
# GR_SRCDIR - repository directory
#
# GR_BRANCH - a branch to check

# Just some target, it is a phony target, so it does not need to exist
# or be meaningful in any way.
_GR_CONFIG_FORCE_CHECK_ := $(GR_SRCDIR)/FORCE_CHECK
# git with src dir as its working directory
_GR_GIT_ := "$(GIT)" -C "$(GR_SRCDIR)"
# git's config file
_GR_CONFIG_FILE_ := $(GR_SRCDIR)/.git/config

# target depends on git config file; we assume that updating the
# config with "git config" will update the timestamp of the config
# file, which in turn will invalidate the target
$(GR_TARGET): $(_GR_CONFIG_FILE_) $(GR_PREREQS)

# This checks if there are new changes in the upstream repository. If
# so, it updates the config file with new FETCH_HEAD.
#
# It depends on a phony target, so the check is always performed.
$(call forward-vars,$(_GR_CONFIG_FILE_), \
	_GR_GIT_ GR_BRANCH GR_SRCDIR)
$(_GR_CONFIG_FILE_): $(_GR_CONFIG_FORCE_CHECK_) $(GR_PREREQS)
	$(VQ) \
	set -e; \
	$(call vb,vt,GIT CHECK,$$($(_GR_GIT_) config remote.origin.url) ($(GR_BRANCH)) => $(call vsp,$(GR_SRCDIR))) \
	$(_GR_GIT_) fetch $(call vl3,--quiet) origin "$(GR_BRANCH)"; \
	old_rev="$$($(_GR_GIT_) config rkt.fetch-head)"; \
	new_rev="$$($(_GR_GIT_) rev-parse FETCH_HEAD)"; \
	if [ "$${old_rev}" != "$${new_rev}" ]; then \
		$(call vb,vt,GIT RESET,$(call vsp,$(GR_SRCDIR)) => $(GR_BRANCH)) \
		$(_GR_GIT_) reset --hard $(call vl3,--quiet) "$${new_rev}"; \
		$(call vb,vt,GIT CLEAN,$(call vsp,$(GR_SRCDIR))) \
		$(_GR_GIT_) clean -ffdx $(call vl3,--quiet); \
		$(_GR_GIT_) config rkt.fetch-head "$${new_rev}"; \
	fi

.PHONY: $(_GR_CONFIG_FORCE_CHECK_)

$(call undefine-namespaces,GR _GR)
