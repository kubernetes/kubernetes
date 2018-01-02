# This file copies the libraries from the host to the ACI rootfs
# directory of a given flavor.
#
# Inputs:
#
# STAGE1_FSD_FLAVOR - a flavor which wants libraries from host

SLD_FLAVOR := $(STAGE1_FSD_FLAVOR)
$(call inc-one,stdlibdirs.mk)

# temporary directory
$(call setup-tmp-dir,STAGE1_FSD_TMPDIR)

# This is where libraries from host are copied too, so we can prepare
# an exact filelist of what was copied.
STAGE1_FSD_LIBSDIR := $(STAGE1_FSD_TMPDIR)/libs-$(STAGE1_FSD_FLAVOR)

# main stamp that makes sure that libs were copied and dependencies
# and cleanfiles were generated
$(call setup-stamp-file,STAGE1_FSD_STAMP,$(STAGE1_FSD_FLAVOR))
# this stamp makes sure that libs were copied
$(call setup-stamp-file,STAGE1_FSD_COPY_STAMP,/$(STAGE1_FSD_FLAVOR)-fsd_copy)

# the ACI rootfs for given flavor
STAGE1_FSD_ACIROOTFSDIR := $(STAGE1_ACIROOTFSDIR_$(STAGE1_FSD_FLAVOR))
# this is to get all stamps to make sure that the copying is done
# after everything was put into the ACI rootfs and just before running
# actool to create the ACI
STAGE1_FSD_ALL_STAMPS := $(STAGE1_ALL_STAMPS_$(STAGE1_FSD_FLAVOR))

STAGE1_FSD_SPACE :=
STAGE1_FSD_SPACE +=

# additions to LD_LIBRARY_PATH environment variable
STAGE1_FSD_LD_LIBRARY_PATH := $(subst $(STAGE1_FSD_SPACE),:,$(foreach l,$(SLD_LOCATIONS),$(STAGE1_FSD_ACIROOTFSDIR)/$l))

ifneq ($(LD_LIBRARY_PATH),)

# the LD_LIBRARY_PATH environment variable is not empty, so append its
# path to ours

STAGE1_FSD_LD_LIBRARY_PATH := $(STAGE1_FSD_LD_LIBRARY_PATH):$(LD_LIBRARY_PATH)

endif

INSTALL_DIRS += $(STAGE1_FSD_LIBSDIR):-

# this makes sure that everything is done
$(call generate-stamp-rule,$(STAGE1_FSD_STAMP),$(STAGE1_FSD_COPY_STAMP))

# this detects which libs need to be copied and copies them into two
# places - the libdirs in the ACI rootfs and a temporary directory at
# the same time.
$(call generate-stamp-rule,$(STAGE1_FSD_COPY_STAMP),$(STAGE1_FSD_ALL_STAMPS),$(STAGE1_FSD_LIBSDIR), \
	$(call vb,vt,FIND SO DEPS,$(STAGE1_FSD_FLAVOR)) \
	all_libs=$$$$(find "$(STAGE1_FSD_ACIROOTFSDIR)" -type f | xargs file | grep ELF | cut -f1 -d: | LD_LIBRARY_PATH="$(STAGE1_FSD_LD_LIBRARY_PATH)" xargs ldd | grep -v '^[^[:space:]]' | grep '/' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*(0x[0-9a-fA-F]*)//' -e 's/.*=>[[:space:]]*//' | grep -Fve "$(STAGE1_FSD_ACIROOTFSDIR)" | sort -u); \
	for f in $$$${all_libs}; do \
		$(INSTALL) -D "$$$${f}" "$(STAGE1_FSD_ACIROOTFSDIR)$$$${f}"; \
		$(INSTALL) -D "$$$${f}" "$(STAGE1_FSD_LIBSDIR)$$$${f}"; \
	done)

# Generate clean file cleaning libraries copied from the host to both
# the temporary directory and the ACI rootfs directory. Can happen
# only after the files were copied.
$(call generate-clean-mk-simple, \
	$(STAGE1_FSD_STAMP), \
	$(STAGE1_FSD_LIBSDIR), \
	$(STAGE1_FSD_LIBSDIR) $(STAGE1_FSD_ACIROOTFSDIR), \
	$(STAGE1_FSD_COPY_STAMP), \
	libs-cleanup)

# STAGE1_FSD_STAMP is deliberately not cleared - it will be used in
# stage1.mk to create the stage1.aci dependency on the stamp.
$(call undefine-namespaces,STAGE1_FSD _STAGE1_FSD,STAGE1_FSD_STAMP)
