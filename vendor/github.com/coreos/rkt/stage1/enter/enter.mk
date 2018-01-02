ETR_FLAVORS := $(filter-out kvm,$(STAGE1_FLAVORS))

ifneq ($(ETR_FLAVORS),)

ASSCB_FLAVORS := $(ETR_FLAVORS)
ASSCB_EXTRA_CFLAGS := $(RKT_DEFINES_FOR_ENTER)

$(foreach f,$(ASSCB_FLAVORS),$(eval STAGE1_ENTER_CMD_$f := /enter))

include stage1/makelib/aci_simple_static_c_bin.mk

endif

$(call undefine-namespaces,ETR)
