STP_FLAVORS := $(filter-out kvm,$(STAGE1_FLAVORS))

ifneq ($(STP_FLAVORS),)

ASGB_FLAVORS := $(STP_FLAVORS)

$(foreach f,$(ASGB_FLAVORS),$(eval STAGE1_STOP_CMD_$f := /stop))

include stage1/makelib/aci_simple_go_bin.mk

endif

$(call undefine-namespaces,STP)
