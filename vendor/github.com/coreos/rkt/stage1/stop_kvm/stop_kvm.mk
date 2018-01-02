STK_FLAVORS := $(filter kvm,$(STAGE1_FLAVORS))

ifneq ($(STK_FLAVORS),)

ASGB_FLAVORS := $(STK_FLAVORS)

$(foreach f,$(ASGB_FLAVORS),$(eval STAGE1_STOP_CMD_$f := /stop_kvm))

include stage1/makelib/aci_simple_go_bin.mk

endif

$(undefine-namespaces,STK)

