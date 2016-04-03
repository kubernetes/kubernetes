ETK_FLAVORS := $(filter kvm,$(STAGE1_FLAVORS))

ifneq ($(ETK_FLAVORS),)

ASGB_FLAVORS := $(ETK_FLAVORS)

$(foreach f,$(ASGB_FLAVORS),$(eval STAGE1_ENTER_CMD_$f := /enter_kvm))

include stage1/makelib/aci_simple_go_bin.mk

endif

$(undefine-namespaces,ETK)
