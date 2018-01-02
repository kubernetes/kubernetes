BGB_GO_FLAGS := $(strip \
	-ldflags "$(RKT_STAGE1_INTERPRETER_LDFLAGS)" \
	$(RKT_TAGS))
include stage1/makelib/aci_simple_go_bin.mk
