# aci directory should be the last one
_S1_SS_SUBDIRS_ := \
	appexec \
	prepare-app \
	enter \
	enter_kvm \
	net-plugins \
	net \
	init \
	gc \
	reaper \
	units \
	aci

$(call inc-many,$(foreach f,$(_S1_SS_SUBDIRS_),$f/$f.mk))

$(call undefine-namespaces,S1_SS _S1_SS)
