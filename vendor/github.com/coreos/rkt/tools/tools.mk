# depsgentool is special (building other tools require it), so it has
# to be included first
$(call inc-one,depsgentool.mk)
$(call inc-many,actool.mk \
	filelistgentool.mk \
	cleangentool.mk \
	quickrmtool.mk \
	rkt-monitor.mk \
	log-stresser.mk \
	cpu-stresser.mk \
	mem-stresser.mk \
	sleeper.mk)
