# inputs cleared after including this file:
# BSCB_BINARY: path of a built binary
# BSCB_SOURCES: sources used to build a binary
# BSCB_HEADERS: headers used to build a binary
# BSCB_ADDITIONAL_CFLAGS: additional CFLAGS passed to CC, just after CFLAGS

# misc inputs (usually provided by default):
# CC - C compiler
# CFLAGS - flags passed to CC.

_BSCB_PATH_ := $(lastword $(MAKEFILE_LIST))

$(call forward-vars,$(BSCB_BINARY), \
	CC CFLAGS BSCB_ADDITIONAL_CFLAGS BSCB_SOURCES)
$(BSCB_BINARY): $(BSCB_SOURCES) $(BSCB_HEADERS)
$(BSCB_BINARY): $(_BSCB_PATH_)
	$(VQ) \
	$(call vb,vt,CC,$(call vsp,$@)) \
	$(CC) $(CFLAGS) $(BSCB_ADDITIONAL_CFLAGS) -o "$@" $(BSCB_SOURCES) -static -s

CLEAN_FILES += $(BSCB_BINARY)

$(call undefine-namespaces,BSCB _BSCB)
