all: build

CSI_SPEC := spec.md
CSI_PROTO := csi.proto

# This is the target for building the temporary CSI protobuf file.
#
# The temporary file is not versioned, and thus will always be
# built on Travis-CI.
$(CSI_PROTO).tmp: $(CSI_SPEC)
	cat $? | \
	  sed -n -e '/```protobuf$$/,/```$$/ p' | \
	  sed -e 's@^```.*$$@////////@g' > $@

# This is the target for building the CSI protobuf file.
#
# This target depends on its temp file, which is not versioned.
# Therefore when built on Travis-CI the temp file will always
# be built and trigger this target. On Travis-CI the temp file
# is compared with the real file, and if they differ the build
# will fail.
#
# Locally the temp file is simply copied over the real file.
$(CSI_PROTO): $(CSI_PROTO).tmp
ifeq (true,$(TRAVIS))
	diff "$@" "$?"
else
	diff "$@" "$?" > /dev/null 2>&1 || cp -f "$?" "$@"
endif

build: check

# If this is not running on Travis-CI then for sake of convenience
# go ahead and update the language bindings as well.
ifneq (true,$(TRAVIS))
build:
	$(MAKE) -C lib/go
	$(MAKE) -C lib/cxx
endif

clean:

clobber: clean
	rm -f $(CSI_PROTO) $(CSI_PROTO).tmp

# check generated files for violation of standards
check: $(CSI_PROTO)
	awk '{ if (length > 72) print NR, $$0 }' $? | diff - /dev/null

.PHONY: clean clobber check
