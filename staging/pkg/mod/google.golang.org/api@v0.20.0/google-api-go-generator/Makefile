# This Makefile is for library maintainers, not end-developers.
# All the generated code is already checked in.

API_JSON = $(wildcard ../*/*/*-api.json ../*/*/*/*-api.json)

GENERATOR=./google-api-go-generator

# Download all API specifications and rebuild Go bindings.
# All downloaded files are cached in $TMPDIR for reuse with 'cached' below.
all: SHELL:=/bin/bash
all: $(GENERATOR)
	cd ../internal/version; go generate
	$(GENERATOR) -cache=false -install -api=* -gendir=..
	go test . -v

# Reuse cached API specifications in $TMPDIR and rebuild Go bindings.
cached: $(GENERATOR)
	$(GENERATOR) -cache=true -install -api=* -gendir=..

# Only rebuild Go bindings, do not modify API specifications.
# For every existing */*/*-api.json file, */*/*-gen.go will be built.
local: $(API_JSON:-api.json=-gen.go)

# Pattern rule for the 'local' target.
# Translates otherwise unnamed targets with a -gen.go suffix into the
# matching input file with a -api.json suffix. $< is the input file.
%-gen.go: %-api.json $(GENERATOR)
	$(GENERATOR) -api_json_file=$< -output=$@

# Alias to rebuild and install $(GENERATOR)
generator: $(GENERATOR)

# Marked as .PHONY so that make always invokes go build.
$(GENERATOR):
	go build -o $@

.PHONY: all cached local generator $(GENERATOR)
