API_JSON = $(wildcard */*/*-api.json */*/*/*-api.json)

# Download all API specifications and rebuild Go bindings.
# All downloaded files are cached in $TMPDIR for reuse with 'cached' below.
all: generator
	$(GOPATH)/bin/google-api-go-generator -cache=false -install -api=*

# Reuse cached API specifications in $TMPDIR and rebuild Go bindings.
cached: generator
	$(GOPATH)/bin/google-api-go-generator -cache=true -install -api=*

# Only rebuild Go bindings, do not modify API specifications.
# For every existing */*/*-api.json file, */*/*-gen.go will be built.
local: $(API_JSON:-api.json=-gen.go)

# Pattern rule for the 'local' target.
# Translates otherwise unnamed targets with a -gen.go suffix into the
# matching input file with a -api.json suffix. $< is the input file.
%-gen.go: %-api.json generator
	$(GOPATH)/bin/google-api-go-generator -api_json_file=$<

# Rebuild and install $(GOPATH)/bin/google-api-go-generator
generator:
	go install google.golang.org/api/googleapi
	go install google.golang.org/api/google-api-go-generator

.PHONY: all cached local generator
