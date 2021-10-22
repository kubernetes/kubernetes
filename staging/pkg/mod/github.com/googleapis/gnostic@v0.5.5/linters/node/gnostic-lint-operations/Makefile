
GNOSTIC = $(GOPATH)/src/github.com/googleapis/gnostic

plugin:
	node_modules/.bin/pbjs -t json \
		$(GNOSTIC)/OpenAPIv2/OpenAPIv2.proto \
		$(GNOSTIC)/OpenAPIv3/OpenAPIv3.proto \
		$(GNOSTIC)/discovery/discovery.proto \
		$(GNOSTIC)/surface/surface.proto \
		$(GNOSTIC)/plugins/plugin.proto \
		> bundle.json
	node_modules/.bin/nexe gnostic-lint-operations.js

run:	plugin
	gnostic $(GNOSTIC)/examples/v2.0/yaml/petstore.yaml --lint-operations

setup:
	npm install protobufjs
	npm install get-stdin
	npm install nexe
