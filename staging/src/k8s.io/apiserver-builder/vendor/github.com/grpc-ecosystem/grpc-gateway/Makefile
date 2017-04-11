# This is a Makefile which maintains files automatically generated but to be
# shipped together with other files.
# You don't have to rebuild these targets by yourself unless you develop
# grpc-gateway itself.

PKG=github.com/grpc-ecosystem/grpc-gateway
GO_PLUGIN=bin/protoc-gen-go
GO_PLUGIN_PKG=github.com/golang/protobuf/protoc-gen-go
SWAGGER_PLUGIN=bin/protoc-gen-swagger
SWAGGER_PLUGIN_SRC= utilities/doc.go \
		    utilities/pattern.go \
		    utilities/trie.go \
		    protoc-gen-swagger/genswagger/generator.go \
		    protoc-gen-swagger/genswagger/template.go \
		    protoc-gen-swagger/main.go
SWAGGER_PLUGIN_PKG=$(PKG)/protoc-gen-swagger
GATEWAY_PLUGIN=bin/protoc-gen-grpc-gateway
GATEWAY_PLUGIN_PKG=$(PKG)/protoc-gen-grpc-gateway
GATEWAY_PLUGIN_SRC= utilities/doc.go \
		    utilities/pattern.go \
		    utilities/trie.go \
		    protoc-gen-grpc-gateway \
		    protoc-gen-grpc-gateway/descriptor \
		    protoc-gen-grpc-gateway/descriptor/registry.go \
		    protoc-gen-grpc-gateway/descriptor/services.go \
		    protoc-gen-grpc-gateway/descriptor/types.go \
		    protoc-gen-grpc-gateway/generator \
		    protoc-gen-grpc-gateway/generator/generator.go \
		    protoc-gen-grpc-gateway/gengateway \
		    protoc-gen-grpc-gateway/gengateway/doc.go \
		    protoc-gen-grpc-gateway/gengateway/generator.go \
		    protoc-gen-grpc-gateway/gengateway/template.go \
		    protoc-gen-grpc-gateway/httprule \
		    protoc-gen-grpc-gateway/httprule/compile.go \
		    protoc-gen-grpc-gateway/httprule/parse.go \
		    protoc-gen-grpc-gateway/httprule/types.go \
		    protoc-gen-grpc-gateway/main.go

GOOGLEAPIS_DIR=third_party/googleapis
OPTIONS_PROTO=$(GOOGLEAPIS_DIR)/google/api/annotations.proto $(GOOGLEAPIS_DIR)/google/api/http.proto
OPTIONS_GO=$(OPTIONS_PROTO:.proto=.pb.go)
OUTPUT_DIR=_output

RUNTIME_PROTO=runtime/internal/stream_chunk.proto
RUNTIME_GO=$(RUNTIME_PROTO:.proto=.pb.go)

PKGMAP=Mgoogle/protobuf/descriptor.proto=$(GO_PLUGIN_PKG)/descriptor,Mgoogle/api/annotations.proto=$(PKG)/$(GOOGLEAPIS_DIR)/google/api,Mexamples/sub/message.proto=$(PKG)/examples/sub
SWAGGER_EXAMPLES=examples/examplepb/echo_service.proto \
	 examples/examplepb/a_bit_of_everything.proto
EXAMPLES=examples/examplepb/echo_service.proto \
	 examples/examplepb/a_bit_of_everything.proto \
	 examples/examplepb/stream.proto \
	 examples/examplepb/flow_combination.proto
EXAMPLE_SVCSRCS=$(EXAMPLES:.proto=.pb.go)
EXAMPLE_GWSRCS=$(EXAMPLES:.proto=.pb.gw.go)
EXAMPLE_SWAGGERSRCS=$(EXAMPLES:.proto=.swagger.json)
EXAMPLE_DEPS=examples/sub/message.proto examples/sub2/message.proto
EXAMPLE_DEPSRCS=$(EXAMPLE_DEPS:.proto=.pb.go)

EXAMPLE_CLIENT_DIR=examples/clients
ECHO_EXAMPLE_SPEC=examples/examplepb/echo_service.swagger.json
ECHO_EXAMPLE_SRCS=$(EXAMPLE_CLIENT_DIR)/echo/EchoServiceApi.go \
		  $(EXAMPLE_CLIENT_DIR)/echo/ExamplepbSimpleMessage.go
ABE_EXAMPLE_SPEC=examples/examplepb/a_bit_of_everything.swagger.json
ABE_EXAMPLE_SRCS=$(EXAMPLE_CLIENT_DIR)/abe/ABitOfEverythingServiceApi.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/ABitOfEverythingNested.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/ExamplepbABitOfEverything.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/ExamplepbNumericEnum.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/ExamplepbIdMessage.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/NestedDeepEnum.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/ProtobufEmpty.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/Sub2IdMessage.go \
		 $(EXAMPLE_CLIENT_DIR)/abe/SubStringMessage.go
EXAMPLE_CLIENT_SRCS=$(ECHO_EXAMPLE_SRCS) $(ABE_EXAMPLE_SRCS)
SWAGGER_CODEGEN=swagger-codegen

PROTOC_INC_PATH=$(dir $(shell which protoc))/../include

generate: $(OPTIONS_GO) $(RUNTIME_GO)

.SUFFIXES: .go .proto

$(GO_PLUGIN): 
	go get $(GO_PLUGIN_PKG)
	go build -o $@ $(GO_PLUGIN_PKG)

$(OPTIONS_GO): $(OPTIONS_PROTO) $(GO_PLUGIN)
	protoc -I $(PROTOC_INC_PATH) -I$(GOOGLEAPIS_DIR) --plugin=$(GO_PLUGIN) --go_out=$(PKGMAP):$(GOOGLEAPIS_DIR) $(OPTIONS_PROTO)
$(RUNTIME_GO): $(RUNTIME_PROTO) $(GO_PLUGIN)
	protoc -I $(PROTOC_INC_PATH) --plugin=$(GO_PLUGIN) -I. --go_out=$(PKGMAP):. $(RUNTIME_PROTO)

$(GATEWAY_PLUGIN): $(OPTIONS_GO) $(RUNTIME_GO) $(GATEWAY_PLUGIN_SRC)
	go build -o $@ $(GATEWAY_PLUGIN_PKG)

$(SWAGGER_PLUGIN): $(OPTIONS_GO) $(SWAGGER_PLUGIN_SRC)
	go build -o $@ $(SWAGGER_PLUGIN_PKG)

$(EXAMPLE_SVCSRCS): $(GO_PLUGIN) $(EXAMPLES)
	protoc -I $(PROTOC_INC_PATH) -I. -I$(GOOGLEAPIS_DIR) --plugin=$(GO_PLUGIN) --go_out=$(PKGMAP),plugins=grpc:. $(EXAMPLES)
$(EXAMPLE_DEPSRCS): $(GO_PLUGIN) $(EXAMPLE_DEPS)
	mkdir -p $(OUTPUT_DIR)
	protoc -I $(PROTOC_INC_PATH) -I. --plugin=$(GO_PLUGIN) --go_out=$(PKGMAP),plugins=grpc:$(OUTPUT_DIR) $(@:.pb.go=.proto)
	cp $(OUTPUT_DIR)/$(PKG)/$@ $@ || cp $(OUTPUT_DIR)/$@ $@
$(EXAMPLE_GWSRCS): $(GATEWAY_PLUGIN) $(EXAMPLES)
	protoc -I $(PROTOC_INC_PATH) -I. -I$(GOOGLEAPIS_DIR) --plugin=$(GATEWAY_PLUGIN) --grpc-gateway_out=logtostderr=true,$(PKGMAP):. $(EXAMPLES)
$(EXAMPLE_SWAGGERSRCS): $(SWAGGER_PLUGIN) $(SWAGGER_EXAMPLES)
	protoc -I $(PROTOC_INC_PATH) -I. -I$(GOOGLEAPIS_DIR) --plugin=$(SWAGGER_PLUGIN) --swagger_out=logtostderr=true,$(PKGMAP):. $(SWAGGER_EXAMPLES)

$(ECHO_EXAMPLE_SRCS): $(ECHO_EXAMPLE_SPEC)
	$(SWAGGER_CODEGEN) generate -i $(ECHO_EXAMPLE_SPEC) \
	    -l go -o examples/clients --additional-properties packageName=echo
	@rm -f $(EXAMPLE_CLIENT_DIR)/README.md $(EXAMPLE_CLIENT_DIR)/git_push.sh $(EXAMPLE_CLIENT_DIR)/.gitignore
$(ABE_EXAMPLE_SRCS): $(ABE_EXAMPLE_SPEC)
	$(SWAGGER_CODEGEN) generate -i $(ABE_EXAMPLE_SPEC) \
	    -l go -o examples/clients --additional-properties packageName=abe
	@rm -f $(EXAMPLE_CLIENT_DIR)/README.md $(EXAMPLE_CLIENT_DIR)/git_push.sh $(EXAMPLE_CLIENT_DIR)/.gitignore

examples: $(EXAMPLE_SVCSRCS) $(EXAMPLE_GWSRCS) $(EXAMPLE_DEPSRCS) $(EXAMPLE_SWAGGERSRCS) $(EXAMPLE_CLIENT_SRCS)
test: examples
	go test -race $(PKG)/...

lint:
	golint --set_exit_status $(PKG)/runtime
	golint --set_exit_status $(PKG)/utilities/...
	golint --set_exit_status $(PKG)/protoc-gen-grpc-gateway/...
	golint --set_exit_status $(PKG)/protoc-gen-swagger/...
	go vet $(PKG)/runtime || true
	go vet $(PKG)/utilities/...
	go vet $(PKG)/protoc-gen-grpc-gateway/...
	go vet $(PKG)/protoc-gen-swagger/...

clean distclean:
	rm -f $(GATEWAY_PLUGIN)
realclean: distclean
	rm -f $(OPTIONS_GO)
	rm -f $(EXAMPLE_SVCSRCS) $(EXAMPLE_DEPSRCS)
	rm -f $(EXAMPLE_GWSRCS)
	rm -f $(EXAMPLE_SWAGGERSRCS)
	rm -f $(GO_PLUGIN)
	rm -f $(SWAGGER_PLUGIN)
	rm -f $(EXAMPLE_CLIENT_SRCS)

.PHONY: generate examples test lint clean distclean realclean
