# protoc-gen-openapi

This directory contains a protoc plugin that generates an
OpenAPI description for a REST API that corresponds to a
Protocol Buffer service.

Installation:

        go get github.com/googleapis/gnostic
        go install github.com/googleapis/gnostic/apps/protoc-gen-openapi
  
  
Usage:

	protoc sample.proto -I. --openapi_out=.

This runs the plugin for a file named `sample.proto` which 
refers to additional .proto files in the same directory as
`sample.proto`. Output is written to the current directory.

