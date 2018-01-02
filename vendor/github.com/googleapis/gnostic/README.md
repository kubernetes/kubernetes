[![Build Status](https://travis-ci.org/googleapis/gnostic.svg?branch=master)](https://travis-ci.org/googleapis/gnostic)

# ⨁ gnostic

This repository contains a Go command line tool which converts
JSON and YAML [OpenAPI](https://github.com/OAI/OpenAPI-Specification)
descriptions to and from equivalent Protocol Buffer representations.

[Protocol Buffers](https://developers.google.com/protocol-buffers/)
provide a language-neutral, platform-neutral, extensible mechanism
for serializing structured data.
**gnostic**'s Protocol Buffer models for the OpenAPI Specification
can be used to generate code that includes data structures with 
explicit fields for the elements of an OpenAPI description.
This makes it possible for developers to work with OpenAPI
descriptions in type-safe ways, which is particularly useful
in strongly-typed languages like Go and Swift.

**gnostic** reads OpenAPI descriptions into
these generated data structures, reports errors,
resolves internal dependencies, and writes the results
in a binary form that can be used in any language that is
supported by the Protocol Buffer tools.
A plugin interface simplifies integration with API
tools written in a variety of different languages,
and when necessary, Protocol Buffer OpenAPI descriptions
can be reexported as JSON or YAML.

**gnostic** compilation code and OpenAPI Protocol Buffer
models are automatically generated from an
[OpenAPI JSON Schema](https://github.com/OAI/OpenAPI-Specification/blob/master/schemas/v2.0/schema.json).
Source code for the generator is in the [generate-gnostic](generate-gnostic) directory.

## Disclaimer

This is prerelease software and work in progress. Feedback and
contributions are welcome, but we currently make no guarantees of
function or stability.

## Requirements

**gnostic** can be run in any environment that supports [Go](http://golang.org)
and the [Google Protocol Buffer Compiler](https://github.com/google/protobuf).

## Installation

1. Get this package by downloading it with `go get`.

        go get github.com/googleapis/gnostic
  
2. [Optional] Build and run the compiler generator. 
This uses the OpenAPI JSON schema to generate a Protocol Buffer language file 
that describes the OpenAPI specification and a Go-language file of code that 
will read a JSON or YAML OpenAPI representation into the generated protocol 
buffers. Pre-generated versions of these files are in the OpenAPIv2 directory.

        cd $GOPATH/src/github.com/googleapis/gnostic/generate-gnostic
        go install
        cd ..
        generate-gnostic --v2

3. [Optional] Generate Protocol Buffer support code. 
A pre-generated version of this file is checked into the OpenAPIv2 directory.
This step requires a local installation of protoc, the Protocol Buffer Compiler.
You can get protoc [here](https://github.com/google/protobuf).

        ./COMPILE-PROTOS.sh

4. [Optional] Rebuild **gnostic**. This is only necessary if you've performed steps
2 or 3 above.

        go install github.com/googleapis/gnostic

5. Run **gnostic**. This will create a file in the current directory named "petstore.pb" that contains a binary
Protocol Buffer description of a sample API.

        gnostic --pb-out=. examples/petstore.json

6. You can also compile files that you specify with a URL. Here's another way to compile the previous 
example. This time we're creating "petstore.text", which contains a textual representation of the
Protocol Buffer description. This is mainly for use in testing and debugging.

        gnostic --text-out=petstore.text https://raw.githubusercontent.com/googleapis/gnostic/master/examples/petstore.json

7. For a sample application, see apps/report.

        go install github.com/googleapis/gnostic/apps/report
        report petstore.pb

8. **gnostic** supports plugins. This builds and runs a sample plugin
that reports some basic information about an API. The "-" causes the plugin to 
write its output to stdout.

        go install github.com/googleapis/gnostic/plugins/gnostic-go-sample
        gnostic examples/petstore.json --go-sample-out=-

## Copyright

Copyright 2017, Google Inc.

## License

Released under the Apache 2.0 license.
