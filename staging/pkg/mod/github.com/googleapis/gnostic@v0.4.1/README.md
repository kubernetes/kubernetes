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

The following instructions are for installing gnostic using
[Go modules](https://blog.golang.org/using-go-modules),
supported by Go 1.11 and later.

1. Get this package by downloading it with `git clone`.

        git clone https://github.com/googleapis/gnostic
        cd gnostic
  
2. [Optional] Build and run the gnostic compiler generator. 
This uses JSON schemas to generate Protocol Buffer language files
that describes supported API specification formats and Go-language
files of code that will read JSON or YAML API descriptions into
the generated protocol buffer models. 

Pre-generated versions of these files are checked into the directories
named OpenAPIv2, OpenAPIv3, and discovery.

        go install ./generate-gnostic
        generate-gnostic --v2
        generate-gnostic --v3
        generate-gnostic --discovery

3. Generate Protocol Buffer support code. 
This step requires a local installation of protoc, the Protocol Buffer Compiler,
and the Go protoc plugin.
You can get protoc [here](https://github.com/google/protobuf).
You can install the plugin with this command:

        go get github.com/golang/protobuf/protoc-gen-go

Then use the following to compile the gnostic Protocol Buffer models:

        ./COMPILE-PROTOS.sh

4. Build **gnostic**. 

        go install .

5. Run **gnostic**. This sample invocation creates a file in the current directory named "petstore.pb" that contains a binary
Protocol Buffer description of a sample API.

        gnostic --pb-out=. examples/v2.0/json/petstore.json

6. You can also compile files that you specify with a URL. Here's another way to compile the previous 
example. This time we're creating "petstore.text", which contains a textual representation of the
Protocol Buffer description. This is mainly for use in testing and debugging.

        gnostic --text-out=petstore.text https://raw.githubusercontent.com/googleapis/gnostic/master/examples/v2.0/json/petstore.json

7. For a sample application, see apps/report.

        go install ./apps/report
        report petstore.pb

8. **gnostic** supports plugins. Some are already implemented in the `plugins` directory.
Others, like [gnostic-go-generator](https://github.com/googleapis/gnostic-go-generator),
are separated into their own repositories.

## Copyright

Copyright 2017, Google Inc.

## License

Released under the Apache 2.0 license.
