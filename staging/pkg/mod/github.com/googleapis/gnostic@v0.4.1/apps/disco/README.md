# disco

This directory contains a tool for working with Google's Discovery API and Discovery Format,
including the ability to convert Discovery Format descriptions to OpenAPI.

Installation:

        go get github.com/googleapis/gnostic
        go install github.com/googleapis/gnostic/apps/disco
  
  
Usage:

        disco help

Prints a list of commands and options.

        disco list [--raw]
        
Calls the Google Discovery API and lists available APIs. 
The `--raw` option prints the raw results of the Discovery List APIs call.

        disco get [<api>] [<version>] [--raw] [--openapi2] [--openapi3] [--features] [--schemas] [--all]
        
Gets the specified API and version from the Google Discovery API.
`<version>` can be omitted if it is unique.
The `--raw` option saves the raw Discovery Format description.
The `--openapi2` option rewrites the API description in OpenAPI v2.
The `--openapi3` option rewrites the API description in OpenAPI v3.
The `--features` option displays the contents of the `features` sections of discovery documents.
The `--schemas` option displays information about the schemas defined for the API.
The `--all` option runs the other associated operations for all of the APIs available from the Discovery Service. 
When `--all` is specified, `<api>` and `<version>` should be omitted.

        disco <file> [--openapi2] [--openapi3] [--features] [--schemas]

Applies the specified operations to a local file. See the `get` command for details.

