# Bookstore Example

This directory contains an OpenAPI description of a simple bookstore API.

Use this example to try the `openapi_swift_generator` plugin, which 
generates Swift code that implements an API client and server for
an OpenAPI description.

Run `make all` to build and install `openapic` and the Swift plugin.
It will generate both client and server code. The API client and
server code will be in the `Sources/Bookstore` package. 

The `Sources/Server` directory contains additional code that completes the server.
To build and run the server, do the following:

    swift build
    .build/debug/Server &

To test the service with the generated client, run `swift build`.
Tests are in the `Tests` directory and use client
code generated in `Bookstore` to verify the service.

