# OpenAPI Go Generator Plugin

This directory contains an `openapic` plugin that can be used to generate a Go client library and scaffolding for a Go server for an API with an OpenAPI description.

The plugin can be invoked like this:

	openapic bookstore.json --go_generator_out=package=bookstore:bookstore

Where `bookstore` is the name of a directory where the generated code will be written and `package=bookstore` indicates that "bookstore" should also be the package name used for generated code. 

By default, both client and server code will be generated. If the `openapi_go_generator` binary is also linked from the names `openapi_go_client` and `openapi_go_server`, then only client or only server code can be generated as follows:

	openapic bookstore.json --go_client_out=package=bookstore:bookstore

	openapic bookstore.json --go_server_out=package=bookstore:bookstore

For example usage, see the [examples/bookstore](examples/bookstore) directory.