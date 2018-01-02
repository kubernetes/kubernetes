# rkt api-service

## Overview

The API service lists and introspects pods and images.
The API service is implemented with [gRPC][grpc].
The API service is designed to run without root privileges, and currently provides a read-only interface.
The API service is optional for running pods, the start/stop/crash of the API service won't affect any pods or images.

## Running the API service

The API service listens for gRPC requests on the address and port specified by the `--listen` option.
The default is to listen on the loopback interface on port number `15441`, equivalent to invoking `rkt api-service --listen=localhost:15441`.
Specify the address `0.0.0.0` to listen on all interfaces.

Typically, the API service will be run via a unit file similar to the one included in the [dist directory][rkt-api].

## Using the API service

The interfaces are defined in the [protobuf here][api_proto].
Here is a small [Go program][client-example] that illustrates how to use the API service.

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--listen` |  `localhost:15441` | An address to listen on | Address to listen for client API requests |

## Global options

See the table with [global options in general commands documentation][global-options].


[api_proto]: https://github.com/coreos/rkt/blob/master/api/v1alpha/api.proto
[client-example]: https://github.com/coreos/rkt/blob/master/api/v1alpha/client_example.go
[global-options]: ../commands.md#global-options
[grpc]: http://www.grpc.io/
[rkt-api]: https://github.com/coreos/rkt/blob/master/dist/init/systemd/rkt-api.service
