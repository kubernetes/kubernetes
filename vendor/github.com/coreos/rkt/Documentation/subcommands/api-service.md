# rkt api-service

## Overview

The API service lists and introspects pods and images.
The API service is implemented with [gRPC](http://www.grpc.io/).
The API service is designed to run without root privileges, and currently provides a read-only interface.
The API service is optional for running pods, the start/stop/crash of the API service won't affect any pods or images.

## Running the API service

The API service listens for gRPC requests on the address and port specified by the `--listen` option.
The default is to listen on the loopback interface on port number `15441`, equivalent to invoking `rkt api-service --listen=localhost:15441`.
Specify the address `0.0.0.0` to listen on all interfaces.

## Using the API service

The interfaces are defined in the [protobuf here](../../api/v1alpha/api.proto).
Here is a small [Go program](../../api/v1alpha/client_example.go) that illustrates how to use the API service.

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--listen` |  `localhost:15441` | An address to listen on | Address to listen for client API requests |

## Global options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--debug` |  `false` | `true` or `false` | Prints out more debug information to `stderr` |
| `--dir` | `/var/lib/rkt` | A directory path | Path to the `rkt` data directory |
| `--insecure-options` |  none | **none**: All security features are enabled<br/>**http**: Allow HTTP connections. Be warned that this will send any credentials as clear text.<br/>**image**: Disables verifying image signatures<br/>**tls**: Accept any certificate from the server and any host name in that certificate<br/>**ondisk**: Disables verifying the integrity of the on-disk, rendered image before running. This significantly speeds up start time.<br/>**all**: Disables all security checks | Comma-separated list of security features to disable |
| `--local-config` |  `/etc/rkt` | A directory path | Path to the local configuration directory |
| `--system-config` |  `/usr/lib/rkt` | A directory path | Path to the system configuration directory |
| `--trust-keys-from-https` |  `false` | `true` or `false` | Automatically trust gpg keys fetched from https |
| `--user-config` |  `` | A directory path | Path to the user configuration directory |