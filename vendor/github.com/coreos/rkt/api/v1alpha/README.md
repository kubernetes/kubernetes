# WARNING

The API defined here is proposed, experimental, and (for now) subject to change at any time.

If you think you want to use it, or for any other queries, contact <rkt-dev@googlegroups.com> or file an [issue](https://github.com/coreos/rkt/issues/new)

For more information, see:
- #1208
- #1359
- #1468
- [API Service Subcommand](../../Documentation/subcommands/api-service.md)

## Protobuf

The rkt gRPC API uses Protocol Buffers for its services.
In order to rebuild the generated code make sure you have protobuf 3.0.0 installed (https://github.com/google/protobuf)
and execute from the top-level directory:

```
$ make protobuf
```
