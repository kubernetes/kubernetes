<!-- Generated. DO NOT MODIFY. -->
# Migration from v1.38.0 to v1.39.0

The `go.opentelemetry.io/otel/semconv/v1.39.0` package should be a drop-in replacement for `go.opentelemetry.io/otel/semconv/v1.38.0` with the following exceptions.

## Removed

The following declarations have been removed.
Refer to the [OpenTelemetry Semantic Conventions documentation] for deprecation instructions.

If the type is not listed in the documentation as deprecated, it has been removed in this version due to lack of applicability or use.
If you use any of these non-deprecated declarations in your Go application, please [open an issue] describing your use-case.

- `LinuxMemorySlabStateKey`
- `LinuxMemorySlabStateReclaimable`
- `LinuxMemorySlabStateUnreclaimable`
- `PeerService`
- `PeerServiceKey`
- `RPCConnectRPCErrorCodeAborted`
- `RPCConnectRPCErrorCodeAlreadyExists`
- `RPCConnectRPCErrorCodeCancelled`
- `RPCConnectRPCErrorCodeDataLoss`
- `RPCConnectRPCErrorCodeDeadlineExceeded`
- `RPCConnectRPCErrorCodeFailedPrecondition`
- `RPCConnectRPCErrorCodeInternal`
- `RPCConnectRPCErrorCodeInvalidArgument`
- `RPCConnectRPCErrorCodeKey`
- `RPCConnectRPCErrorCodeNotFound`
- `RPCConnectRPCErrorCodeOutOfRange`
- `RPCConnectRPCErrorCodePermissionDenied`
- `RPCConnectRPCErrorCodeResourceExhausted`
- `RPCConnectRPCErrorCodeUnauthenticated`
- `RPCConnectRPCErrorCodeUnavailable`
- `RPCConnectRPCErrorCodeUnimplemented`
- `RPCConnectRPCErrorCodeUnknown`
- `RPCConnectRPCRequestMetadata`
- `RPCConnectRPCResponseMetadata`
- `RPCGRPCRequestMetadata`
- `RPCGRPCResponseMetadata`
- `RPCGRPCStatusCodeAborted`
- `RPCGRPCStatusCodeAlreadyExists`
- `RPCGRPCStatusCodeCancelled`
- `RPCGRPCStatusCodeDataLoss`
- `RPCGRPCStatusCodeDeadlineExceeded`
- `RPCGRPCStatusCodeFailedPrecondition`
- `RPCGRPCStatusCodeInternal`
- `RPCGRPCStatusCodeInvalidArgument`
- `RPCGRPCStatusCodeKey`
- `RPCGRPCStatusCodeNotFound`
- `RPCGRPCStatusCodeOk`
- `RPCGRPCStatusCodeOutOfRange`
- `RPCGRPCStatusCodePermissionDenied`
- `RPCGRPCStatusCodeResourceExhausted`
- `RPCGRPCStatusCodeUnauthenticated`
- `RPCGRPCStatusCodeUnavailable`
- `RPCGRPCStatusCodeUnimplemented`
- `RPCGRPCStatusCodeUnknown`
- `RPCJSONRPCErrorCode`
- `RPCJSONRPCErrorCodeKey`
- `RPCJSONRPCErrorMessage`
- `RPCJSONRPCErrorMessageKey`
- `RPCJSONRPCRequestID`
- `RPCJSONRPCRequestIDKey`
- `RPCJSONRPCVersion`
- `RPCJSONRPCVersionKey`
- `RPCService`
- `RPCServiceKey`
- `RPCSystemApacheDubbo`
- `RPCSystemConnectRPC`
- `RPCSystemDotnetWcf`
- `RPCSystemGRPC`
- `RPCSystemJSONRPC`
- `RPCSystemJavaRmi`
- `RPCSystemKey`
- `RPCSystemOncRPC`

[OpenTelemetry Semantic Conventions documentation]: https://github.com/open-telemetry/semantic-conventions
[open an issue]: https://github.com/open-telemetry/opentelemetry-go/issues/new?template=Blank+issue
