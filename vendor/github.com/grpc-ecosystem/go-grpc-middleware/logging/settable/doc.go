//
/*
grpc_logsettable contains a thread-safe wrapper around grpc-logging
infrastructure.

The go-grpc assumes that logger can be only configured once as the `SetLoggerV2`
method is:
```Not mutex-protected, should be called before any gRPC functions.```

This package allows to supply parent logger once ("before any grpc"), but
later change underlying implementation in thread-safe way when needed.

It's in particular useful for testing, where each testcase might need its own
logger.
*/
package grpc_logsettable
