// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

// gRPC Server Interceptor chaining middleware.

package grpc_middleware

import (
	"context"

	"google.golang.org/grpc"
)

// ChainUnaryServer creates a single interceptor out of a chain of many interceptors.
//
// Execution is done in left-to-right order, including passing of context.
// For example ChainUnaryServer(one, two, three) will execute one before two before three, and three
// will see context changes of one and two.
//
// While this can be useful in some scenarios, it is generally advisable to use google.golang.org/grpc.ChainUnaryInterceptor directly.
func ChainUnaryServer(interceptors ...grpc.UnaryServerInterceptor) grpc.UnaryServerInterceptor {
	n := len(interceptors)

	// Dummy interceptor maintained for backward compatibility to avoid returning nil.
	if n == 0 {
		return func(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
			return handler(ctx, req)
		}
	}

	// The degenerate case, just return the single wrapped interceptor directly.
	if n == 1 {
		return interceptors[0]
	}

	// Return a function which satisfies the interceptor interface, and which is
	// a closure over the given list of interceptors to be chained.
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		currHandler := handler
		// Iterate backwards through all interceptors except the first (outermost).
		// Wrap each one in a function which satisfies the handler interface, but
		// is also a closure over the `info` and `handler` parameters. Then pass
		// each pseudo-handler to the next outer interceptor as the handler to be called.
		for i := n - 1; i > 0; i-- {
			// Rebind to loop-local vars so they can be closed over.
			innerHandler, i := currHandler, i
			currHandler = func(currentCtx context.Context, currentReq interface{}) (interface{}, error) {
				return interceptors[i](currentCtx, currentReq, info, innerHandler)
			}
		}
		// Finally return the result of calling the outermost interceptor with the
		// outermost pseudo-handler created above as its handler.
		return interceptors[0](ctx, req, info, currHandler)
	}
}

// ChainStreamServer creates a single interceptor out of a chain of many interceptors.
//
// Execution is done in left-to-right order, including passing of context.
// For example ChainUnaryServer(one, two, three) will execute one before two before three.
// If you want to pass context between interceptors, use WrapServerStream.
//
// While this can be useful in some scenarios, it is generally advisable to use google.golang.org/grpc.ChainStreamInterceptor directly.
func ChainStreamServer(interceptors ...grpc.StreamServerInterceptor) grpc.StreamServerInterceptor {
	n := len(interceptors)

	// Dummy interceptor maintained for backward compatibility to avoid returning nil.
	if n == 0 {
		return func(srv interface{}, stream grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
			return handler(srv, stream)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		currHandler := handler
		for i := n - 1; i > 0; i-- {
			innerHandler, i := currHandler, i
			currHandler = func(currentSrv interface{}, currentStream grpc.ServerStream) error {
				return interceptors[i](currentSrv, currentStream, info, innerHandler)
			}
		}
		return interceptors[0](srv, stream, info, currHandler)
	}
}

// ChainUnaryClient creates a single interceptor out of a chain of many interceptors.
//
// Execution is done in left-to-right order, including passing of context.
// For example ChainUnaryClient(one, two, three) will execute one before two before three.
func ChainUnaryClient(interceptors ...grpc.UnaryClientInterceptor) grpc.UnaryClientInterceptor {
	n := len(interceptors)

	// Dummy interceptor maintained for backward compatibility to avoid returning nil.
	if n == 0 {
		return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
			return invoker(ctx, method, req, reply, cc, opts...)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		currInvoker := invoker
		for i := n - 1; i > 0; i-- {
			innerInvoker, i := currInvoker, i
			currInvoker = func(currentCtx context.Context, currentMethod string, currentReq, currentRepl interface{}, currentConn *grpc.ClientConn, currentOpts ...grpc.CallOption) error {
				return interceptors[i](currentCtx, currentMethod, currentReq, currentRepl, currentConn, innerInvoker, currentOpts...)
			}
		}
		return interceptors[0](ctx, method, req, reply, cc, currInvoker, opts...)
	}
}

// ChainStreamClient creates a single interceptor out of a chain of many interceptors.
//
// Execution is done in left-to-right order, including passing of context.
// For example ChainStreamClient(one, two, three) will execute one before two before three.
func ChainStreamClient(interceptors ...grpc.StreamClientInterceptor) grpc.StreamClientInterceptor {
	n := len(interceptors)

	// Dummy interceptor maintained for backward compatibility to avoid returning nil.
	if n == 0 {
		return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
			return streamer(ctx, desc, cc, method, opts...)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		currStreamer := streamer
		for i := n - 1; i > 0; i-- {
			innerStreamer, i := currStreamer, i
			currStreamer = func(currentCtx context.Context, currentDesc *grpc.StreamDesc, currentConn *grpc.ClientConn, currentMethod string, currentOpts ...grpc.CallOption) (grpc.ClientStream, error) {
				return interceptors[i](currentCtx, currentDesc, currentConn, currentMethod, innerStreamer, currentOpts...)
			}
		}
		return interceptors[0](ctx, desc, cc, method, currStreamer, opts...)
	}
}

// Chain creates a single interceptor out of a chain of many interceptors.
//
// WithUnaryServerChain is a grpc.Server config option that accepts multiple unary interceptors.
// Basically syntactic sugar.
//
// Deprecated: use google.golang.org/grpc.ChainUnaryInterceptor instead.
func WithUnaryServerChain(interceptors ...grpc.UnaryServerInterceptor) grpc.ServerOption {
	return grpc.ChainUnaryInterceptor(interceptors...)
}

// WithStreamServerChain is a grpc.Server config option that accepts multiple stream interceptors.
// Basically syntactic sugar.
//
// Deprecated: use google.golang.org/grpc.ChainStreamInterceptor instead.
func WithStreamServerChain(interceptors ...grpc.StreamServerInterceptor) grpc.ServerOption {
	return grpc.ChainStreamInterceptor(interceptors...)
}
