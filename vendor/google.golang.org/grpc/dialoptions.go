/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"context"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal"
	internalbackoff "google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/stats"
)

// dialOptions configure a Dial call. dialOptions are set by the DialOption
// values passed to Dial.
type dialOptions struct {
	unaryInt  UnaryClientInterceptor
	streamInt StreamClientInterceptor

	chainUnaryInts  []UnaryClientInterceptor
	chainStreamInts []StreamClientInterceptor

	cp          Compressor
	dc          Decompressor
	bs          internalbackoff.Strategy
	block       bool
	insecure    bool
	timeout     time.Duration
	scChan      <-chan ServiceConfig
	authority   string
	copts       transport.ConnectOptions
	callOptions []CallOption
	// This is used by v1 balancer dial option WithBalancer to support v1
	// balancer, and also by WithBalancerName dial option.
	balancerBuilder             balancer.Builder
	channelzParentID            int64
	disableServiceConfig        bool
	disableRetry                bool
	disableHealthCheck          bool
	healthCheckFunc             internal.HealthChecker
	minConnectTimeout           func() time.Duration
	defaultServiceConfig        *ServiceConfig // defaultServiceConfig is parsed from defaultServiceConfigRawJSON.
	defaultServiceConfigRawJSON *string
	// This is used by ccResolverWrapper to backoff between successive calls to
	// resolver.ResolveNow(). The user will have no need to configure this, but
	// we need to be able to configure this in tests.
	resolveNowBackoff func(int) time.Duration
	resolvers         []resolver.Builder
}

// DialOption configures how we set up the connection.
type DialOption interface {
	apply(*dialOptions)
}

// EmptyDialOption does not alter the dial configuration. It can be embedded in
// another structure to build custom dial options.
//
// This API is EXPERIMENTAL.
type EmptyDialOption struct{}

func (EmptyDialOption) apply(*dialOptions) {}

// funcDialOption wraps a function that modifies dialOptions into an
// implementation of the DialOption interface.
type funcDialOption struct {
	f func(*dialOptions)
}

func (fdo *funcDialOption) apply(do *dialOptions) {
	fdo.f(do)
}

func newFuncDialOption(f func(*dialOptions)) *funcDialOption {
	return &funcDialOption{
		f: f,
	}
}

// WithWriteBufferSize determines how much data can be batched before doing a
// write on the wire. The corresponding memory allocation for this buffer will
// be twice the size to keep syscalls low. The default value for this buffer is
// 32KB.
//
// Zero will disable the write buffer such that each write will be on underlying
// connection. Note: A Send call may not directly translate to a write.
func WithWriteBufferSize(s int) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.WriteBufferSize = s
	})
}

// WithReadBufferSize lets you set the size of read buffer, this determines how
// much data can be read at most for each read syscall.
//
// The default value for this buffer is 32KB. Zero will disable read buffer for
// a connection so data framer can access the underlying conn directly.
func WithReadBufferSize(s int) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.ReadBufferSize = s
	})
}

// WithInitialWindowSize returns a DialOption which sets the value for initial
// window size on a stream. The lower bound for window size is 64K and any value
// smaller than that will be ignored.
func WithInitialWindowSize(s int32) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.InitialWindowSize = s
	})
}

// WithInitialConnWindowSize returns a DialOption which sets the value for
// initial window size on a connection. The lower bound for window size is 64K
// and any value smaller than that will be ignored.
func WithInitialConnWindowSize(s int32) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.InitialConnWindowSize = s
	})
}

// WithMaxMsgSize returns a DialOption which sets the maximum message size the
// client can receive.
//
// Deprecated: use WithDefaultCallOptions(MaxCallRecvMsgSize(s)) instead.  Will
// be supported throughout 1.x.
func WithMaxMsgSize(s int) DialOption {
	return WithDefaultCallOptions(MaxCallRecvMsgSize(s))
}

// WithDefaultCallOptions returns a DialOption which sets the default
// CallOptions for calls over the connection.
func WithDefaultCallOptions(cos ...CallOption) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.callOptions = append(o.callOptions, cos...)
	})
}

// WithCodec returns a DialOption which sets a codec for message marshaling and
// unmarshaling.
//
// Deprecated: use WithDefaultCallOptions(ForceCodec(_)) instead.  Will be
// supported throughout 1.x.
func WithCodec(c Codec) DialOption {
	return WithDefaultCallOptions(CallCustomCodec(c))
}

// WithCompressor returns a DialOption which sets a Compressor to use for
// message compression. It has lower priority than the compressor set by the
// UseCompressor CallOption.
//
// Deprecated: use UseCompressor instead.  Will be supported throughout 1.x.
func WithCompressor(cp Compressor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.cp = cp
	})
}

// WithDecompressor returns a DialOption which sets a Decompressor to use for
// incoming message decompression.  If incoming response messages are encoded
// using the decompressor's Type(), it will be used.  Otherwise, the message
// encoding will be used to look up the compressor registered via
// encoding.RegisterCompressor, which will then be used to decompress the
// message.  If no compressor is registered for the encoding, an Unimplemented
// status error will be returned.
//
// Deprecated: use encoding.RegisterCompressor instead.  Will be supported
// throughout 1.x.
func WithDecompressor(dc Decompressor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.dc = dc
	})
}

// WithBalancer returns a DialOption which sets a load balancer with the v1 API.
// Name resolver will be ignored if this DialOption is specified.
//
// Deprecated: use the new balancer APIs in balancer package and
// WithBalancerName.  Will be removed in a future 1.x release.
func WithBalancer(b Balancer) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.balancerBuilder = &balancerWrapperBuilder{
			b: b,
		}
	})
}

// WithBalancerName sets the balancer that the ClientConn will be initialized
// with. Balancer registered with balancerName will be used. This function
// panics if no balancer was registered by balancerName.
//
// The balancer cannot be overridden by balancer option specified by service
// config.
//
// Deprecated: use WithDefaultServiceConfig and WithDisableServiceConfig
// instead.  Will be removed in a future 1.x release.
func WithBalancerName(balancerName string) DialOption {
	builder := balancer.Get(balancerName)
	if builder == nil {
		panic(fmt.Sprintf("grpc.WithBalancerName: no balancer is registered for name %v", balancerName))
	}
	return newFuncDialOption(func(o *dialOptions) {
		o.balancerBuilder = builder
	})
}

// WithServiceConfig returns a DialOption which has a channel to read the
// service configuration.
//
// Deprecated: service config should be received through name resolver or via
// WithDefaultServiceConfig, as specified at
// https://github.com/grpc/grpc/blob/master/doc/service_config.md.  Will be
// removed in a future 1.x release.
func WithServiceConfig(c <-chan ServiceConfig) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.scChan = c
	})
}

// WithConnectParams configures the dialer to use the provided ConnectParams.
//
// The backoff configuration specified as part of the ConnectParams overrides
// all defaults specified in
// https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md. Consider
// using the backoff.DefaultConfig as a base, in cases where you want to
// override only a subset of the backoff configuration.
//
// This API is EXPERIMENTAL.
func WithConnectParams(p ConnectParams) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.bs = internalbackoff.Exponential{Config: p.Backoff}
		o.minConnectTimeout = func() time.Duration {
			return p.MinConnectTimeout
		}
	})
}

// WithBackoffMaxDelay configures the dialer to use the provided maximum delay
// when backing off after failed connection attempts.
//
// Deprecated: use WithConnectParams instead. Will be supported throughout 1.x.
func WithBackoffMaxDelay(md time.Duration) DialOption {
	return WithBackoffConfig(BackoffConfig{MaxDelay: md})
}

// WithBackoffConfig configures the dialer to use the provided backoff
// parameters after connection failures.
//
// Deprecated: use WithConnectParams instead. Will be supported throughout 1.x.
func WithBackoffConfig(b BackoffConfig) DialOption {
	bc := backoff.DefaultConfig
	bc.MaxDelay = b.MaxDelay
	return withBackoff(internalbackoff.Exponential{Config: bc})
}

// withBackoff sets the backoff strategy used for connectRetryNum after a failed
// connection attempt.
//
// This can be exported if arbitrary backoff strategies are allowed by gRPC.
func withBackoff(bs internalbackoff.Strategy) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.bs = bs
	})
}

// WithBlock returns a DialOption which makes caller of Dial blocks until the
// underlying connection is up. Without this, Dial returns immediately and
// connecting the server happens in background.
func WithBlock() DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.block = true
	})
}

// WithInsecure returns a DialOption which disables transport security for this
// ClientConn. Note that transport security is required unless WithInsecure is
// set.
func WithInsecure() DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.insecure = true
	})
}

// WithTransportCredentials returns a DialOption which configures a connection
// level security credentials (e.g., TLS/SSL). This should not be used together
// with WithCredentialsBundle.
func WithTransportCredentials(creds credentials.TransportCredentials) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.TransportCredentials = creds
	})
}

// WithPerRPCCredentials returns a DialOption which sets credentials and places
// auth state on each outbound RPC.
func WithPerRPCCredentials(creds credentials.PerRPCCredentials) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.PerRPCCredentials = append(o.copts.PerRPCCredentials, creds)
	})
}

// WithCredentialsBundle returns a DialOption to set a credentials bundle for
// the ClientConn.WithCreds. This should not be used together with
// WithTransportCredentials.
//
// This API is experimental.
func WithCredentialsBundle(b credentials.Bundle) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.CredsBundle = b
	})
}

// WithTimeout returns a DialOption that configures a timeout for dialing a
// ClientConn initially. This is valid if and only if WithBlock() is present.
//
// Deprecated: use DialContext instead of Dial and context.WithTimeout
// instead.  Will be supported throughout 1.x.
func WithTimeout(d time.Duration) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.timeout = d
	})
}

// WithContextDialer returns a DialOption that sets a dialer to create
// connections. If FailOnNonTempDialError() is set to true, and an error is
// returned by f, gRPC checks the error's Temporary() method to decide if it
// should try to reconnect to the network address.
func WithContextDialer(f func(context.Context, string) (net.Conn, error)) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.Dialer = f
	})
}

func init() {
	internal.WithHealthCheckFunc = withHealthCheckFunc
}

// WithDialer returns a DialOption that specifies a function to use for dialing
// network addresses. If FailOnNonTempDialError() is set to true, and an error
// is returned by f, gRPC checks the error's Temporary() method to decide if it
// should try to reconnect to the network address.
//
// Deprecated: use WithContextDialer instead.  Will be supported throughout
// 1.x.
func WithDialer(f func(string, time.Duration) (net.Conn, error)) DialOption {
	return WithContextDialer(
		func(ctx context.Context, addr string) (net.Conn, error) {
			if deadline, ok := ctx.Deadline(); ok {
				return f(addr, time.Until(deadline))
			}
			return f(addr, 0)
		})
}

// WithStatsHandler returns a DialOption that specifies the stats handler for
// all the RPCs and underlying network connections in this ClientConn.
func WithStatsHandler(h stats.Handler) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.StatsHandler = h
	})
}

// FailOnNonTempDialError returns a DialOption that specifies if gRPC fails on
// non-temporary dial errors. If f is true, and dialer returns a non-temporary
// error, gRPC will fail the connection to the network address and won't try to
// reconnect. The default value of FailOnNonTempDialError is false.
//
// FailOnNonTempDialError only affects the initial dial, and does not do
// anything useful unless you are also using WithBlock().
//
// This is an EXPERIMENTAL API.
func FailOnNonTempDialError(f bool) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.FailOnNonTempDialError = f
	})
}

// WithUserAgent returns a DialOption that specifies a user agent string for all
// the RPCs.
func WithUserAgent(s string) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.UserAgent = s
	})
}

// WithKeepaliveParams returns a DialOption that specifies keepalive parameters
// for the client transport.
func WithKeepaliveParams(kp keepalive.ClientParameters) DialOption {
	if kp.Time < internal.KeepaliveMinPingTime {
		grpclog.Warningf("Adjusting keepalive ping interval to minimum period of %v", internal.KeepaliveMinPingTime)
		kp.Time = internal.KeepaliveMinPingTime
	}
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.KeepaliveParams = kp
	})
}

// WithUnaryInterceptor returns a DialOption that specifies the interceptor for
// unary RPCs.
func WithUnaryInterceptor(f UnaryClientInterceptor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.unaryInt = f
	})
}

// WithChainUnaryInterceptor returns a DialOption that specifies the chained
// interceptor for unary RPCs. The first interceptor will be the outer most,
// while the last interceptor will be the inner most wrapper around the real call.
// All interceptors added by this method will be chained, and the interceptor
// defined by WithUnaryInterceptor will always be prepended to the chain.
func WithChainUnaryInterceptor(interceptors ...UnaryClientInterceptor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.chainUnaryInts = append(o.chainUnaryInts, interceptors...)
	})
}

// WithStreamInterceptor returns a DialOption that specifies the interceptor for
// streaming RPCs.
func WithStreamInterceptor(f StreamClientInterceptor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.streamInt = f
	})
}

// WithChainStreamInterceptor returns a DialOption that specifies the chained
// interceptor for unary RPCs. The first interceptor will be the outer most,
// while the last interceptor will be the inner most wrapper around the real call.
// All interceptors added by this method will be chained, and the interceptor
// defined by WithStreamInterceptor will always be prepended to the chain.
func WithChainStreamInterceptor(interceptors ...StreamClientInterceptor) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.chainStreamInts = append(o.chainStreamInts, interceptors...)
	})
}

// WithAuthority returns a DialOption that specifies the value to be used as the
// :authority pseudo-header. This value only works with WithInsecure and has no
// effect if TransportCredentials are present.
func WithAuthority(a string) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.authority = a
	})
}

// WithChannelzParentID returns a DialOption that specifies the channelz ID of
// current ClientConn's parent. This function is used in nested channel creation
// (e.g. grpclb dial).
//
// This API is EXPERIMENTAL.
func WithChannelzParentID(id int64) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.channelzParentID = id
	})
}

// WithDisableServiceConfig returns a DialOption that causes gRPC to ignore any
// service config provided by the resolver and provides a hint to the resolver
// to not fetch service configs.
//
// Note that this dial option only disables service config from resolver. If
// default service config is provided, gRPC will use the default service config.
func WithDisableServiceConfig() DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.disableServiceConfig = true
	})
}

// WithDefaultServiceConfig returns a DialOption that configures the default
// service config, which will be used in cases where:
//
// 1. WithDisableServiceConfig is also used.
// 2. Resolver does not return a service config or if the resolver returns an
//    invalid service config.
//
// This API is EXPERIMENTAL.
func WithDefaultServiceConfig(s string) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.defaultServiceConfigRawJSON = &s
	})
}

// WithDisableRetry returns a DialOption that disables retries, even if the
// service config enables them.  This does not impact transparent retries, which
// will happen automatically if no data is written to the wire or if the RPC is
// unprocessed by the remote server.
//
// Retry support is currently disabled by default, but will be enabled by
// default in the future.  Until then, it may be enabled by setting the
// environment variable "GRPC_GO_RETRY" to "on".
//
// This API is EXPERIMENTAL.
func WithDisableRetry() DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.disableRetry = true
	})
}

// WithMaxHeaderListSize returns a DialOption that specifies the maximum
// (uncompressed) size of header list that the client is prepared to accept.
func WithMaxHeaderListSize(s uint32) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.copts.MaxHeaderListSize = &s
	})
}

// WithDisableHealthCheck disables the LB channel health checking for all
// SubConns of this ClientConn.
//
// This API is EXPERIMENTAL.
func WithDisableHealthCheck() DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.disableHealthCheck = true
	})
}

// withHealthCheckFunc replaces the default health check function with the
// provided one. It makes tests easier to change the health check function.
//
// For testing purpose only.
func withHealthCheckFunc(f internal.HealthChecker) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.healthCheckFunc = f
	})
}

func defaultDialOptions() dialOptions {
	return dialOptions{
		disableRetry:    !envconfig.Retry,
		healthCheckFunc: internal.HealthCheckFunc,
		copts: transport.ConnectOptions{
			WriteBufferSize: defaultWriteBufSize,
			ReadBufferSize:  defaultReadBufSize,
		},
		resolveNowBackoff: internalbackoff.DefaultExponential.Backoff,
	}
}

// withGetMinConnectDeadline specifies the function that clientconn uses to
// get minConnectDeadline. This can be used to make connection attempts happen
// faster/slower.
//
// For testing purpose only.
func withMinConnectDeadline(f func() time.Duration) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.minConnectTimeout = f
	})
}

// withResolveNowBackoff specifies the function that clientconn uses to backoff
// between successive calls to resolver.ResolveNow().
//
// For testing purpose only.
func withResolveNowBackoff(f func(int) time.Duration) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.resolveNowBackoff = f
	})
}

// WithResolvers allows a list of resolver implementations to be registered
// locally with the ClientConn without needing to be globally registered via
// resolver.Register.  They will be matched against the scheme used for the
// current Dial only, and will take precedence over the global registry.
//
// This API is EXPERIMENTAL.
func WithResolvers(rs ...resolver.Builder) DialOption {
	return newFuncDialOption(func(o *dialOptions) {
		o.resolvers = append(o.resolvers, rs...)
	})
}
