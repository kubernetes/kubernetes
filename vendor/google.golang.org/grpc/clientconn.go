/*
 *
 * Copyright 2014 gRPC authors.
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
	"errors"
	"fmt"
	"math"
	"net"
	"reflect"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/trace"
	"google.golang.org/grpc/balancer"
	_ "google.golang.org/grpc/balancer/roundrobin" // To register roundrobin.
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/resolver"
	_ "google.golang.org/grpc/resolver/dns"         // To register dns resolver.
	_ "google.golang.org/grpc/resolver/passthrough" // To register passthrough resolver.
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

const (
	// minimum time to give a connection to complete
	minConnectTimeout = 20 * time.Second
	// must match grpclbName in grpclb/grpclb.go
	grpclbName = "grpclb"
)

var (
	// ErrClientConnClosing indicates that the operation is illegal because
	// the ClientConn is closing.
	//
	// Deprecated: this error should not be relied upon by users; use the status
	// code of Canceled instead.
	ErrClientConnClosing = status.Error(codes.Canceled, "grpc: the client connection is closing")
	// errConnDrain indicates that the connection starts to be drained and does not accept any new RPCs.
	errConnDrain = errors.New("grpc: the connection is drained")
	// errConnClosing indicates that the connection is closing.
	errConnClosing = errors.New("grpc: the connection is closing")
	// errConnUnavailable indicates that the connection is unavailable.
	errConnUnavailable = errors.New("grpc: the connection is unavailable")
	// errBalancerClosed indicates that the balancer is closed.
	errBalancerClosed = errors.New("grpc: balancer is closed")
	// We use an accessor so that minConnectTimeout can be
	// atomically read and updated while testing.
	getMinConnectTimeout = func() time.Duration {
		return minConnectTimeout
	}
)

// The following errors are returned from Dial and DialContext
var (
	// errNoTransportSecurity indicates that there is no transport security
	// being set for ClientConn. Users should either set one or explicitly
	// call WithInsecure DialOption to disable security.
	errNoTransportSecurity = errors.New("grpc: no transport security set (use grpc.WithInsecure() explicitly or set credentials)")
	// errTransportCredentialsMissing indicates that users want to transmit security
	// information (e.g., oauth2 token) which requires secure connection on an insecure
	// connection.
	errTransportCredentialsMissing = errors.New("grpc: the credentials require transport level security (use grpc.WithTransportCredentials() to set)")
	// errCredentialsConflict indicates that grpc.WithTransportCredentials()
	// and grpc.WithInsecure() are both called for a connection.
	errCredentialsConflict = errors.New("grpc: transport credentials are set for an insecure connection (grpc.WithTransportCredentials() and grpc.WithInsecure() are both called)")
	// errNetworkIO indicates that the connection is down due to some network I/O error.
	errNetworkIO = errors.New("grpc: failed with network I/O error")
)

// dialOptions configure a Dial call. dialOptions are set by the DialOption
// values passed to Dial.
type dialOptions struct {
	unaryInt    UnaryClientInterceptor
	streamInt   StreamClientInterceptor
	cp          Compressor
	dc          Decompressor
	bs          backoff.Strategy
	block       bool
	insecure    bool
	timeout     time.Duration
	scChan      <-chan ServiceConfig
	copts       transport.ConnectOptions
	callOptions []CallOption
	// This is used by v1 balancer dial option WithBalancer to support v1
	// balancer, and also by WithBalancerName dial option.
	balancerBuilder balancer.Builder
	// This is to support grpclb.
	resolverBuilder      resolver.Builder
	waitForHandshake     bool
	channelzParentID     int64
	disableServiceConfig bool
}

const (
	defaultClientMaxReceiveMessageSize = 1024 * 1024 * 4
	defaultClientMaxSendMessageSize    = math.MaxInt32
)

// RegisterChannelz turns on channelz service.
// This is an EXPERIMENTAL API.
func RegisterChannelz() {
	channelz.TurnOn()
}

// DialOption configures how we set up the connection.
type DialOption func(*dialOptions)

// WithWaitForHandshake blocks until the initial settings frame is received from the
// server before assigning RPCs to the connection.
// Experimental API.
func WithWaitForHandshake() DialOption {
	return func(o *dialOptions) {
		o.waitForHandshake = true
	}
}

// WithWriteBufferSize lets you set the size of write buffer, this determines how much data can be batched
// before doing a write on the wire.
func WithWriteBufferSize(s int) DialOption {
	return func(o *dialOptions) {
		o.copts.WriteBufferSize = s
	}
}

// WithReadBufferSize lets you set the size of read buffer, this determines how much data can be read at most
// for each read syscall.
func WithReadBufferSize(s int) DialOption {
	return func(o *dialOptions) {
		o.copts.ReadBufferSize = s
	}
}

// WithInitialWindowSize returns a DialOption which sets the value for initial window size on a stream.
// The lower bound for window size is 64K and any value smaller than that will be ignored.
func WithInitialWindowSize(s int32) DialOption {
	return func(o *dialOptions) {
		o.copts.InitialWindowSize = s
	}
}

// WithInitialConnWindowSize returns a DialOption which sets the value for initial window size on a connection.
// The lower bound for window size is 64K and any value smaller than that will be ignored.
func WithInitialConnWindowSize(s int32) DialOption {
	return func(o *dialOptions) {
		o.copts.InitialConnWindowSize = s
	}
}

// WithMaxMsgSize returns a DialOption which sets the maximum message size the client can receive.
//
// Deprecated: use WithDefaultCallOptions(MaxCallRecvMsgSize(s)) instead.
func WithMaxMsgSize(s int) DialOption {
	return WithDefaultCallOptions(MaxCallRecvMsgSize(s))
}

// WithDefaultCallOptions returns a DialOption which sets the default CallOptions for calls over the connection.
func WithDefaultCallOptions(cos ...CallOption) DialOption {
	return func(o *dialOptions) {
		o.callOptions = append(o.callOptions, cos...)
	}
}

// WithCodec returns a DialOption which sets a codec for message marshaling and unmarshaling.
//
// Deprecated: use WithDefaultCallOptions(CallCustomCodec(c)) instead.
func WithCodec(c Codec) DialOption {
	return WithDefaultCallOptions(CallCustomCodec(c))
}

// WithCompressor returns a DialOption which sets a Compressor to use for
// message compression. It has lower priority than the compressor set by
// the UseCompressor CallOption.
//
// Deprecated: use UseCompressor instead.
func WithCompressor(cp Compressor) DialOption {
	return func(o *dialOptions) {
		o.cp = cp
	}
}

// WithDecompressor returns a DialOption which sets a Decompressor to use for
// incoming message decompression.  If incoming response messages are encoded
// using the decompressor's Type(), it will be used.  Otherwise, the message
// encoding will be used to look up the compressor registered via
// encoding.RegisterCompressor, which will then be used to decompress the
// message.  If no compressor is registered for the encoding, an Unimplemented
// status error will be returned.
//
// Deprecated: use encoding.RegisterCompressor instead.
func WithDecompressor(dc Decompressor) DialOption {
	return func(o *dialOptions) {
		o.dc = dc
	}
}

// WithBalancer returns a DialOption which sets a load balancer with the v1 API.
// Name resolver will be ignored if this DialOption is specified.
//
// Deprecated: use the new balancer APIs in balancer package and WithBalancerName.
func WithBalancer(b Balancer) DialOption {
	return func(o *dialOptions) {
		o.balancerBuilder = &balancerWrapperBuilder{
			b: b,
		}
	}
}

// WithBalancerName sets the balancer that the ClientConn will be initialized
// with. Balancer registered with balancerName will be used. This function
// panics if no balancer was registered by balancerName.
//
// The balancer cannot be overridden by balancer option specified by service
// config.
//
// This is an EXPERIMENTAL API.
func WithBalancerName(balancerName string) DialOption {
	builder := balancer.Get(balancerName)
	if builder == nil {
		panic(fmt.Sprintf("grpc.WithBalancerName: no balancer is registered for name %v", balancerName))
	}
	return func(o *dialOptions) {
		o.balancerBuilder = builder
	}
}

// withResolverBuilder is only for grpclb.
func withResolverBuilder(b resolver.Builder) DialOption {
	return func(o *dialOptions) {
		o.resolverBuilder = b
	}
}

// WithServiceConfig returns a DialOption which has a channel to read the service configuration.
//
// Deprecated: service config should be received through name resolver, as specified here.
// https://github.com/grpc/grpc/blob/master/doc/service_config.md
func WithServiceConfig(c <-chan ServiceConfig) DialOption {
	return func(o *dialOptions) {
		o.scChan = c
	}
}

// WithBackoffMaxDelay configures the dialer to use the provided maximum delay
// when backing off after failed connection attempts.
func WithBackoffMaxDelay(md time.Duration) DialOption {
	return WithBackoffConfig(BackoffConfig{MaxDelay: md})
}

// WithBackoffConfig configures the dialer to use the provided backoff
// parameters after connection failures.
//
// Use WithBackoffMaxDelay until more parameters on BackoffConfig are opened up
// for use.
func WithBackoffConfig(b BackoffConfig) DialOption {

	return withBackoff(backoff.Exponential{
		MaxDelay: b.MaxDelay,
	})
}

// withBackoff sets the backoff strategy used for connectRetryNum after a
// failed connection attempt.
//
// This can be exported if arbitrary backoff strategies are allowed by gRPC.
func withBackoff(bs backoff.Strategy) DialOption {
	return func(o *dialOptions) {
		o.bs = bs
	}
}

// WithBlock returns a DialOption which makes caller of Dial blocks until the underlying
// connection is up. Without this, Dial returns immediately and connecting the server
// happens in background.
func WithBlock() DialOption {
	return func(o *dialOptions) {
		o.block = true
	}
}

// WithInsecure returns a DialOption which disables transport security for this ClientConn.
// Note that transport security is required unless WithInsecure is set.
func WithInsecure() DialOption {
	return func(o *dialOptions) {
		o.insecure = true
	}
}

// WithTransportCredentials returns a DialOption which configures a
// connection level security credentials (e.g., TLS/SSL).
func WithTransportCredentials(creds credentials.TransportCredentials) DialOption {
	return func(o *dialOptions) {
		o.copts.TransportCredentials = creds
	}
}

// WithPerRPCCredentials returns a DialOption which sets
// credentials and places auth state on each outbound RPC.
func WithPerRPCCredentials(creds credentials.PerRPCCredentials) DialOption {
	return func(o *dialOptions) {
		o.copts.PerRPCCredentials = append(o.copts.PerRPCCredentials, creds)
	}
}

// WithTimeout returns a DialOption that configures a timeout for dialing a ClientConn
// initially. This is valid if and only if WithBlock() is present.
//
// Deprecated: use DialContext and context.WithTimeout instead.
func WithTimeout(d time.Duration) DialOption {
	return func(o *dialOptions) {
		o.timeout = d
	}
}

func withContextDialer(f func(context.Context, string) (net.Conn, error)) DialOption {
	return func(o *dialOptions) {
		o.copts.Dialer = f
	}
}

func init() {
	internal.WithContextDialer = withContextDialer
	internal.WithResolverBuilder = withResolverBuilder
}

// WithDialer returns a DialOption that specifies a function to use for dialing network addresses.
// If FailOnNonTempDialError() is set to true, and an error is returned by f, gRPC checks the error's
// Temporary() method to decide if it should try to reconnect to the network address.
func WithDialer(f func(string, time.Duration) (net.Conn, error)) DialOption {
	return withContextDialer(
		func(ctx context.Context, addr string) (net.Conn, error) {
			if deadline, ok := ctx.Deadline(); ok {
				return f(addr, deadline.Sub(time.Now()))
			}
			return f(addr, 0)
		})
}

// WithStatsHandler returns a DialOption that specifies the stats handler
// for all the RPCs and underlying network connections in this ClientConn.
func WithStatsHandler(h stats.Handler) DialOption {
	return func(o *dialOptions) {
		o.copts.StatsHandler = h
	}
}

// FailOnNonTempDialError returns a DialOption that specifies if gRPC fails on non-temporary dial errors.
// If f is true, and dialer returns a non-temporary error, gRPC will fail the connection to the network
// address and won't try to reconnect.
// The default value of FailOnNonTempDialError is false.
// This is an EXPERIMENTAL API.
func FailOnNonTempDialError(f bool) DialOption {
	return func(o *dialOptions) {
		o.copts.FailOnNonTempDialError = f
	}
}

// WithUserAgent returns a DialOption that specifies a user agent string for all the RPCs.
func WithUserAgent(s string) DialOption {
	return func(o *dialOptions) {
		o.copts.UserAgent = s
	}
}

// WithKeepaliveParams returns a DialOption that specifies keepalive parameters for the client transport.
func WithKeepaliveParams(kp keepalive.ClientParameters) DialOption {
	return func(o *dialOptions) {
		o.copts.KeepaliveParams = kp
	}
}

// WithUnaryInterceptor returns a DialOption that specifies the interceptor for unary RPCs.
func WithUnaryInterceptor(f UnaryClientInterceptor) DialOption {
	return func(o *dialOptions) {
		o.unaryInt = f
	}
}

// WithStreamInterceptor returns a DialOption that specifies the interceptor for streaming RPCs.
func WithStreamInterceptor(f StreamClientInterceptor) DialOption {
	return func(o *dialOptions) {
		o.streamInt = f
	}
}

// WithAuthority returns a DialOption that specifies the value to be used as
// the :authority pseudo-header. This value only works with WithInsecure and
// has no effect if TransportCredentials are present.
func WithAuthority(a string) DialOption {
	return func(o *dialOptions) {
		o.copts.Authority = a
	}
}

// WithChannelzParentID returns a DialOption that specifies the channelz ID of current ClientConn's
// parent. This function is used in nested channel creation (e.g. grpclb dial).
func WithChannelzParentID(id int64) DialOption {
	return func(o *dialOptions) {
		o.channelzParentID = id
	}
}

// WithDisableServiceConfig returns a DialOption that causes grpc to ignore any
// service config provided by the resolver and provides a hint to the resolver
// to not fetch service configs.
func WithDisableServiceConfig() DialOption {
	return func(o *dialOptions) {
		o.disableServiceConfig = true
	}
}

// Dial creates a client connection to the given target.
func Dial(target string, opts ...DialOption) (*ClientConn, error) {
	return DialContext(context.Background(), target, opts...)
}

// DialContext creates a client connection to the given target. By default, it's
// a non-blocking dial (the function won't wait for connections to be
// established, and connecting happens in the background). To make it a blocking
// dial, use WithBlock() dial option.
//
// In the non-blocking case, the ctx does not act against the connection. It
// only controls the setup steps.
//
// In the blocking case, ctx can be used to cancel or expire the pending
// connection. Once this function returns, the cancellation and expiration of
// ctx will be noop. Users should call ClientConn.Close to terminate all the
// pending operations after this function returns.
//
// The target name syntax is defined in
// https://github.com/grpc/grpc/blob/master/doc/naming.md.
// e.g. to use dns resolver, a "dns:///" prefix should be applied to the target.
func DialContext(ctx context.Context, target string, opts ...DialOption) (conn *ClientConn, err error) {
	cc := &ClientConn{
		target: target,
		csMgr:  &connectivityStateManager{},
		conns:  make(map[*addrConn]struct{}),

		blockingpicker: newPickerWrapper(),
	}
	cc.ctx, cc.cancel = context.WithCancel(context.Background())

	for _, opt := range opts {
		opt(&cc.dopts)
	}

	if channelz.IsOn() {
		if cc.dopts.channelzParentID != 0 {
			cc.channelzID = channelz.RegisterChannel(cc, cc.dopts.channelzParentID, target)
		} else {
			cc.channelzID = channelz.RegisterChannel(cc, 0, target)
		}
	}

	if !cc.dopts.insecure {
		if cc.dopts.copts.TransportCredentials == nil {
			return nil, errNoTransportSecurity
		}
	} else {
		if cc.dopts.copts.TransportCredentials != nil {
			return nil, errCredentialsConflict
		}
		for _, cd := range cc.dopts.copts.PerRPCCredentials {
			if cd.RequireTransportSecurity() {
				return nil, errTransportCredentialsMissing
			}
		}
	}

	cc.mkp = cc.dopts.copts.KeepaliveParams

	if cc.dopts.copts.Dialer == nil {
		cc.dopts.copts.Dialer = newProxyDialer(
			func(ctx context.Context, addr string) (net.Conn, error) {
				network, addr := parseDialTarget(addr)
				return dialContext(ctx, network, addr)
			},
		)
	}

	if cc.dopts.copts.UserAgent != "" {
		cc.dopts.copts.UserAgent += " " + grpcUA
	} else {
		cc.dopts.copts.UserAgent = grpcUA
	}

	if cc.dopts.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cc.dopts.timeout)
		defer cancel()
	}

	defer func() {
		select {
		case <-ctx.Done():
			conn, err = nil, ctx.Err()
		default:
		}

		if err != nil {
			cc.Close()
		}
	}()

	scSet := false
	if cc.dopts.scChan != nil {
		// Try to get an initial service config.
		select {
		case sc, ok := <-cc.dopts.scChan:
			if ok {
				cc.sc = sc
				scSet = true
			}
		default:
		}
	}
	if cc.dopts.bs == nil {
		cc.dopts.bs = backoff.Exponential{
			MaxDelay: DefaultBackoffConfig.MaxDelay,
		}
	}
	if cc.dopts.resolverBuilder == nil {
		// Only try to parse target when resolver builder is not already set.
		cc.parsedTarget = parseTarget(cc.target)
		grpclog.Infof("parsed scheme: %q", cc.parsedTarget.Scheme)
		cc.dopts.resolverBuilder = resolver.Get(cc.parsedTarget.Scheme)
		if cc.dopts.resolverBuilder == nil {
			// If resolver builder is still nil, the parse target's scheme is
			// not registered. Fallback to default resolver and set Endpoint to
			// the original unparsed target.
			grpclog.Infof("scheme %q not registered, fallback to default scheme", cc.parsedTarget.Scheme)
			cc.parsedTarget = resolver.Target{
				Scheme:   resolver.GetDefaultScheme(),
				Endpoint: target,
			}
			cc.dopts.resolverBuilder = resolver.Get(cc.parsedTarget.Scheme)
		}
	} else {
		cc.parsedTarget = resolver.Target{Endpoint: target}
	}
	creds := cc.dopts.copts.TransportCredentials
	if creds != nil && creds.Info().ServerName != "" {
		cc.authority = creds.Info().ServerName
	} else if cc.dopts.insecure && cc.dopts.copts.Authority != "" {
		cc.authority = cc.dopts.copts.Authority
	} else {
		// Use endpoint from "scheme://authority/endpoint" as the default
		// authority for ClientConn.
		cc.authority = cc.parsedTarget.Endpoint
	}

	if cc.dopts.scChan != nil && !scSet {
		// Blocking wait for the initial service config.
		select {
		case sc, ok := <-cc.dopts.scChan:
			if ok {
				cc.sc = sc
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if cc.dopts.scChan != nil {
		go cc.scWatcher()
	}

	var credsClone credentials.TransportCredentials
	if creds := cc.dopts.copts.TransportCredentials; creds != nil {
		credsClone = creds.Clone()
	}
	cc.balancerBuildOpts = balancer.BuildOptions{
		DialCreds:        credsClone,
		Dialer:           cc.dopts.copts.Dialer,
		ChannelzParentID: cc.channelzID,
	}

	// Build the resolver.
	cc.resolverWrapper, err = newCCResolverWrapper(cc)
	if err != nil {
		return nil, fmt.Errorf("failed to build resolver: %v", err)
	}
	// Start the resolver wrapper goroutine after resolverWrapper is created.
	//
	// If the goroutine is started before resolverWrapper is ready, the
	// following may happen: The goroutine sends updates to cc. cc forwards
	// those to balancer. Balancer creates new addrConn. addrConn fails to
	// connect, and calls resolveNow(). resolveNow() tries to use the non-ready
	// resolverWrapper.
	cc.resolverWrapper.start()

	// A blocking dial blocks until the clientConn is ready.
	if cc.dopts.block {
		for {
			s := cc.GetState()
			if s == connectivity.Ready {
				break
			}
			if !cc.WaitForStateChange(ctx, s) {
				// ctx got timeout or canceled.
				return nil, ctx.Err()
			}
		}
	}

	return cc, nil
}

// connectivityStateManager keeps the connectivity.State of ClientConn.
// This struct will eventually be exported so the balancers can access it.
type connectivityStateManager struct {
	mu         sync.Mutex
	state      connectivity.State
	notifyChan chan struct{}
}

// updateState updates the connectivity.State of ClientConn.
// If there's a change it notifies goroutines waiting on state change to
// happen.
func (csm *connectivityStateManager) updateState(state connectivity.State) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	if csm.state == connectivity.Shutdown {
		return
	}
	if csm.state == state {
		return
	}
	csm.state = state
	if csm.notifyChan != nil {
		// There are other goroutines waiting on this channel.
		close(csm.notifyChan)
		csm.notifyChan = nil
	}
}

func (csm *connectivityStateManager) getState() connectivity.State {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	return csm.state
}

func (csm *connectivityStateManager) getNotifyChan() <-chan struct{} {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	if csm.notifyChan == nil {
		csm.notifyChan = make(chan struct{})
	}
	return csm.notifyChan
}

// ClientConn represents a client connection to an RPC server.
type ClientConn struct {
	ctx    context.Context
	cancel context.CancelFunc

	target       string
	parsedTarget resolver.Target
	authority    string
	dopts        dialOptions
	csMgr        *connectivityStateManager

	balancerBuildOpts balancer.BuildOptions
	resolverWrapper   *ccResolverWrapper
	blockingpicker    *pickerWrapper

	mu    sync.RWMutex
	sc    ServiceConfig
	scRaw string
	conns map[*addrConn]struct{}
	// Keepalive parameter can be updated if a GoAway is received.
	mkp             keepalive.ClientParameters
	curBalancerName string
	preBalancerName string // previous balancer name.
	curAddresses    []resolver.Address
	balancerWrapper *ccBalancerWrapper

	channelzID          int64 // channelz unique identification number
	czmu                sync.RWMutex
	callsStarted        int64
	callsSucceeded      int64
	callsFailed         int64
	lastCallStartedTime time.Time
}

// WaitForStateChange waits until the connectivity.State of ClientConn changes from sourceState or
// ctx expires. A true value is returned in former case and false in latter.
// This is an EXPERIMENTAL API.
func (cc *ClientConn) WaitForStateChange(ctx context.Context, sourceState connectivity.State) bool {
	ch := cc.csMgr.getNotifyChan()
	if cc.csMgr.getState() != sourceState {
		return true
	}
	select {
	case <-ctx.Done():
		return false
	case <-ch:
		return true
	}
}

// GetState returns the connectivity.State of ClientConn.
// This is an EXPERIMENTAL API.
func (cc *ClientConn) GetState() connectivity.State {
	return cc.csMgr.getState()
}

func (cc *ClientConn) scWatcher() {
	for {
		select {
		case sc, ok := <-cc.dopts.scChan:
			if !ok {
				return
			}
			cc.mu.Lock()
			// TODO: load balance policy runtime change is ignored.
			// We may revist this decision in the future.
			cc.sc = sc
			cc.scRaw = ""
			cc.mu.Unlock()
		case <-cc.ctx.Done():
			return
		}
	}
}

func (cc *ClientConn) handleResolvedAddrs(addrs []resolver.Address, err error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if cc.conns == nil {
		// cc was closed.
		return
	}

	if reflect.DeepEqual(cc.curAddresses, addrs) {
		return
	}

	cc.curAddresses = addrs

	if cc.dopts.balancerBuilder == nil {
		// Only look at balancer types and switch balancer if balancer dial
		// option is not set.
		var isGRPCLB bool
		for _, a := range addrs {
			if a.Type == resolver.GRPCLB {
				isGRPCLB = true
				break
			}
		}
		var newBalancerName string
		if isGRPCLB {
			newBalancerName = grpclbName
		} else {
			// Address list doesn't contain grpclb address. Try to pick a
			// non-grpclb balancer.
			newBalancerName = cc.curBalancerName
			// If current balancer is grpclb, switch to the previous one.
			if newBalancerName == grpclbName {
				newBalancerName = cc.preBalancerName
			}
			// The following could be true in two cases:
			// - the first time handling resolved addresses
			//   (curBalancerName="")
			// - the first time handling non-grpclb addresses
			//   (curBalancerName="grpclb", preBalancerName="")
			if newBalancerName == "" {
				newBalancerName = PickFirstBalancerName
			}
		}
		cc.switchBalancer(newBalancerName)
	} else if cc.balancerWrapper == nil {
		// Balancer dial option was set, and this is the first time handling
		// resolved addresses. Build a balancer with dopts.balancerBuilder.
		cc.balancerWrapper = newCCBalancerWrapper(cc, cc.dopts.balancerBuilder, cc.balancerBuildOpts)
	}

	cc.balancerWrapper.handleResolvedAddrs(addrs, nil)
}

// switchBalancer starts the switching from current balancer to the balancer
// with the given name.
//
// It will NOT send the current address list to the new balancer. If needed,
// caller of this function should send address list to the new balancer after
// this function returns.
//
// Caller must hold cc.mu.
func (cc *ClientConn) switchBalancer(name string) {
	if cc.conns == nil {
		return
	}

	if strings.ToLower(cc.curBalancerName) == strings.ToLower(name) {
		return
	}

	grpclog.Infof("ClientConn switching balancer to %q", name)
	if cc.dopts.balancerBuilder != nil {
		grpclog.Infoln("ignoring balancer switching: Balancer DialOption used instead")
		return
	}
	// TODO(bar switching) change this to two steps: drain and close.
	// Keep track of sc in wrapper.
	if cc.balancerWrapper != nil {
		cc.balancerWrapper.close()
	}
	// Clear all stickiness state.
	cc.blockingpicker.clearStickinessState()

	builder := balancer.Get(name)
	if builder == nil {
		grpclog.Infof("failed to get balancer builder for: %v, using pick_first instead", name)
		builder = newPickfirstBuilder()
	}
	cc.preBalancerName = cc.curBalancerName
	cc.curBalancerName = builder.Name()
	cc.balancerWrapper = newCCBalancerWrapper(cc, builder, cc.balancerBuildOpts)
}

func (cc *ClientConn) handleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	cc.mu.Lock()
	if cc.conns == nil {
		cc.mu.Unlock()
		return
	}
	// TODO(bar switching) send updates to all balancer wrappers when balancer
	// gracefully switching is supported.
	cc.balancerWrapper.handleSubConnStateChange(sc, s)
	cc.mu.Unlock()
}

// newAddrConn creates an addrConn for addrs and adds it to cc.conns.
//
// Caller needs to make sure len(addrs) > 0.
func (cc *ClientConn) newAddrConn(addrs []resolver.Address) (*addrConn, error) {
	ac := &addrConn{
		cc:    cc,
		addrs: addrs,
		dopts: cc.dopts,
	}
	ac.ctx, ac.cancel = context.WithCancel(cc.ctx)
	// Track ac in cc. This needs to be done before any getTransport(...) is called.
	cc.mu.Lock()
	if cc.conns == nil {
		cc.mu.Unlock()
		return nil, ErrClientConnClosing
	}
	if channelz.IsOn() {
		ac.channelzID = channelz.RegisterSubChannel(ac, cc.channelzID, "")
	}
	cc.conns[ac] = struct{}{}
	cc.mu.Unlock()
	return ac, nil
}

// removeAddrConn removes the addrConn in the subConn from clientConn.
// It also tears down the ac with the given error.
func (cc *ClientConn) removeAddrConn(ac *addrConn, err error) {
	cc.mu.Lock()
	if cc.conns == nil {
		cc.mu.Unlock()
		return
	}
	delete(cc.conns, ac)
	cc.mu.Unlock()
	ac.tearDown(err)
}

// ChannelzMetric returns ChannelInternalMetric of current ClientConn.
// This is an EXPERIMENTAL API.
func (cc *ClientConn) ChannelzMetric() *channelz.ChannelInternalMetric {
	state := cc.GetState()
	cc.czmu.RLock()
	defer cc.czmu.RUnlock()
	return &channelz.ChannelInternalMetric{
		State:                    state,
		Target:                   cc.target,
		CallsStarted:             cc.callsStarted,
		CallsSucceeded:           cc.callsSucceeded,
		CallsFailed:              cc.callsFailed,
		LastCallStartedTimestamp: cc.lastCallStartedTime,
	}
}

func (cc *ClientConn) incrCallsStarted() {
	cc.czmu.Lock()
	cc.callsStarted++
	// TODO(yuxuanli): will make this a time.Time pointer improve performance?
	cc.lastCallStartedTime = time.Now()
	cc.czmu.Unlock()
}

func (cc *ClientConn) incrCallsSucceeded() {
	cc.czmu.Lock()
	cc.callsSucceeded++
	cc.czmu.Unlock()
}

func (cc *ClientConn) incrCallsFailed() {
	cc.czmu.Lock()
	cc.callsFailed++
	cc.czmu.Unlock()
}

// connect starts to creating transport and also starts the transport monitor
// goroutine for this ac.
// It does nothing if the ac is not IDLE.
// TODO(bar) Move this to the addrConn section.
// This was part of resetAddrConn, keep it here to make the diff look clean.
func (ac *addrConn) connect() error {
	ac.mu.Lock()
	if ac.state == connectivity.Shutdown {
		ac.mu.Unlock()
		return errConnClosing
	}
	if ac.state != connectivity.Idle {
		ac.mu.Unlock()
		return nil
	}
	ac.state = connectivity.Connecting
	ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
	ac.mu.Unlock()

	// Start a goroutine connecting to the server asynchronously.
	go func() {
		if err := ac.resetTransport(); err != nil {
			grpclog.Warningf("Failed to dial %s: %v; please retry.", ac.addrs[0].Addr, err)
			if err != errConnClosing {
				// Keep this ac in cc.conns, to get the reason it's torn down.
				ac.tearDown(err)
			}
			return
		}
		ac.transportMonitor()
	}()
	return nil
}

// tryUpdateAddrs tries to update ac.addrs with the new addresses list.
//
// It checks whether current connected address of ac is in the new addrs list.
//  - If true, it updates ac.addrs and returns true. The ac will keep using
//    the existing connection.
//  - If false, it does nothing and returns false.
func (ac *addrConn) tryUpdateAddrs(addrs []resolver.Address) bool {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	grpclog.Infof("addrConn: tryUpdateAddrs curAddr: %v, addrs: %v", ac.curAddr, addrs)
	if ac.state == connectivity.Shutdown {
		ac.addrs = addrs
		return true
	}

	var curAddrFound bool
	for _, a := range addrs {
		if reflect.DeepEqual(ac.curAddr, a) {
			curAddrFound = true
			break
		}
	}
	grpclog.Infof("addrConn: tryUpdateAddrs curAddrFound: %v", curAddrFound)
	if curAddrFound {
		ac.addrs = addrs
		ac.reconnectIdx = 0 // Start reconnecting from beginning in the new list.
	}

	return curAddrFound
}

// GetMethodConfig gets the method config of the input method.
// If there's an exact match for input method (i.e. /service/method), we return
// the corresponding MethodConfig.
// If there isn't an exact match for the input method, we look for the default config
// under the service (i.e /service/). If there is a default MethodConfig for
// the service, we return it.
// Otherwise, we return an empty MethodConfig.
func (cc *ClientConn) GetMethodConfig(method string) MethodConfig {
	// TODO: Avoid the locking here.
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	m, ok := cc.sc.Methods[method]
	if !ok {
		i := strings.LastIndex(method, "/")
		m = cc.sc.Methods[method[:i+1]]
	}
	return m
}

func (cc *ClientConn) getTransport(ctx context.Context, failfast bool) (transport.ClientTransport, func(balancer.DoneInfo), error) {
	t, done, err := cc.blockingpicker.pick(ctx, failfast, balancer.PickOptions{})
	if err != nil {
		return nil, nil, toRPCErr(err)
	}
	return t, done, nil
}

// handleServiceConfig parses the service config string in JSON format to Go native
// struct ServiceConfig, and store both the struct and the JSON string in ClientConn.
func (cc *ClientConn) handleServiceConfig(js string) error {
	if cc.dopts.disableServiceConfig {
		return nil
	}
	sc, err := parseServiceConfig(js)
	if err != nil {
		return err
	}
	cc.mu.Lock()
	cc.scRaw = js
	cc.sc = sc
	if sc.LB != nil && *sc.LB != grpclbName { // "grpclb" is not a valid balancer option in service config.
		if cc.curBalancerName == grpclbName {
			// If current balancer is grpclb, there's at least one grpclb
			// balancer address in the resolved list. Don't switch the balancer,
			// but change the previous balancer name, so if a new resolved
			// address list doesn't contain grpclb address, balancer will be
			// switched to *sc.LB.
			cc.preBalancerName = *sc.LB
		} else {
			cc.switchBalancer(*sc.LB)
			cc.balancerWrapper.handleResolvedAddrs(cc.curAddresses, nil)
		}
	}

	if envConfigStickinessOn {
		var newStickinessMDKey string
		if sc.stickinessMetadataKey != nil && *sc.stickinessMetadataKey != "" {
			newStickinessMDKey = *sc.stickinessMetadataKey
		}
		// newStickinessMDKey is "" if one of the following happens:
		// - stickinessMetadataKey is set to ""
		// - stickinessMetadataKey field doesn't exist in service config
		cc.blockingpicker.updateStickinessMDKey(strings.ToLower(newStickinessMDKey))
	}

	cc.mu.Unlock()
	return nil
}

func (cc *ClientConn) resolveNow(o resolver.ResolveNowOption) {
	cc.mu.RLock()
	r := cc.resolverWrapper
	cc.mu.RUnlock()
	if r == nil {
		return
	}
	go r.resolveNow(o)
}

// Close tears down the ClientConn and all underlying connections.
func (cc *ClientConn) Close() error {
	defer cc.cancel()

	cc.mu.Lock()
	if cc.conns == nil {
		cc.mu.Unlock()
		return ErrClientConnClosing
	}
	conns := cc.conns
	cc.conns = nil
	cc.csMgr.updateState(connectivity.Shutdown)

	rWrapper := cc.resolverWrapper
	cc.resolverWrapper = nil
	bWrapper := cc.balancerWrapper
	cc.balancerWrapper = nil
	cc.mu.Unlock()

	cc.blockingpicker.close()

	if rWrapper != nil {
		rWrapper.close()
	}
	if bWrapper != nil {
		bWrapper.close()
	}

	for ac := range conns {
		ac.tearDown(ErrClientConnClosing)
	}
	if channelz.IsOn() {
		channelz.RemoveEntry(cc.channelzID)
	}
	return nil
}

// addrConn is a network connection to a given address.
type addrConn struct {
	ctx    context.Context
	cancel context.CancelFunc

	cc     *ClientConn
	addrs  []resolver.Address
	dopts  dialOptions
	events trace.EventLog
	acbw   balancer.SubConn

	mu           sync.Mutex
	curAddr      resolver.Address
	reconnectIdx int // The index in addrs list to start reconnecting from.
	state        connectivity.State
	// ready is closed and becomes nil when a new transport is up or failed
	// due to timeout.
	ready     chan struct{}
	transport transport.ClientTransport

	// The reason this addrConn is torn down.
	tearDownErr error

	connectRetryNum int
	// backoffDeadline is the time until which resetTransport needs to
	// wait before increasing connectRetryNum count.
	backoffDeadline time.Time
	// connectDeadline is the time by which all connection
	// negotiations must complete.
	connectDeadline time.Time

	channelzID          int64 // channelz unique identification number
	czmu                sync.RWMutex
	callsStarted        int64
	callsSucceeded      int64
	callsFailed         int64
	lastCallStartedTime time.Time
}

// adjustParams updates parameters used to create transports upon
// receiving a GoAway.
func (ac *addrConn) adjustParams(r transport.GoAwayReason) {
	switch r {
	case transport.GoAwayTooManyPings:
		v := 2 * ac.dopts.copts.KeepaliveParams.Time
		ac.cc.mu.Lock()
		if v > ac.cc.mkp.Time {
			ac.cc.mkp.Time = v
		}
		ac.cc.mu.Unlock()
	}
}

// printf records an event in ac's event log, unless ac has been closed.
// REQUIRES ac.mu is held.
func (ac *addrConn) printf(format string, a ...interface{}) {
	if ac.events != nil {
		ac.events.Printf(format, a...)
	}
}

// errorf records an error in ac's event log, unless ac has been closed.
// REQUIRES ac.mu is held.
func (ac *addrConn) errorf(format string, a ...interface{}) {
	if ac.events != nil {
		ac.events.Errorf(format, a...)
	}
}

// resetTransport recreates a transport to the address for ac.  The old
// transport will close itself on error or when the clientconn is closed.
// The created transport must receive initial settings frame from the server.
// In case that doesn't happen, transportMonitor will kill the newly created
// transport after connectDeadline has expired.
// In case there was an error on the transport before the settings frame was
// received, resetTransport resumes connecting to backends after the one that
// was previously connected to. In case end of the list is reached, resetTransport
// backs off until the original deadline.
// If the DialOption WithWaitForHandshake was set, resetTrasport returns
// successfully only after server settings are received.
//
// TODO(bar) make sure all state transitions are valid.
func (ac *addrConn) resetTransport() error {
	ac.mu.Lock()
	if ac.state == connectivity.Shutdown {
		ac.mu.Unlock()
		return errConnClosing
	}
	if ac.ready != nil {
		close(ac.ready)
		ac.ready = nil
	}
	ac.transport = nil
	ridx := ac.reconnectIdx
	ac.mu.Unlock()
	ac.cc.mu.RLock()
	ac.dopts.copts.KeepaliveParams = ac.cc.mkp
	ac.cc.mu.RUnlock()
	var backoffDeadline, connectDeadline time.Time
	for connectRetryNum := 0; ; connectRetryNum++ {
		ac.mu.Lock()
		if ac.backoffDeadline.IsZero() {
			// This means either a successful HTTP2 connection was established
			// or this is the first time this addrConn is trying to establish a
			// connection.
			backoffFor := ac.dopts.bs.Backoff(connectRetryNum) // time.Duration.
			// This will be the duration that dial gets to finish.
			dialDuration := getMinConnectTimeout()
			if backoffFor > dialDuration {
				// Give dial more time as we keep failing to connect.
				dialDuration = backoffFor
			}
			start := time.Now()
			backoffDeadline = start.Add(backoffFor)
			connectDeadline = start.Add(dialDuration)
			ridx = 0 // Start connecting from the beginning.
		} else {
			// Continue trying to connect with the same deadlines.
			connectRetryNum = ac.connectRetryNum
			backoffDeadline = ac.backoffDeadline
			connectDeadline = ac.connectDeadline
			ac.backoffDeadline = time.Time{}
			ac.connectDeadline = time.Time{}
			ac.connectRetryNum = 0
		}
		if ac.state == connectivity.Shutdown {
			ac.mu.Unlock()
			return errConnClosing
		}
		ac.printf("connecting")
		if ac.state != connectivity.Connecting {
			ac.state = connectivity.Connecting
			ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
		}
		// copy ac.addrs in case of race
		addrsIter := make([]resolver.Address, len(ac.addrs))
		copy(addrsIter, ac.addrs)
		copts := ac.dopts.copts
		ac.mu.Unlock()
		connected, err := ac.createTransport(connectRetryNum, ridx, backoffDeadline, connectDeadline, addrsIter, copts)
		if err != nil {
			return err
		}
		if connected {
			return nil
		}
	}
}

// createTransport creates a connection to one of the backends in addrs.
// It returns true if a connection was established.
func (ac *addrConn) createTransport(connectRetryNum, ridx int, backoffDeadline, connectDeadline time.Time, addrs []resolver.Address, copts transport.ConnectOptions) (bool, error) {
	for i := ridx; i < len(addrs); i++ {
		addr := addrs[i]
		target := transport.TargetInfo{
			Addr:      addr.Addr,
			Metadata:  addr.Metadata,
			Authority: ac.cc.authority,
		}
		done := make(chan struct{})
		onPrefaceReceipt := func() {
			ac.mu.Lock()
			close(done)
			if !ac.backoffDeadline.IsZero() {
				// If we haven't already started reconnecting to
				// other backends.
				// Note, this can happen when writer notices an error
				// and triggers resetTransport while at the same time
				// reader receives the preface and invokes this closure.
				ac.backoffDeadline = time.Time{}
				ac.connectDeadline = time.Time{}
				ac.connectRetryNum = 0
			}
			ac.mu.Unlock()
		}
		// Do not cancel in the success path because of
		// this issue in Go1.6: https://github.com/golang/go/issues/15078.
		connectCtx, cancel := context.WithDeadline(ac.ctx, connectDeadline)
		if channelz.IsOn() {
			copts.ChannelzParentID = ac.channelzID
		}
		newTr, err := transport.NewClientTransport(connectCtx, ac.cc.ctx, target, copts, onPrefaceReceipt)
		if err != nil {
			cancel()
			ac.cc.blockingpicker.updateConnectionError(err)
			ac.mu.Lock()
			if ac.state == connectivity.Shutdown {
				// ac.tearDown(...) has been invoked.
				ac.mu.Unlock()
				return false, errConnClosing
			}
			ac.mu.Unlock()
			grpclog.Warningf("grpc: addrConn.createTransport failed to connect to %v. Err :%v. Reconnecting...", addr, err)
			continue
		}
		if ac.dopts.waitForHandshake {
			select {
			case <-done:
			case <-connectCtx.Done():
				// Didn't receive server preface, must kill this new transport now.
				grpclog.Warningf("grpc: addrConn.createTransport failed to receive server preface before deadline.")
				newTr.Close()
				break
			case <-ac.ctx.Done():
			}
		}
		ac.mu.Lock()
		if ac.state == connectivity.Shutdown {
			ac.mu.Unlock()
			// ac.tearDonn(...) has been invoked.
			newTr.Close()
			return false, errConnClosing
		}
		ac.printf("ready")
		ac.state = connectivity.Ready
		ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
		ac.transport = newTr
		ac.curAddr = addr
		if ac.ready != nil {
			close(ac.ready)
			ac.ready = nil
		}
		select {
		case <-done:
			// If the server has responded back with preface already,
			// don't set the reconnect parameters.
		default:
			ac.connectRetryNum = connectRetryNum
			ac.backoffDeadline = backoffDeadline
			ac.connectDeadline = connectDeadline
			ac.reconnectIdx = i + 1 // Start reconnecting from the next backend in the list.
		}
		ac.mu.Unlock()
		return true, nil
	}
	ac.mu.Lock()
	if ac.state == connectivity.Shutdown {
		ac.mu.Unlock()
		return false, errConnClosing
	}
	ac.state = connectivity.TransientFailure
	ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
	ac.cc.resolveNow(resolver.ResolveNowOption{})
	if ac.ready != nil {
		close(ac.ready)
		ac.ready = nil
	}
	ac.mu.Unlock()
	timer := time.NewTimer(backoffDeadline.Sub(time.Now()))
	select {
	case <-timer.C:
	case <-ac.ctx.Done():
		timer.Stop()
		return false, ac.ctx.Err()
	}
	return false, nil
}

// Run in a goroutine to track the error in transport and create the
// new transport if an error happens. It returns when the channel is closing.
func (ac *addrConn) transportMonitor() {
	for {
		var timer *time.Timer
		var cdeadline <-chan time.Time
		ac.mu.Lock()
		t := ac.transport
		if !ac.connectDeadline.IsZero() {
			timer = time.NewTimer(ac.connectDeadline.Sub(time.Now()))
			cdeadline = timer.C
		}
		ac.mu.Unlock()
		// Block until we receive a goaway or an error occurs.
		select {
		case <-t.GoAway():
			done := t.Error()
			cleanup := t.Close
			// Since this transport will be orphaned (won't have a transportMonitor)
			// we need to launch a goroutine to keep track of clientConn.Close()
			// happening since it might not be noticed by any other goroutine for a while.
			go func() {
				<-done
				cleanup()
			}()
		case <-t.Error():
			// In case this is triggered because clientConn.Close()
			// was called, we want to immeditately close the transport
			// since no other goroutine might notice it for a while.
			t.Close()
		case <-cdeadline:
			ac.mu.Lock()
			// This implies that client received server preface.
			if ac.backoffDeadline.IsZero() {
				ac.mu.Unlock()
				continue
			}
			ac.mu.Unlock()
			timer = nil
			// No server preface received until deadline.
			// Kill the connection.
			grpclog.Warningf("grpc: addrConn.transportMonitor didn't get server preface after waiting. Closing the new transport now.")
			t.Close()
		}
		if timer != nil {
			timer.Stop()
		}
		// If a GoAway happened, regardless of error, adjust our keepalive
		// parameters as appropriate.
		select {
		case <-t.GoAway():
			ac.adjustParams(t.GetGoAwayReason())
		default:
		}
		ac.mu.Lock()
		if ac.state == connectivity.Shutdown {
			ac.mu.Unlock()
			return
		}
		// Set connectivity state to TransientFailure before calling
		// resetTransport. Transition READY->CONNECTING is not valid.
		ac.state = connectivity.TransientFailure
		ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
		ac.cc.resolveNow(resolver.ResolveNowOption{})
		ac.curAddr = resolver.Address{}
		ac.mu.Unlock()
		if err := ac.resetTransport(); err != nil {
			ac.mu.Lock()
			ac.printf("transport exiting: %v", err)
			ac.mu.Unlock()
			grpclog.Warningf("grpc: addrConn.transportMonitor exits due to: %v", err)
			if err != errConnClosing {
				// Keep this ac in cc.conns, to get the reason it's torn down.
				ac.tearDown(err)
			}
			return
		}
	}
}

// wait blocks until i) the new transport is up or ii) ctx is done or iii) ac is closed or
// iv) transport is in connectivity.TransientFailure and there is a balancer/failfast is true.
func (ac *addrConn) wait(ctx context.Context, hasBalancer, failfast bool) (transport.ClientTransport, error) {
	for {
		ac.mu.Lock()
		switch {
		case ac.state == connectivity.Shutdown:
			if failfast || !hasBalancer {
				// RPC is failfast or balancer is nil. This RPC should fail with ac.tearDownErr.
				err := ac.tearDownErr
				ac.mu.Unlock()
				return nil, err
			}
			ac.mu.Unlock()
			return nil, errConnClosing
		case ac.state == connectivity.Ready:
			ct := ac.transport
			ac.mu.Unlock()
			return ct, nil
		case ac.state == connectivity.TransientFailure:
			if failfast || hasBalancer {
				ac.mu.Unlock()
				return nil, errConnUnavailable
			}
		}
		ready := ac.ready
		if ready == nil {
			ready = make(chan struct{})
			ac.ready = ready
		}
		ac.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, toRPCErr(ctx.Err())
		// Wait until the new transport is ready or failed.
		case <-ready:
		}
	}
}

// getReadyTransport returns the transport if ac's state is READY.
// Otherwise it returns nil, false.
// If ac's state is IDLE, it will trigger ac to connect.
func (ac *addrConn) getReadyTransport() (transport.ClientTransport, bool) {
	ac.mu.Lock()
	if ac.state == connectivity.Ready {
		t := ac.transport
		ac.mu.Unlock()
		return t, true
	}
	var idle bool
	if ac.state == connectivity.Idle {
		idle = true
	}
	ac.mu.Unlock()
	// Trigger idle ac to connect.
	if idle {
		ac.connect()
	}
	return nil, false
}

// tearDown starts to tear down the addrConn.
// TODO(zhaoq): Make this synchronous to avoid unbounded memory consumption in
// some edge cases (e.g., the caller opens and closes many addrConn's in a
// tight loop.
// tearDown doesn't remove ac from ac.cc.conns.
func (ac *addrConn) tearDown(err error) {
	ac.cancel()
	ac.mu.Lock()
	defer ac.mu.Unlock()
	if ac.state == connectivity.Shutdown {
		return
	}
	ac.curAddr = resolver.Address{}
	if err == errConnDrain && ac.transport != nil {
		// GracefulClose(...) may be executed multiple times when
		// i) receiving multiple GoAway frames from the server; or
		// ii) there are concurrent name resolver/Balancer triggered
		// address removal and GoAway.
		ac.transport.GracefulClose()
	}
	ac.state = connectivity.Shutdown
	ac.tearDownErr = err
	ac.cc.handleSubConnStateChange(ac.acbw, ac.state)
	if ac.events != nil {
		ac.events.Finish()
		ac.events = nil
	}
	if ac.ready != nil {
		close(ac.ready)
		ac.ready = nil
	}
	if channelz.IsOn() {
		channelz.RemoveEntry(ac.channelzID)
	}
}

func (ac *addrConn) getState() connectivity.State {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	return ac.state
}

func (ac *addrConn) getCurAddr() (ret resolver.Address) {
	ac.mu.Lock()
	ret = ac.curAddr
	ac.mu.Unlock()
	return
}

func (ac *addrConn) ChannelzMetric() *channelz.ChannelInternalMetric {
	ac.mu.Lock()
	addr := ac.curAddr.Addr
	ac.mu.Unlock()
	state := ac.getState()
	ac.czmu.RLock()
	defer ac.czmu.RUnlock()
	return &channelz.ChannelInternalMetric{
		State:                    state,
		Target:                   addr,
		CallsStarted:             ac.callsStarted,
		CallsSucceeded:           ac.callsSucceeded,
		CallsFailed:              ac.callsFailed,
		LastCallStartedTimestamp: ac.lastCallStartedTime,
	}
}

func (ac *addrConn) incrCallsStarted() {
	ac.czmu.Lock()
	ac.callsStarted++
	ac.lastCallStartedTime = time.Now()
	ac.czmu.Unlock()
}

func (ac *addrConn) incrCallsSucceeded() {
	ac.czmu.Lock()
	ac.callsSucceeded++
	ac.czmu.Unlock()
}

func (ac *addrConn) incrCallsFailed() {
	ac.czmu.Lock()
	ac.callsFailed++
	ac.czmu.Unlock()
}

// ErrClientConnTimeout indicates that the ClientConn cannot establish the
// underlying connections within the specified timeout.
//
// Deprecated: This error is never returned by grpc and should not be
// referenced by users.
var ErrClientConnTimeout = errors.New("grpc: timed out when dialing")
