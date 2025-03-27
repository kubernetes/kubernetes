/*
 *
 * Copyright 2017 gRPC authors.
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
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/balancer/gracefulswitch"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/status"
)

var (
	setConnectedAddress = internal.SetConnectedAddress.(func(*balancer.SubConnState, resolver.Address))
	// noOpRegisterHealthListenerFn is used when client side health checking is
	// disabled. It sends a single READY update on the registered listener.
	noOpRegisterHealthListenerFn = func(_ context.Context, listener func(balancer.SubConnState)) func() {
		listener(balancer.SubConnState{ConnectivityState: connectivity.Ready})
		return func() {}
	}
)

// ccBalancerWrapper sits between the ClientConn and the Balancer.
//
// ccBalancerWrapper implements methods corresponding to the ones on the
// balancer.Balancer interface. The ClientConn is free to call these methods
// concurrently and the ccBalancerWrapper ensures that calls from the ClientConn
// to the Balancer happen in order by performing them in the serializer, without
// any mutexes held.
//
// ccBalancerWrapper also implements the balancer.ClientConn interface and is
// passed to the Balancer implementations. It invokes unexported methods on the
// ClientConn to handle these calls from the Balancer.
//
// It uses the gracefulswitch.Balancer internally to ensure that balancer
// switches happen in a graceful manner.
type ccBalancerWrapper struct {
	// The following fields are initialized when the wrapper is created and are
	// read-only afterwards, and therefore can be accessed without a mutex.
	cc               *ClientConn
	opts             balancer.BuildOptions
	serializer       *grpcsync.CallbackSerializer
	serializerCancel context.CancelFunc

	// The following fields are only accessed within the serializer or during
	// initialization.
	curBalancerName string
	balancer        *gracefulswitch.Balancer

	// The following field is protected by mu.  Caller must take cc.mu before
	// taking mu.
	mu     sync.Mutex
	closed bool
}

// newCCBalancerWrapper creates a new balancer wrapper in idle state. The
// underlying balancer is not created until the updateClientConnState() method
// is invoked.
func newCCBalancerWrapper(cc *ClientConn) *ccBalancerWrapper {
	ctx, cancel := context.WithCancel(cc.ctx)
	ccb := &ccBalancerWrapper{
		cc: cc,
		opts: balancer.BuildOptions{
			DialCreds:       cc.dopts.copts.TransportCredentials,
			CredsBundle:     cc.dopts.copts.CredsBundle,
			Dialer:          cc.dopts.copts.Dialer,
			Authority:       cc.authority,
			CustomUserAgent: cc.dopts.copts.UserAgent,
			ChannelzParent:  cc.channelz,
			Target:          cc.parsedTarget,
			MetricsRecorder: cc.metricsRecorderList,
		},
		serializer:       grpcsync.NewCallbackSerializer(ctx),
		serializerCancel: cancel,
	}
	ccb.balancer = gracefulswitch.NewBalancer(ccb, ccb.opts)
	return ccb
}

// updateClientConnState is invoked by grpc to push a ClientConnState update to
// the underlying balancer.  This is always executed from the serializer, so
// it is safe to call into the balancer here.
func (ccb *ccBalancerWrapper) updateClientConnState(ccs *balancer.ClientConnState) error {
	errCh := make(chan error)
	uccs := func(ctx context.Context) {
		defer close(errCh)
		if ctx.Err() != nil || ccb.balancer == nil {
			return
		}
		name := gracefulswitch.ChildName(ccs.BalancerConfig)
		if ccb.curBalancerName != name {
			ccb.curBalancerName = name
			channelz.Infof(logger, ccb.cc.channelz, "Channel switches to new LB policy %q", name)
		}
		err := ccb.balancer.UpdateClientConnState(*ccs)
		if logger.V(2) && err != nil {
			logger.Infof("error from balancer.UpdateClientConnState: %v", err)
		}
		errCh <- err
	}
	onFailure := func() { close(errCh) }

	// UpdateClientConnState can race with Close, and when the latter wins, the
	// serializer is closed, and the attempt to schedule the callback will fail.
	// It is acceptable to ignore this failure. But since we want to handle the
	// state update in a blocking fashion (when we successfully schedule the
	// callback), we have to use the ScheduleOr method and not the MaybeSchedule
	// method on the serializer.
	ccb.serializer.ScheduleOr(uccs, onFailure)
	return <-errCh
}

// resolverError is invoked by grpc to push a resolver error to the underlying
// balancer.  The call to the balancer is executed from the serializer.
func (ccb *ccBalancerWrapper) resolverError(err error) {
	ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || ccb.balancer == nil {
			return
		}
		ccb.balancer.ResolverError(err)
	})
}

// close initiates async shutdown of the wrapper.  cc.mu must be held when
// calling this function.  To determine the wrapper has finished shutting down,
// the channel should block on ccb.serializer.Done() without cc.mu held.
func (ccb *ccBalancerWrapper) close() {
	ccb.mu.Lock()
	ccb.closed = true
	ccb.mu.Unlock()
	channelz.Info(logger, ccb.cc.channelz, "ccBalancerWrapper: closing")
	ccb.serializer.TrySchedule(func(context.Context) {
		if ccb.balancer == nil {
			return
		}
		ccb.balancer.Close()
		ccb.balancer = nil
	})
	ccb.serializerCancel()
}

// exitIdle invokes the balancer's exitIdle method in the serializer.
func (ccb *ccBalancerWrapper) exitIdle() {
	ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || ccb.balancer == nil {
			return
		}
		ccb.balancer.ExitIdle()
	})
}

func (ccb *ccBalancerWrapper) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	ccb.cc.mu.Lock()
	defer ccb.cc.mu.Unlock()

	ccb.mu.Lock()
	if ccb.closed {
		ccb.mu.Unlock()
		return nil, fmt.Errorf("balancer is being closed; no new SubConns allowed")
	}
	ccb.mu.Unlock()

	if len(addrs) == 0 {
		return nil, fmt.Errorf("grpc: cannot create SubConn with empty address list")
	}
	ac, err := ccb.cc.newAddrConnLocked(addrs, opts)
	if err != nil {
		channelz.Warningf(logger, ccb.cc.channelz, "acBalancerWrapper: NewSubConn: failed to newAddrConn: %v", err)
		return nil, err
	}
	acbw := &acBalancerWrapper{
		ccb:           ccb,
		ac:            ac,
		producers:     make(map[balancer.ProducerBuilder]*refCountedProducer),
		stateListener: opts.StateListener,
		healthData:    newHealthData(connectivity.Idle),
	}
	ac.acbw = acbw
	return acbw, nil
}

func (ccb *ccBalancerWrapper) RemoveSubConn(balancer.SubConn) {
	// The graceful switch balancer will never call this.
	logger.Errorf("ccb RemoveSubConn(%v) called unexpectedly, sc")
}

func (ccb *ccBalancerWrapper) UpdateAddresses(sc balancer.SubConn, addrs []resolver.Address) {
	acbw, ok := sc.(*acBalancerWrapper)
	if !ok {
		return
	}
	acbw.UpdateAddresses(addrs)
}

func (ccb *ccBalancerWrapper) UpdateState(s balancer.State) {
	ccb.cc.mu.Lock()
	defer ccb.cc.mu.Unlock()
	if ccb.cc.conns == nil {
		// The CC has been closed; ignore this update.
		return
	}

	ccb.mu.Lock()
	if ccb.closed {
		ccb.mu.Unlock()
		return
	}
	ccb.mu.Unlock()
	// Update picker before updating state.  Even though the ordering here does
	// not matter, it can lead to multiple calls of Pick in the common start-up
	// case where we wait for ready and then perform an RPC.  If the picker is
	// updated later, we could call the "connecting" picker when the state is
	// updated, and then call the "ready" picker after the picker gets updated.

	// Note that there is no need to check if the balancer wrapper was closed,
	// as we know the graceful switch LB policy will not call cc if it has been
	// closed.
	ccb.cc.pickerWrapper.updatePicker(s.Picker)
	ccb.cc.csMgr.updateState(s.ConnectivityState)
}

func (ccb *ccBalancerWrapper) ResolveNow(o resolver.ResolveNowOptions) {
	ccb.cc.mu.RLock()
	defer ccb.cc.mu.RUnlock()

	ccb.mu.Lock()
	if ccb.closed {
		ccb.mu.Unlock()
		return
	}
	ccb.mu.Unlock()
	ccb.cc.resolveNowLocked(o)
}

func (ccb *ccBalancerWrapper) Target() string {
	return ccb.cc.target
}

// acBalancerWrapper is a wrapper on top of ac for balancers.
// It implements balancer.SubConn interface.
type acBalancerWrapper struct {
	internal.EnforceSubConnEmbedding
	ac            *addrConn          // read-only
	ccb           *ccBalancerWrapper // read-only
	stateListener func(balancer.SubConnState)

	producersMu sync.Mutex
	producers   map[balancer.ProducerBuilder]*refCountedProducer

	// Access to healthData is protected by healthMu.
	healthMu sync.Mutex
	// healthData is stored as a pointer to detect when the health listener is
	// dropped or updated. This is required as closures can't be compared for
	// equality.
	healthData *healthData
}

// healthData holds data related to health state reporting.
type healthData struct {
	// connectivityState stores the most recent connectivity state delivered
	// to the LB policy. This is stored to avoid sending updates when the
	// SubConn has already exited connectivity state READY.
	connectivityState connectivity.State
	// closeHealthProducer stores function to close the ref counted health
	// producer. The health producer is automatically closed when the SubConn
	// state changes.
	closeHealthProducer func()
}

func newHealthData(s connectivity.State) *healthData {
	return &healthData{
		connectivityState:   s,
		closeHealthProducer: func() {},
	}
}

// updateState is invoked by grpc to push a subConn state update to the
// underlying balancer.
func (acbw *acBalancerWrapper) updateState(s connectivity.State, curAddr resolver.Address, err error) {
	acbw.ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || acbw.ccb.balancer == nil {
			return
		}
		// Invalidate all producers on any state change.
		acbw.closeProducers()

		// Even though it is optional for balancers, gracefulswitch ensures
		// opts.StateListener is set, so this cannot ever be nil.
		// TODO: delete this comment when UpdateSubConnState is removed.
		scs := balancer.SubConnState{ConnectivityState: s, ConnectionError: err}
		if s == connectivity.Ready {
			setConnectedAddress(&scs, curAddr)
		}
		// Invalidate the health listener by updating the healthData.
		acbw.healthMu.Lock()
		// A race may occur if a health listener is registered soon after the
		// connectivity state is set but before the stateListener is called.
		// Two cases may arise:
		// 1. The new state is not READY: RegisterHealthListener has checks to
		//    ensure no updates are sent when the connectivity state is not
		//    READY.
		// 2. The new state is READY: This means that the old state wasn't Ready.
		//    The RegisterHealthListener API mentions that a health listener
		//    must not be registered when a SubConn is not ready to avoid such
		//    races. When this happens, the LB policy would get health updates
		//    on the old listener. When the LB policy registers a new listener
		//    on receiving the connectivity update, the health updates will be
		//    sent to the new health listener.
		acbw.healthData = newHealthData(scs.ConnectivityState)
		acbw.healthMu.Unlock()

		acbw.stateListener(scs)
	})
}

func (acbw *acBalancerWrapper) String() string {
	return fmt.Sprintf("SubConn(id:%d)", acbw.ac.channelz.ID)
}

func (acbw *acBalancerWrapper) UpdateAddresses(addrs []resolver.Address) {
	acbw.ac.updateAddrs(addrs)
}

func (acbw *acBalancerWrapper) Connect() {
	go acbw.ac.connect()
}

func (acbw *acBalancerWrapper) Shutdown() {
	acbw.closeProducers()
	acbw.ccb.cc.removeAddrConn(acbw.ac, errConnDrain)
}

// NewStream begins a streaming RPC on the addrConn.  If the addrConn is not
// ready, blocks until it is or ctx expires.  Returns an error when the context
// expires or the addrConn is shut down.
func (acbw *acBalancerWrapper) NewStream(ctx context.Context, desc *StreamDesc, method string, opts ...CallOption) (ClientStream, error) {
	transport := acbw.ac.getReadyTransport()
	if transport == nil {
		return nil, status.Errorf(codes.Unavailable, "SubConn state is not Ready")

	}
	return newNonRetryClientStream(ctx, desc, method, transport, acbw.ac, opts...)
}

// Invoke performs a unary RPC.  If the addrConn is not ready, returns
// errSubConnNotReady.
func (acbw *acBalancerWrapper) Invoke(ctx context.Context, method string, args any, reply any, opts ...CallOption) error {
	cs, err := acbw.NewStream(ctx, unaryStreamDesc, method, opts...)
	if err != nil {
		return err
	}
	if err := cs.SendMsg(args); err != nil {
		return err
	}
	return cs.RecvMsg(reply)
}

type refCountedProducer struct {
	producer balancer.Producer
	refs     int    // number of current refs to the producer
	close    func() // underlying producer's close function
}

func (acbw *acBalancerWrapper) GetOrBuildProducer(pb balancer.ProducerBuilder) (balancer.Producer, func()) {
	acbw.producersMu.Lock()
	defer acbw.producersMu.Unlock()

	// Look up existing producer from this builder.
	pData := acbw.producers[pb]
	if pData == nil {
		// Not found; create a new one and add it to the producers map.
		p, closeFn := pb.Build(acbw)
		pData = &refCountedProducer{producer: p, close: closeFn}
		acbw.producers[pb] = pData
	}
	// Account for this new reference.
	pData.refs++

	// Return a cleanup function wrapped in a OnceFunc to remove this reference
	// and delete the refCountedProducer from the map if the total reference
	// count goes to zero.
	unref := func() {
		acbw.producersMu.Lock()
		// If closeProducers has already closed this producer instance, refs is
		// set to 0, so the check after decrementing will never pass, and the
		// producer will not be double-closed.
		pData.refs--
		if pData.refs == 0 {
			defer pData.close() // Run outside the acbw mutex
			delete(acbw.producers, pb)
		}
		acbw.producersMu.Unlock()
	}
	return pData.producer, grpcsync.OnceFunc(unref)
}

func (acbw *acBalancerWrapper) closeProducers() {
	acbw.producersMu.Lock()
	defer acbw.producersMu.Unlock()
	for pb, pData := range acbw.producers {
		pData.refs = 0
		pData.close()
		delete(acbw.producers, pb)
	}
}

// healthProducerRegisterFn is a type alias for the health producer's function
// for registering listeners.
type healthProducerRegisterFn = func(context.Context, balancer.SubConn, string, func(balancer.SubConnState)) func()

// healthListenerRegFn returns a function to register a listener for health
// updates. If client side health checks are disabled, the registered listener
// will get a single READY (raw connectivity state) update.
//
// Client side health checking is enabled when all the following
// conditions are satisfied:
// 1. Health checking is not disabled using the dial option.
// 2. The health package is imported.
// 3. The health check config is present in the service config.
func (acbw *acBalancerWrapper) healthListenerRegFn() func(context.Context, func(balancer.SubConnState)) func() {
	if acbw.ccb.cc.dopts.disableHealthCheck {
		return noOpRegisterHealthListenerFn
	}
	regHealthLisFn := internal.RegisterClientHealthCheckListener
	if regHealthLisFn == nil {
		// The health package is not imported.
		return noOpRegisterHealthListenerFn
	}
	cfg := acbw.ac.cc.healthCheckConfig()
	if cfg == nil {
		return noOpRegisterHealthListenerFn
	}
	return func(ctx context.Context, listener func(balancer.SubConnState)) func() {
		return regHealthLisFn.(healthProducerRegisterFn)(ctx, acbw, cfg.ServiceName, listener)
	}
}

// RegisterHealthListener accepts a health listener from the LB policy. It sends
// updates to the health listener as long as the SubConn's connectivity state
// doesn't change and a new health listener is not registered. To invalidate
// the currently registered health listener, acbw updates the healthData. If a
// nil listener is registered, the active health listener is dropped.
func (acbw *acBalancerWrapper) RegisterHealthListener(listener func(balancer.SubConnState)) {
	acbw.healthMu.Lock()
	defer acbw.healthMu.Unlock()
	acbw.healthData.closeHealthProducer()
	// listeners should not be registered when the connectivity state
	// isn't Ready. This may happen when the balancer registers a listener
	// after the connectivityState is updated, but before it is notified
	// of the update.
	if acbw.healthData.connectivityState != connectivity.Ready {
		return
	}
	// Replace the health data to stop sending updates to any previously
	// registered health listeners.
	hd := newHealthData(connectivity.Ready)
	acbw.healthData = hd
	if listener == nil {
		return
	}

	registerFn := acbw.healthListenerRegFn()
	acbw.ccb.serializer.TrySchedule(func(ctx context.Context) {
		if ctx.Err() != nil || acbw.ccb.balancer == nil {
			return
		}
		// Don't send updates if a new listener is registered.
		acbw.healthMu.Lock()
		defer acbw.healthMu.Unlock()
		if acbw.healthData != hd {
			return
		}
		// Serialize the health updates from the health producer with
		// other calls into the LB policy.
		listenerWrapper := func(scs balancer.SubConnState) {
			acbw.ccb.serializer.TrySchedule(func(ctx context.Context) {
				if ctx.Err() != nil || acbw.ccb.balancer == nil {
					return
				}
				acbw.healthMu.Lock()
				defer acbw.healthMu.Unlock()
				if acbw.healthData != hd {
					return
				}
				listener(scs)
			})
		}

		hd.closeHealthProducer = registerFn(ctx, listenerWrapper)
	})
}
