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
	"strings"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/balancer/gracefulswitch"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/resolver"
)

type ccbMode int

const (
	ccbModeActive = iota
	ccbModeIdle
	ccbModeClosed
	ccbModeExitingIdle
)

// ccBalancerWrapper sits between the ClientConn and the Balancer.
//
// ccBalancerWrapper implements methods corresponding to the ones on the
// balancer.Balancer interface. The ClientConn is free to call these methods
// concurrently and the ccBalancerWrapper ensures that calls from the ClientConn
// to the Balancer happen synchronously and in order.
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
	cc   *ClientConn
	opts balancer.BuildOptions

	// Outgoing (gRPC --> balancer) calls are guaranteed to execute in a
	// mutually exclusive manner as they are scheduled in the serializer. Fields
	// accessed *only* in these serializer callbacks, can therefore be accessed
	// without a mutex.
	balancer        *gracefulswitch.Balancer
	curBalancerName string

	// mu guards access to the below fields. Access to the serializer and its
	// cancel function needs to be mutex protected because they are overwritten
	// when the wrapper exits idle mode.
	mu               sync.Mutex
	serializer       *grpcsync.CallbackSerializer // To serialize all outoing calls.
	serializerCancel context.CancelFunc           // To close the seralizer at close/enterIdle time.
	mode             ccbMode                      // Tracks the current mode of the wrapper.
}

// newCCBalancerWrapper creates a new balancer wrapper. The underlying balancer
// is not created until the switchTo() method is invoked.
func newCCBalancerWrapper(cc *ClientConn, bopts balancer.BuildOptions) *ccBalancerWrapper {
	ctx, cancel := context.WithCancel(context.Background())
	ccb := &ccBalancerWrapper{
		cc:               cc,
		opts:             bopts,
		serializer:       grpcsync.NewCallbackSerializer(ctx),
		serializerCancel: cancel,
	}
	ccb.balancer = gracefulswitch.NewBalancer(ccb, bopts)
	return ccb
}

// updateClientConnState is invoked by grpc to push a ClientConnState update to
// the underlying balancer.
func (ccb *ccBalancerWrapper) updateClientConnState(ccs *balancer.ClientConnState) error {
	ccb.mu.Lock()
	errCh := make(chan error, 1)
	// Here and everywhere else where Schedule() is called, it is done with the
	// lock held. But the lock guards only the scheduling part. The actual
	// callback is called asynchronously without the lock being held.
	ok := ccb.serializer.Schedule(func(_ context.Context) {
		errCh <- ccb.balancer.UpdateClientConnState(*ccs)
	})
	if !ok {
		// If we are unable to schedule a function with the serializer, it
		// indicates that it has been closed. A serializer is only closed when
		// the wrapper is closed or is in idle.
		ccb.mu.Unlock()
		return fmt.Errorf("grpc: cannot send state update to a closed or idle balancer")
	}
	ccb.mu.Unlock()

	// We get here only if the above call to Schedule succeeds, in which case it
	// is guaranteed that the scheduled function will run. Therefore it is safe
	// to block on this channel.
	err := <-errCh
	if logger.V(2) && err != nil {
		logger.Infof("error from balancer.UpdateClientConnState: %v", err)
	}
	return err
}

// updateSubConnState is invoked by grpc to push a subConn state update to the
// underlying balancer.
func (ccb *ccBalancerWrapper) updateSubConnState(sc balancer.SubConn, s connectivity.State, err error) {
	ccb.mu.Lock()
	ccb.serializer.Schedule(func(_ context.Context) {
		// Even though it is optional for balancers, gracefulswitch ensures
		// opts.StateListener is set, so this cannot ever be nil.
		sc.(*acBalancerWrapper).stateListener(balancer.SubConnState{ConnectivityState: s, ConnectionError: err})
	})
	ccb.mu.Unlock()
}

func (ccb *ccBalancerWrapper) resolverError(err error) {
	ccb.mu.Lock()
	ccb.serializer.Schedule(func(_ context.Context) {
		ccb.balancer.ResolverError(err)
	})
	ccb.mu.Unlock()
}

// switchTo is invoked by grpc to instruct the balancer wrapper to switch to the
// LB policy identified by name.
//
// ClientConn calls newCCBalancerWrapper() at creation time. Upon receipt of the
// first good update from the name resolver, it determines the LB policy to use
// and invokes the switchTo() method. Upon receipt of every subsequent update
// from the name resolver, it invokes this method.
//
// the ccBalancerWrapper keeps track of the current LB policy name, and skips
// the graceful balancer switching process if the name does not change.
func (ccb *ccBalancerWrapper) switchTo(name string) {
	ccb.mu.Lock()
	ccb.serializer.Schedule(func(_ context.Context) {
		// TODO: Other languages use case-sensitive balancer registries. We should
		// switch as well. See: https://github.com/grpc/grpc-go/issues/5288.
		if strings.EqualFold(ccb.curBalancerName, name) {
			return
		}
		ccb.buildLoadBalancingPolicy(name)
	})
	ccb.mu.Unlock()
}

// buildLoadBalancingPolicy performs the following:
//   - retrieve a balancer builder for the given name. Use the default LB
//     policy, pick_first, if no LB policy with name is found in the registry.
//   - instruct the gracefulswitch balancer to switch to the above builder. This
//     will actually build the new balancer.
//   - update the `curBalancerName` field
//
// Must be called from a serializer callback.
func (ccb *ccBalancerWrapper) buildLoadBalancingPolicy(name string) {
	builder := balancer.Get(name)
	if builder == nil {
		channelz.Warningf(logger, ccb.cc.channelzID, "Channel switches to new LB policy %q, since the specified LB policy %q was not registered", PickFirstBalancerName, name)
		builder = newPickfirstBuilder()
	} else {
		channelz.Infof(logger, ccb.cc.channelzID, "Channel switches to new LB policy %q", name)
	}

	if err := ccb.balancer.SwitchTo(builder); err != nil {
		channelz.Errorf(logger, ccb.cc.channelzID, "Channel failed to build new LB policy %q: %v", name, err)
		return
	}
	ccb.curBalancerName = builder.Name()
}

func (ccb *ccBalancerWrapper) close() {
	channelz.Info(logger, ccb.cc.channelzID, "ccBalancerWrapper: closing")
	ccb.closeBalancer(ccbModeClosed)
}

// enterIdleMode is invoked by grpc when the channel enters idle mode upon
// expiry of idle_timeout. This call blocks until the balancer is closed.
func (ccb *ccBalancerWrapper) enterIdleMode() {
	channelz.Info(logger, ccb.cc.channelzID, "ccBalancerWrapper: entering idle mode")
	ccb.closeBalancer(ccbModeIdle)
}

// closeBalancer is invoked when the channel is being closed or when it enters
// idle mode upon expiry of idle_timeout.
func (ccb *ccBalancerWrapper) closeBalancer(m ccbMode) {
	ccb.mu.Lock()
	if ccb.mode == ccbModeClosed || ccb.mode == ccbModeIdle {
		ccb.mu.Unlock()
		return
	}

	ccb.mode = m
	done := ccb.serializer.Done()
	b := ccb.balancer
	ok := ccb.serializer.Schedule(func(_ context.Context) {
		// Close the serializer to ensure that no more calls from gRPC are sent
		// to the balancer.
		ccb.serializerCancel()
		// Empty the current balancer name because we don't have a balancer
		// anymore and also so that we act on the next call to switchTo by
		// creating a new balancer specified by the new resolver.
		ccb.curBalancerName = ""
	})
	if !ok {
		ccb.mu.Unlock()
		return
	}
	ccb.mu.Unlock()

	// Give enqueued callbacks a chance to finish before closing the balancer.
	<-done
	b.Close()
}

// exitIdleMode is invoked by grpc when the channel exits idle mode either
// because of an RPC or because of an invocation of the Connect() API. This
// recreates the balancer that was closed previously when entering idle mode.
//
// If the channel is not in idle mode, we know for a fact that we are here as a
// result of the user calling the Connect() method on the ClientConn. In this
// case, we can simply forward the call to the underlying balancer, instructing
// it to reconnect to the backends.
func (ccb *ccBalancerWrapper) exitIdleMode() {
	ccb.mu.Lock()
	if ccb.mode == ccbModeClosed {
		// Request to exit idle is a no-op when wrapper is already closed.
		ccb.mu.Unlock()
		return
	}

	if ccb.mode == ccbModeIdle {
		// Recreate the serializer which was closed when we entered idle.
		ctx, cancel := context.WithCancel(context.Background())
		ccb.serializer = grpcsync.NewCallbackSerializer(ctx)
		ccb.serializerCancel = cancel
	}

	// The ClientConn guarantees that mutual exclusion between close() and
	// exitIdleMode(), and since we just created a new serializer, we can be
	// sure that the below function will be scheduled.
	done := make(chan struct{})
	ccb.serializer.Schedule(func(_ context.Context) {
		defer close(done)

		ccb.mu.Lock()
		defer ccb.mu.Unlock()

		if ccb.mode != ccbModeIdle {
			ccb.balancer.ExitIdle()
			return
		}

		// Gracefulswitch balancer does not support a switchTo operation after
		// being closed. Hence we need to create a new one here.
		ccb.balancer = gracefulswitch.NewBalancer(ccb, ccb.opts)
		ccb.mode = ccbModeActive
		channelz.Info(logger, ccb.cc.channelzID, "ccBalancerWrapper: exiting idle mode")

	})
	ccb.mu.Unlock()

	<-done
}

func (ccb *ccBalancerWrapper) isIdleOrClosed() bool {
	ccb.mu.Lock()
	defer ccb.mu.Unlock()
	return ccb.mode == ccbModeIdle || ccb.mode == ccbModeClosed
}

func (ccb *ccBalancerWrapper) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	if ccb.isIdleOrClosed() {
		return nil, fmt.Errorf("grpc: cannot create SubConn when balancer is closed or idle")
	}

	if len(addrs) == 0 {
		return nil, fmt.Errorf("grpc: cannot create SubConn with empty address list")
	}
	ac, err := ccb.cc.newAddrConn(addrs, opts)
	if err != nil {
		channelz.Warningf(logger, ccb.cc.channelzID, "acBalancerWrapper: NewSubConn: failed to newAddrConn: %v", err)
		return nil, err
	}
	acbw := &acBalancerWrapper{
		ccb:           ccb,
		ac:            ac,
		producers:     make(map[balancer.ProducerBuilder]*refCountedProducer),
		stateListener: opts.StateListener,
	}
	ac.acbw = acbw
	return acbw, nil
}

func (ccb *ccBalancerWrapper) RemoveSubConn(sc balancer.SubConn) {
	// The graceful switch balancer will never call this.
	logger.Errorf("ccb RemoveSubConn(%v) called unexpectedly, sc")
}

func (ccb *ccBalancerWrapper) UpdateAddresses(sc balancer.SubConn, addrs []resolver.Address) {
	if ccb.isIdleOrClosed() {
		return
	}

	acbw, ok := sc.(*acBalancerWrapper)
	if !ok {
		return
	}
	acbw.UpdateAddresses(addrs)
}

func (ccb *ccBalancerWrapper) UpdateState(s balancer.State) {
	if ccb.isIdleOrClosed() {
		return
	}

	// Update picker before updating state.  Even though the ordering here does
	// not matter, it can lead to multiple calls of Pick in the common start-up
	// case where we wait for ready and then perform an RPC.  If the picker is
	// updated later, we could call the "connecting" picker when the state is
	// updated, and then call the "ready" picker after the picker gets updated.
	ccb.cc.blockingpicker.updatePicker(s.Picker)
	ccb.cc.csMgr.updateState(s.ConnectivityState)
}

func (ccb *ccBalancerWrapper) ResolveNow(o resolver.ResolveNowOptions) {
	if ccb.isIdleOrClosed() {
		return
	}

	ccb.cc.resolveNow(o)
}

func (ccb *ccBalancerWrapper) Target() string {
	return ccb.cc.target
}

// acBalancerWrapper is a wrapper on top of ac for balancers.
// It implements balancer.SubConn interface.
type acBalancerWrapper struct {
	ac            *addrConn          // read-only
	ccb           *ccBalancerWrapper // read-only
	stateListener func(balancer.SubConnState)

	mu        sync.Mutex
	producers map[balancer.ProducerBuilder]*refCountedProducer
}

func (acbw *acBalancerWrapper) String() string {
	return fmt.Sprintf("SubConn(id:%d)", acbw.ac.channelzID.Int())
}

func (acbw *acBalancerWrapper) UpdateAddresses(addrs []resolver.Address) {
	acbw.ac.updateAddrs(addrs)
}

func (acbw *acBalancerWrapper) Connect() {
	go acbw.ac.connect()
}

func (acbw *acBalancerWrapper) Shutdown() {
	ccb := acbw.ccb
	if ccb.isIdleOrClosed() {
		// It it safe to ignore this call when the balancer is closed or in idle
		// because the ClientConn takes care of closing the connections.
		//
		// Not returning early from here when the balancer is closed or in idle
		// leads to a deadlock though, because of the following sequence of
		// calls when holding cc.mu:
		// cc.exitIdleMode --> ccb.enterIdleMode --> gsw.Close -->
		// ccb.RemoveAddrConn --> cc.removeAddrConn
		return
	}

	ccb.cc.removeAddrConn(acbw.ac, errConnDrain)
}

// NewStream begins a streaming RPC on the addrConn.  If the addrConn is not
// ready, blocks until it is or ctx expires.  Returns an error when the context
// expires or the addrConn is shut down.
func (acbw *acBalancerWrapper) NewStream(ctx context.Context, desc *StreamDesc, method string, opts ...CallOption) (ClientStream, error) {
	transport, err := acbw.ac.getTransport(ctx)
	if err != nil {
		return nil, err
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
	acbw.mu.Lock()
	defer acbw.mu.Unlock()

	// Look up existing producer from this builder.
	pData := acbw.producers[pb]
	if pData == nil {
		// Not found; create a new one and add it to the producers map.
		p, close := pb.Build(acbw)
		pData = &refCountedProducer{producer: p, close: close}
		acbw.producers[pb] = pData
	}
	// Account for this new reference.
	pData.refs++

	// Return a cleanup function wrapped in a OnceFunc to remove this reference
	// and delete the refCountedProducer from the map if the total reference
	// count goes to zero.
	unref := func() {
		acbw.mu.Lock()
		pData.refs--
		if pData.refs == 0 {
			defer pData.close() // Run outside the acbw mutex
			delete(acbw.producers, pb)
		}
		acbw.mu.Unlock()
	}
	return pData.producer, grpcsync.OnceFunc(unref)
}
