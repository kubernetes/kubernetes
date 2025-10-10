/*
 * Copyright 2019 gRPC authors.
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
 */

// Package balancergroup implements a utility struct to bind multiple balancers
// into one balancer.
package balancergroup

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/balancer/gracefulswitch"
	"google.golang.org/grpc/internal/cache"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

// subBalancerWrapper is used to keep the configurations that will be used to start
// the underlying balancer. It can be called to start/stop the underlying
// balancer.
//
// When the config changes, it will pass the update to the underlying balancer
// if it exists.
//
// TODO: move to a separate file?
type subBalancerWrapper struct {
	// subBalancerWrapper is passed to the sub-balancer as a ClientConn
	// wrapper, only to keep the state and picker.  When sub-balancer is
	// restarted while in cache, the picker needs to be resent.
	//
	// It also contains the sub-balancer ID, so the parent balancer group can
	// keep track of SubConn/pickers and the sub-balancers they belong to. Some
	// of the actions are forwarded to the parent ClientConn with no change.
	// Some are forward to balancer group with the sub-balancer ID.
	balancer.ClientConn
	id    string
	group *BalancerGroup

	mu    sync.Mutex
	state balancer.State

	// The static part of sub-balancer. Keeps balancerBuilders and addresses.
	// To be used when restarting sub-balancer.
	builder balancer.Builder
	// Options to be passed to sub-balancer at the time of creation.
	buildOpts balancer.BuildOptions
	// ccState is a cache of the addresses/balancer config, so when the balancer
	// is restarted after close, it will get the previous update. It's a pointer
	// and is set to nil at init, so when the balancer is built for the first
	// time (not a restart), it won't receive an empty update. Note that this
	// isn't reset to nil when the underlying balancer is closed.
	ccState *balancer.ClientConnState
	// The dynamic part of sub-balancer. Only used when balancer group is
	// started. Gets cleared when sub-balancer is closed.
	balancer *gracefulswitch.Balancer
}

// UpdateState overrides balancer.ClientConn, to keep state and picker.
func (sbc *subBalancerWrapper) UpdateState(state balancer.State) {
	sbc.mu.Lock()
	sbc.state = state
	sbc.group.updateBalancerState(sbc.id, state)
	sbc.mu.Unlock()
}

// NewSubConn overrides balancer.ClientConn, so balancer group can keep track of
// the relation between subconns and sub-balancers.
func (sbc *subBalancerWrapper) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	return sbc.group.newSubConn(sbc, addrs, opts)
}

func (sbc *subBalancerWrapper) updateBalancerStateWithCachedPicker() {
	sbc.mu.Lock()
	if sbc.state.Picker != nil {
		sbc.group.updateBalancerState(sbc.id, sbc.state)
	}
	sbc.mu.Unlock()
}

func (sbc *subBalancerWrapper) startBalancer() {
	if sbc.balancer == nil {
		sbc.balancer = gracefulswitch.NewBalancer(sbc, sbc.buildOpts)
	}
	sbc.group.logger.Infof("Creating child policy of type %q for child %q", sbc.builder.Name(), sbc.id)
	sbc.balancer.SwitchTo(sbc.builder)
	if sbc.ccState != nil {
		sbc.balancer.UpdateClientConnState(*sbc.ccState)
	}
}

// exitIdle invokes the ExitIdle method on the sub-balancer, a gracefulswitch
// balancer.
func (sbc *subBalancerWrapper) exitIdle() {
	b := sbc.balancer
	if b == nil {
		return
	}
	b.ExitIdle()
}

func (sbc *subBalancerWrapper) updateClientConnState(s balancer.ClientConnState) error {
	sbc.ccState = &s
	b := sbc.balancer
	if b == nil {
		// A sub-balancer is closed when it is removed from the group or the
		// group is closed as a whole, and is not expected to receive updates
		// after that. But when used with the priority LB policy a sub-balancer
		// (and the whole balancer group) could be closed because it's the lower
		// priority, but it can still get address updates.
		return nil
	}
	return b.UpdateClientConnState(s)
}

func (sbc *subBalancerWrapper) resolverError(err error) {
	b := sbc.balancer
	if b == nil {
		// A sub-balancer is closed when it is removed from the group or the
		// group is closed as a whole, and is not expected to receive updates
		// after that. But when used with the priority LB policy a sub-balancer
		// (and the whole balancer group) could be closed because it's the lower
		// priority, but it can still get address updates.
		return
	}
	b.ResolverError(err)
}

func (sbc *subBalancerWrapper) stopBalancer() {
	if sbc.balancer == nil {
		return
	}
	sbc.balancer.Close()
	sbc.balancer = nil
}

// BalancerGroup takes a list of balancers, each behind a gracefulswitch
// balancer, and make them into one balancer.
//
// Note that this struct doesn't implement balancer.Balancer, because it's not
// intended to be used directly as a balancer. It's expected to be used as a
// sub-balancer manager by a high level balancer.
//
//	Updates from ClientConn are forwarded to sub-balancers
//	- service config update
//	- address update
//	- subConn state change
//	  - find the corresponding balancer and forward
//
//	Actions from sub-balances are forwarded to parent ClientConn
//	- new/remove SubConn
//	- picker update and health states change
//	  - sub-pickers are sent to an aggregator provided by the parent, which
//	    will group them into a group-picker. The aggregated connectivity state is
//	    also handled by the aggregator.
//	- resolveNow
//
// Sub-balancers are only built when the balancer group is started. If the
// balancer group is closed, the sub-balancers are also closed. And it's
// guaranteed that no updates will be sent to parent ClientConn from a closed
// balancer group.
type BalancerGroup struct {
	cc        balancer.ClientConn
	buildOpts balancer.BuildOptions
	logger    *grpclog.PrefixLogger

	// stateAggregator is where the state/picker updates will be sent to. It's
	// provided by the parent balancer, to build a picker with all the
	// sub-pickers.
	stateAggregator BalancerStateAggregator

	// outgoingMu guards all operations in the direction:
	// ClientConn-->Sub-balancer. Including start, stop, resolver updates and
	// SubConn state changes.
	//
	// The corresponding boolean outgoingStarted is used to stop further updates
	// to sub-balancers after they are closed.
	outgoingMu         sync.Mutex
	outgoingClosed     bool
	idToBalancerConfig map[string]*subBalancerWrapper
	// Cache for sub-balancers when they are removed. This is `nil` if caching
	// is disabled by passing `0` for Options.SubBalancerCloseTimeout`.
	deletedBalancerCache *cache.TimeoutCache

	// incomingMu is to make sure this balancer group doesn't send updates to cc
	// after it's closed.
	//
	// We don't share the mutex to avoid deadlocks (e.g. a call to sub-balancer
	// may call back to balancer group inline. It causes deadlock if they
	// require the same mutex).
	//
	// We should never need to hold multiple locks at the same time in this
	// struct. The case where two locks are held can only happen when the
	// underlying balancer calls back into balancer group inline. So there's an
	// implicit lock acquisition order that outgoingMu is locked before
	// incomingMu.

	// incomingMu guards all operations in the direction:
	// Sub-balancer-->ClientConn. Including NewSubConn, RemoveSubConn. It also
	// guards the map from SubConn to balancer ID, so updateSubConnState needs
	// to hold it shortly to potentially delete from the map.
	//
	// UpdateState is called by the balancer state aggregator, and it will
	// decide when and whether to call.
	//
	// The corresponding boolean incomingStarted is used to stop further updates
	// from sub-balancers after they are closed.
	incomingMu      sync.Mutex
	incomingClosed  bool // This boolean only guards calls back to ClientConn.
	scToSubBalancer map[balancer.SubConn]*subBalancerWrapper
}

// Options wraps the arguments to be passed to the BalancerGroup ctor.
type Options struct {
	// CC is a reference to the parent balancer.ClientConn.
	CC balancer.ClientConn
	// BuildOpts contains build options to be used when creating sub-balancers.
	BuildOpts balancer.BuildOptions
	// StateAggregator is an implementation of the BalancerStateAggregator
	// interface to aggregate picker and connectivity states from sub-balancers.
	StateAggregator BalancerStateAggregator
	// Logger is a group specific prefix logger.
	Logger *grpclog.PrefixLogger
	// SubBalancerCloseTimeout is the amount of time deleted sub-balancers spend
	// in the idle cache. A value of zero here disables caching of deleted
	// sub-balancers.
	SubBalancerCloseTimeout time.Duration
}

// New creates a new BalancerGroup. Note that the BalancerGroup
// needs to be started to work.
func New(opts Options) *BalancerGroup {
	var bc *cache.TimeoutCache
	if opts.SubBalancerCloseTimeout != time.Duration(0) {
		bc = cache.NewTimeoutCache(opts.SubBalancerCloseTimeout)
	}

	return &BalancerGroup{
		cc:              opts.CC,
		buildOpts:       opts.BuildOpts,
		stateAggregator: opts.StateAggregator,
		logger:          opts.Logger,

		deletedBalancerCache: bc,
		idToBalancerConfig:   make(map[string]*subBalancerWrapper),
		scToSubBalancer:      make(map[balancer.SubConn]*subBalancerWrapper),
	}
}

// AddWithClientConn adds a balancer with the given id to the group. The
// balancer is built with a balancer builder registered with balancerName. The
// given ClientConn is passed to the newly built balancer instead of the
// one passed to balancergroup.New().
//
// TODO: Get rid of the existing Add() API and replace it with this.
func (bg *BalancerGroup) AddWithClientConn(id, balancerName string, cc balancer.ClientConn) error {
	bg.logger.Infof("Adding child policy of type %q for child %q", balancerName, id)
	builder := balancer.Get(balancerName)
	if builder == nil {
		return fmt.Errorf("balancergroup: unregistered balancer name %q", balancerName)
	}

	// Store data in static map, and then check to see if bg is started.
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return fmt.Errorf("balancergroup: already closed")
	}
	var sbc *subBalancerWrapper
	// Skip searching the cache if disabled.
	if bg.deletedBalancerCache != nil {
		if old, ok := bg.deletedBalancerCache.Remove(id); ok {
			if bg.logger.V(2) {
				bg.logger.Infof("Removing and reusing child policy of type %q for child %q from the balancer cache", balancerName, id)
				bg.logger.Infof("Number of items remaining in the balancer cache: %d", bg.deletedBalancerCache.Len())
			}

			sbc, _ = old.(*subBalancerWrapper)
			if sbc != nil && sbc.builder != builder {
				// If the sub-balancer in cache was built with a different
				// balancer builder, don't use it, cleanup this old-balancer,
				// and behave as sub-balancer is not found in cache.
				//
				// NOTE that this will also drop the cached addresses for this
				// sub-balancer, which seems to be reasonable.
				sbc.stopBalancer()
				// cleanupSubConns must be done before the new balancer starts,
				// otherwise new SubConns created by the new balancer might be
				// removed by mistake.
				bg.cleanupSubConns(sbc)
				sbc = nil
			}
		}
	}
	if sbc == nil {
		sbc = &subBalancerWrapper{
			ClientConn: cc,
			id:         id,
			group:      bg,
			builder:    builder,
			buildOpts:  bg.buildOpts,
		}
		sbc.startBalancer()
	} else {
		// When brining back a sub-balancer from cache, re-send the cached
		// picker and state.
		sbc.updateBalancerStateWithCachedPicker()
	}
	bg.idToBalancerConfig[id] = sbc
	return nil
}

// Add adds a balancer built by builder to the group, with given id.
func (bg *BalancerGroup) Add(id string, builder balancer.Builder) {
	bg.AddWithClientConn(id, builder.Name(), bg.cc)
}

// Remove removes the balancer with id from the group.
//
// But doesn't close the balancer. The balancer is kept in a cache, and will be
// closed after timeout. Cleanup work (closing sub-balancer and removing
// subconns) will be done after timeout.
func (bg *BalancerGroup) Remove(id string) {
	bg.logger.Infof("Removing child policy for child %q", id)

	bg.outgoingMu.Lock()
	if bg.outgoingClosed {
		bg.outgoingMu.Unlock()
		return
	}

	sbToRemove, ok := bg.idToBalancerConfig[id]
	if !ok {
		bg.logger.Errorf("Child policy for child %q does not exist in the balancer group", id)
		bg.outgoingMu.Unlock()
		return
	}

	// Unconditionally remove the sub-balancer config from the map.
	delete(bg.idToBalancerConfig, id)

	if bg.deletedBalancerCache != nil {
		if bg.logger.V(2) {
			bg.logger.Infof("Adding child policy for child %q to the balancer cache", id)
			bg.logger.Infof("Number of items remaining in the balancer cache: %d", bg.deletedBalancerCache.Len())
		}

		bg.deletedBalancerCache.Add(id, sbToRemove, func() {
			if bg.logger.V(2) {
				bg.logger.Infof("Removing child policy for child %q from the balancer cache after timeout", id)
				bg.logger.Infof("Number of items remaining in the balancer cache: %d", bg.deletedBalancerCache.Len())
			}

			// A sub-balancer evicted from the timeout cache needs to closed
			// and its subConns need to removed, unconditionally. There is a
			// possibility that a sub-balancer might be removed (thereby
			// moving it to the cache) around the same time that the
			// balancergroup is closed, and by the time we get here the
			// balancergroup might be closed.  Check for `outgoingStarted ==
			// true` at that point can lead to a leaked sub-balancer.
			bg.outgoingMu.Lock()
			sbToRemove.stopBalancer()
			bg.outgoingMu.Unlock()
			bg.cleanupSubConns(sbToRemove)
		})
		bg.outgoingMu.Unlock()
		return
	}

	// Remove the sub-balancer with immediate effect if we are not caching.
	sbToRemove.stopBalancer()
	bg.outgoingMu.Unlock()
	bg.cleanupSubConns(sbToRemove)
}

// bg.remove(id) doesn't do cleanup for the sub-balancer. This function does
// cleanup after the timeout.
func (bg *BalancerGroup) cleanupSubConns(config *subBalancerWrapper) {
	bg.incomingMu.Lock()
	defer bg.incomingMu.Unlock()
	// Remove SubConns. This is only done after the balancer is
	// actually closed.
	//
	// NOTE: if NewSubConn is called by this (closed) balancer later, the
	// SubConn will be leaked. This shouldn't happen if the balancer
	// implementation is correct. To make sure this never happens, we need to
	// add another layer (balancer manager) between balancer group and the
	// sub-balancers.
	for sc, b := range bg.scToSubBalancer {
		if b == config {
			delete(bg.scToSubBalancer, sc)
		}
	}
}

// Following are actions from the parent grpc.ClientConn, forward to sub-balancers.

// updateSubConnState forwards the update to cb and updates scToSubBalancer if
// needed.
func (bg *BalancerGroup) updateSubConnState(sc balancer.SubConn, state balancer.SubConnState, cb func(balancer.SubConnState)) {
	bg.incomingMu.Lock()
	if bg.incomingClosed {
		bg.incomingMu.Unlock()
		return
	}
	if _, ok := bg.scToSubBalancer[sc]; !ok {
		bg.incomingMu.Unlock()
		return
	}
	if state.ConnectivityState == connectivity.Shutdown {
		// Only delete sc from the map when state changed to Shutdown.
		delete(bg.scToSubBalancer, sc)
	}
	bg.incomingMu.Unlock()

	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return
	}
	if cb != nil {
		cb(state)
	}
}

// UpdateSubConnState handles the state for the subconn. It finds the
// corresponding balancer and forwards the update.
func (bg *BalancerGroup) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	bg.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

// UpdateClientConnState handles ClientState (including balancer config and
// addresses) from resolver. It finds the balancer and forwards the update.
func (bg *BalancerGroup) UpdateClientConnState(id string, s balancer.ClientConnState) error {
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return nil
	}
	if config, ok := bg.idToBalancerConfig[id]; ok {
		return config.updateClientConnState(s)
	}
	return nil
}

// ResolverError forwards resolver errors to all sub-balancers.
func (bg *BalancerGroup) ResolverError(err error) {
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return
	}
	for _, config := range bg.idToBalancerConfig {
		config.resolverError(err)
	}
}

// Following are actions from sub-balancers, forward to ClientConn.

// newSubConn: forward to ClientConn, and also create a map from sc to balancer,
// so state update will find the right balancer.
//
// One note about removing SubConn: only forward to ClientConn, but not delete
// from map. Delete sc from the map only when state changes to Shutdown. Since
// it's just forwarding the action, there's no need for a removeSubConn()
// wrapper function.
func (bg *BalancerGroup) newSubConn(config *subBalancerWrapper, addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	// NOTE: if balancer with id was already removed, this should also return
	// error. But since we call balancer.stopBalancer when removing the balancer, this
	// shouldn't happen.
	bg.incomingMu.Lock()
	if bg.incomingClosed {
		bg.incomingMu.Unlock()
		return nil, fmt.Errorf("balancergroup: NewSubConn is called after balancer group is closed")
	}
	var sc balancer.SubConn
	oldListener := opts.StateListener
	opts.StateListener = func(state balancer.SubConnState) { bg.updateSubConnState(sc, state, oldListener) }
	sc, err := bg.cc.NewSubConn(addrs, opts)
	if err != nil {
		bg.incomingMu.Unlock()
		return nil, err
	}
	bg.scToSubBalancer[sc] = config
	bg.incomingMu.Unlock()
	return sc, nil
}

// updateBalancerState: forward the new state to balancer state aggregator. The
// aggregator will create an aggregated picker and an aggregated connectivity
// state, then forward to ClientConn.
func (bg *BalancerGroup) updateBalancerState(id string, state balancer.State) {
	bg.logger.Infof("Balancer state update from child %v, new state: %+v", id, state)

	// Send new state to the aggregator, without holding the incomingMu.
	// incomingMu is to protect all calls to the parent ClientConn, this update
	// doesn't necessary trigger a call to ClientConn, and should already be
	// protected by aggregator's mutex if necessary.
	if bg.stateAggregator != nil {
		bg.stateAggregator.UpdateState(id, state)
	}
}

// Close closes the balancer. It stops sub-balancers, and removes the subconns.
// When a BalancerGroup is closed, it can not receive further address updates.
func (bg *BalancerGroup) Close() {
	bg.incomingMu.Lock()
	bg.incomingClosed = true
	// Also remove all SubConns.
	for sc := range bg.scToSubBalancer {
		sc.Shutdown()
		delete(bg.scToSubBalancer, sc)
	}
	bg.incomingMu.Unlock()

	bg.outgoingMu.Lock()
	// Setting `outgoingClosed` ensures that no entries are added to
	// `deletedBalancerCache` after this point.
	bg.outgoingClosed = true
	bg.outgoingMu.Unlock()

	// Clear(true) runs clear function to close sub-balancers in cache. It
	// must be called out of outgoing mutex.
	if bg.deletedBalancerCache != nil {
		bg.deletedBalancerCache.Clear(true)
	}

	bg.outgoingMu.Lock()
	for id, config := range bg.idToBalancerConfig {
		config.stopBalancer()
		delete(bg.idToBalancerConfig, id)
	}
	bg.outgoingMu.Unlock()
}

// ExitIdle should be invoked when the parent LB policy's ExitIdle is invoked.
// It will trigger this on all sub-balancers, or reconnect their subconns if
// not supported.
func (bg *BalancerGroup) ExitIdle() {
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return
	}
	for _, config := range bg.idToBalancerConfig {
		config.exitIdle()
	}
}

// ExitIdleOne instructs the sub-balancer `id` to exit IDLE state, if
// appropriate and possible.
func (bg *BalancerGroup) ExitIdleOne(id string) {
	bg.outgoingMu.Lock()
	defer bg.outgoingMu.Unlock()
	if bg.outgoingClosed {
		return
	}
	if config := bg.idToBalancerConfig[id]; config != nil {
		config.exitIdle()
	}
}

// ParseConfig parses a child config list and returns a LB config for the
// gracefulswitch Balancer.
//
// cfg is expected to be a json.RawMessage containing a JSON array of LB policy
// names + configs as the format of the "loadBalancingConfig" field in
// ServiceConfig.  It returns a type that should be passed to
// UpdateClientConnState in the BalancerConfig field.
func ParseConfig(cfg json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return gracefulswitch.ParseConfig(cfg)
}
