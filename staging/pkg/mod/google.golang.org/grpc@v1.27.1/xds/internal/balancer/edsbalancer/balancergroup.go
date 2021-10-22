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

package edsbalancer

import (
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/cache"
	"google.golang.org/grpc/internal/wrr"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/xds/internal"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
	orcapb "google.golang.org/grpc/xds/internal/proto/udpa/data/orca/v1"
)

// subBalancerWithConfig is used to keep the configurations that will be used to start
// the underlying balancer. It can be called to start/stop the underlying
// balancer.
//
// When the config changes, it will pass the update to the underlying balancer
// if it exists.
//
// TODO: rename to subBalanceWrapper (and move to a separate file?)
type subBalancerWithConfig struct {
	// subBalancerWithConfig is passed to the sub-balancer as a ClientConn
	// wrapper, only to keep the state and picker.  When sub-balancer is
	// restarted while in cache, the picker needs to be resent.
	//
	// It also contains the sub-balancer ID, so the parent balancer group can
	// keep track of SubConn/pickers and the sub-balancers they belong to. Some
	// of the actions are forwarded to the parent ClientConn with no change.
	// Some are forward to balancer group with the sub-balancer ID.
	balancer.ClientConn
	id    internal.Locality
	group *balancerGroup

	mu    sync.Mutex
	state balancer.State

	// The static part of sub-balancer. Keeps balancerBuilders and addresses.
	// To be used when restarting sub-balancer.
	builder balancer.Builder
	addrs   []resolver.Address
	// The dynamic part of sub-balancer. Only used when balancer group is
	// started. Gets cleared when sub-balancer is closed.
	balancer balancer.Balancer
}

func (sbc *subBalancerWithConfig) UpdateBalancerState(state connectivity.State, picker balancer.Picker) {
	grpclog.Fatalln("not implemented")
}

// UpdateState overrides balancer.ClientConn, to keep state and picker.
func (sbc *subBalancerWithConfig) UpdateState(state balancer.State) {
	sbc.mu.Lock()
	sbc.state = state
	sbc.group.updateBalancerState(sbc.id, state)
	sbc.mu.Unlock()
}

// NewSubConn overrides balancer.ClientConn, so balancer group can keep track of
// the relation between subconns and sub-balancers.
func (sbc *subBalancerWithConfig) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	return sbc.group.newSubConn(sbc, addrs, opts)
}

func (sbc *subBalancerWithConfig) updateBalancerStateWithCachedPicker() {
	sbc.mu.Lock()
	if sbc.state.Picker != nil {
		sbc.group.updateBalancerState(sbc.id, sbc.state)
	}
	sbc.mu.Unlock()
}

func (sbc *subBalancerWithConfig) startBalancer() {
	b := sbc.builder.Build(sbc, balancer.BuildOptions{})
	sbc.balancer = b
	if ub, ok := b.(balancer.V2Balancer); ok {
		ub.UpdateClientConnState(balancer.ClientConnState{ResolverState: resolver.State{Addresses: sbc.addrs}})
	} else {
		b.HandleResolvedAddrs(sbc.addrs, nil)
	}
}

func (sbc *subBalancerWithConfig) handleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	b := sbc.balancer
	if b == nil {
		// This sub-balancer was closed. This can happen when EDS removes a
		// locality. The balancer for this locality was already closed, and the
		// SubConns are being deleted. But SubConn state change can still
		// happen.
		return
	}
	if ub, ok := b.(balancer.V2Balancer); ok {
		ub.UpdateSubConnState(sc, balancer.SubConnState{ConnectivityState: state})
	} else {
		b.HandleSubConnStateChange(sc, state)
	}
}

func (sbc *subBalancerWithConfig) updateAddrs(addrs []resolver.Address) {
	sbc.addrs = addrs
	b := sbc.balancer
	if b == nil {
		// This sub-balancer was closed. This should never happen because
		// sub-balancers are closed when the locality is removed from EDS, or
		// the balancer group is closed. There should be no further address
		// updates when either of this happened.
		//
		// This will be a common case with priority support, because a
		// sub-balancer (and the whole balancer group) could be closed because
		// it's the lower priority, but it can still get address updates.
		return
	}
	if ub, ok := b.(balancer.V2Balancer); ok {
		ub.UpdateClientConnState(balancer.ClientConnState{ResolverState: resolver.State{Addresses: addrs}})
	} else {
		b.HandleResolvedAddrs(addrs, nil)
	}
}

func (sbc *subBalancerWithConfig) stopBalancer() {
	sbc.balancer.Close()
	sbc.balancer = nil
}

type pickerState struct {
	weight uint32
	picker balancer.V2Picker
	state  connectivity.State
}

// balancerGroup takes a list of balancers, and make then into one balancer.
//
// Note that this struct doesn't implement balancer.Balancer, because it's not
// intended to be used directly as a balancer. It's expected to be used as a
// sub-balancer manager by a high level balancer.
//
// Updates from ClientConn are forwarded to sub-balancers
//  - service config update
//     - Not implemented
//  - address update
//  - subConn state change
//     - find the corresponding balancer and forward
//
// Actions from sub-balances are forwarded to parent ClientConn
//  - new/remove SubConn
//  - picker update and health states change
//     - sub-pickers are grouped into a group-picker
//     - aggregated connectivity state is the overall state of all pickers.
//  - resolveNow
//
// Sub-balancers are only built when the balancer group is started. If the
// balancer group is closed, the sub-balancers are also closed. And it's
// guaranteed that no updates will be sent to parent ClientConn from a closed
// balancer group.
type balancerGroup struct {
	cc        balancer.ClientConn
	loadStore lrs.Store

	// outgoingMu guards all operations in the direction:
	// ClientConn-->Sub-balancer. Including start, stop, resolver updates and
	// SubConn state changes.
	//
	// The corresponding boolean outgoingStarted is used to stop further updates
	// to sub-balancers after they are closed.
	outgoingMu         sync.Mutex
	outgoingStarted    bool
	idToBalancerConfig map[internal.Locality]*subBalancerWithConfig
	// Cache for sub-balancers when they are removed.
	balancerCache *cache.TimeoutCache

	// incomingMu and pickerMu are to make sure this balancer group doesn't send
	// updates to cc after it's closed.
	//
	// We don't share the mutex to avoid deadlocks (e.g. a call to sub-balancer
	// may call back to balancer group inline. It causes deaclock if they
	// require the same mutex).
	//
	// We should never need to hold multiple locks at the same time in this
	// struct. The case where two locks are held can only happen when the
	// underlying balancer calls back into balancer group inline. So there's an
	// implicit lock acquisition order that outgoingMu is locked before either
	// incomingMu or pickerMu.

	// incomingMu guards all operations in the direction:
	// Sub-balancer-->ClientConn. Including NewSubConn, RemoveSubConn, and
	// updatePicker. It also guards the map from SubConn to balancer ID, so
	// handleSubConnStateChange needs to hold it shortly to find the
	// sub-balancer to forward the update.
	//
	// The corresponding boolean incomingStarted is used to stop further updates
	// from sub-balancers after they are closed.
	incomingMu      sync.Mutex
	incomingStarted bool // This boolean only guards calls back to ClientConn.
	scToSubBalancer map[balancer.SubConn]*subBalancerWithConfig
	// All balancer IDs exist as keys in this map, even if balancer group is not
	// started.
	//
	// If an ID is not in map, it's either removed or never added.
	idToPickerState map[internal.Locality]*pickerState
}

// defaultSubBalancerCloseTimeout is defined as a variable instead of const for
// testing.
//
// TODO: make it a parameter for newBalancerGroup().
var defaultSubBalancerCloseTimeout = 15 * time.Minute

func newBalancerGroup(cc balancer.ClientConn, loadStore lrs.Store) *balancerGroup {
	return &balancerGroup{
		cc:        cc,
		loadStore: loadStore,

		idToBalancerConfig: make(map[internal.Locality]*subBalancerWithConfig),
		balancerCache:      cache.NewTimeoutCache(defaultSubBalancerCloseTimeout),
		scToSubBalancer:    make(map[balancer.SubConn]*subBalancerWithConfig),
		idToPickerState:    make(map[internal.Locality]*pickerState),
	}
}

func (bg *balancerGroup) start() {
	bg.incomingMu.Lock()
	bg.incomingStarted = true
	bg.incomingMu.Unlock()

	bg.outgoingMu.Lock()
	if bg.outgoingStarted {
		bg.outgoingMu.Unlock()
		return
	}

	for _, config := range bg.idToBalancerConfig {
		config.startBalancer()
	}
	bg.outgoingStarted = true
	bg.outgoingMu.Unlock()
}

// add adds a balancer built by builder to the group, with given id and weight.
//
// weight should never be zero.
func (bg *balancerGroup) add(id internal.Locality, weight uint32, builder balancer.Builder) {
	if weight == 0 {
		grpclog.Errorf("balancerGroup.add called with weight 0, locality: %v. Locality is not added to balancer group", id)
		return
	}

	// First, add things to the picker map. Do this even if incomingStarted is
	// false, because the data is static.
	bg.incomingMu.Lock()
	bg.idToPickerState[id] = &pickerState{
		weight: weight,
		// Start everything in IDLE. It's doesn't affect the overall state
		// because we don't count IDLE when aggregating (as opposite to e.g.
		// READY, 1 READY results in overall READY).
		state: connectivity.Idle,
	}
	bg.incomingMu.Unlock()

	// Store data in static map, and then check to see if bg is started.
	bg.outgoingMu.Lock()
	var sbc *subBalancerWithConfig
	// If outgoingStarted is true, search in the cache. Otherwise, cache is
	// guaranteed to be empty, searching is unnecessary.
	if bg.outgoingStarted {
		if old, ok := bg.balancerCache.Remove(id); ok {
			sbc, _ = old.(*subBalancerWithConfig)
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
		sbc = &subBalancerWithConfig{
			ClientConn: bg.cc,
			id:         id,
			group:      bg,
			builder:    builder,
		}
		if bg.outgoingStarted {
			// Only start the balancer if bg is started. Otherwise, we only keep the
			// static data.
			sbc.startBalancer()
		}
	} else {
		// When brining back a sub-balancer from cache, re-send the cached
		// picker and state.
		sbc.updateBalancerStateWithCachedPicker()
	}
	bg.idToBalancerConfig[id] = sbc
	bg.outgoingMu.Unlock()
}

// remove removes the balancer with id from the group.
//
// But doesn't close the balancer. The balancer is kept in a cache, and will be
// closed after timeout. Cleanup work (closing sub-balancer and removing
// subconns) will be done after timeout.
//
// It also removes the picker generated from this balancer from the picker
// group. It always results in a picker update.
func (bg *balancerGroup) remove(id internal.Locality) {
	bg.outgoingMu.Lock()
	if sbToRemove, ok := bg.idToBalancerConfig[id]; ok {
		if bg.outgoingStarted {
			bg.balancerCache.Add(id, sbToRemove, func() {
				// After timeout, when sub-balancer is removed from cache, need
				// to close the underlying sub-balancer, and remove all its
				// subconns.
				bg.outgoingMu.Lock()
				if bg.outgoingStarted {
					sbToRemove.stopBalancer()
				}
				bg.outgoingMu.Unlock()
				bg.cleanupSubConns(sbToRemove)
			})
		}
		delete(bg.idToBalancerConfig, id)
	} else {
		grpclog.Infof("balancer group: trying to remove a non-existing locality from balancer group: %v", id)
	}
	bg.outgoingMu.Unlock()

	bg.incomingMu.Lock()
	// Remove id and picker from picker map. This also results in future updates
	// for this ID to be ignored.
	delete(bg.idToPickerState, id)
	if bg.incomingStarted {
		// Normally picker update is triggered by SubConn state change. But we
		// want to update state and picker to reflect the changes, too. Because
		// we don't want `ClientConn` to pick this sub-balancer anymore.
		bg.cc.UpdateState(buildPickerAndState(bg.idToPickerState))
	}
	bg.incomingMu.Unlock()
}

// bg.remove(id) doesn't do cleanup for the sub-balancer. This function does
// cleanup after the timeout.
func (bg *balancerGroup) cleanupSubConns(config *subBalancerWithConfig) {
	bg.incomingMu.Lock()
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
			bg.cc.RemoveSubConn(sc)
			delete(bg.scToSubBalancer, sc)
		}
	}
	bg.incomingMu.Unlock()
}

// changeWeight changes the weight of the balancer.
//
// newWeight should never be zero.
//
// NOTE: It always results in a picker update now. This probably isn't
// necessary. But it seems better to do the update because it's a change in the
// picker (which is balancer's snapshot).
func (bg *balancerGroup) changeWeight(id internal.Locality, newWeight uint32) {
	if newWeight == 0 {
		grpclog.Errorf("balancerGroup.changeWeight called with newWeight 0. Weight is not changed")
		return
	}
	bg.incomingMu.Lock()
	defer bg.incomingMu.Unlock()
	pState, ok := bg.idToPickerState[id]
	if !ok {
		return
	}
	if pState.weight == newWeight {
		return
	}
	pState.weight = newWeight
	if bg.incomingStarted {
		// Normally picker update is triggered by SubConn state change. But we
		// want to update state and picker to reflect the changes, too. Because
		// `ClientConn` should do pick with the new weights now.
		bg.cc.UpdateState(buildPickerAndState(bg.idToPickerState))
	}
}

// Following are actions from the parent grpc.ClientConn, forward to sub-balancers.

// SubConn state change: find the corresponding balancer and then forward.
func (bg *balancerGroup) handleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	grpclog.Infof("balancer group: handle subconn state change: %p, %v", sc, state)
	bg.incomingMu.Lock()
	config, ok := bg.scToSubBalancer[sc]
	if !ok {
		bg.incomingMu.Unlock()
		return
	}
	if state == connectivity.Shutdown {
		// Only delete sc from the map when state changed to Shutdown.
		delete(bg.scToSubBalancer, sc)
	}
	bg.incomingMu.Unlock()

	bg.outgoingMu.Lock()
	config.handleSubConnStateChange(sc, state)
	bg.outgoingMu.Unlock()
}

// Address change: forward to balancer.
func (bg *balancerGroup) handleResolvedAddrs(id internal.Locality, addrs []resolver.Address) {
	bg.outgoingMu.Lock()
	if config, ok := bg.idToBalancerConfig[id]; ok {
		config.updateAddrs(addrs)
	}
	bg.outgoingMu.Unlock()
}

// TODO: handleServiceConfig()
//
// For BNS address for slicer, comes from endpoint.Metadata. It will be sent
// from parent to sub-balancers as service config.

// Following are actions from sub-balancers, forward to ClientConn.

// newSubConn: forward to ClientConn, and also create a map from sc to balancer,
// so state update will find the right balancer.
//
// One note about removing SubConn: only forward to ClientConn, but not delete
// from map. Delete sc from the map only when state changes to Shutdown. Since
// it's just forwarding the action, there's no need for a removeSubConn()
// wrapper function.
func (bg *balancerGroup) newSubConn(config *subBalancerWithConfig, addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	// NOTE: if balancer with id was already removed, this should also return
	// error. But since we call balancer.stopBalancer when removing the balancer, this
	// shouldn't happen.
	bg.incomingMu.Lock()
	if !bg.incomingStarted {
		bg.incomingMu.Unlock()
		return nil, fmt.Errorf("NewSubConn is called after balancer group is closed")
	}
	sc, err := bg.cc.NewSubConn(addrs, opts)
	if err != nil {
		bg.incomingMu.Unlock()
		return nil, err
	}
	bg.scToSubBalancer[sc] = config
	bg.incomingMu.Unlock()
	return sc, nil
}

// updateBalancerState: create an aggregated picker and an aggregated
// connectivity state, then forward to ClientConn.
func (bg *balancerGroup) updateBalancerState(id internal.Locality, state balancer.State) {
	grpclog.Infof("balancer group: update balancer state: %v, %v", id, state)

	bg.incomingMu.Lock()
	defer bg.incomingMu.Unlock()
	pickerSt, ok := bg.idToPickerState[id]
	if !ok {
		// All state starts in IDLE. If ID is not in map, it's either removed,
		// or never existed.
		grpclog.Warningf("balancer group: pickerState for %v not found when update picker/state", id)
		return
	}
	pickerSt.picker = newLoadReportPicker(state.Picker, id, bg.loadStore)
	pickerSt.state = state.ConnectivityState
	if bg.incomingStarted {
		bg.cc.UpdateState(buildPickerAndState(bg.idToPickerState))
	}
}

func (bg *balancerGroup) close() {
	bg.incomingMu.Lock()
	if bg.incomingStarted {
		bg.incomingStarted = false

		for _, pState := range bg.idToPickerState {
			// Reset everything to IDLE but keep the entry in map (to keep the
			// weight).
			pState.picker = nil
			pState.state = connectivity.Idle
		}

		// Also remove all SubConns.
		for sc := range bg.scToSubBalancer {
			bg.cc.RemoveSubConn(sc)
			delete(bg.scToSubBalancer, sc)
		}
	}
	bg.incomingMu.Unlock()

	bg.outgoingMu.Lock()
	if bg.outgoingStarted {
		bg.outgoingStarted = false
		for _, config := range bg.idToBalancerConfig {
			config.stopBalancer()
		}
	}
	bg.outgoingMu.Unlock()
	// Clear(true) runs clear function to close sub-balancers in cache. It
	// must be called out of outgoing mutex.
	bg.balancerCache.Clear(true)
}

func buildPickerAndState(m map[internal.Locality]*pickerState) balancer.State {
	var readyN, connectingN int
	readyPickerWithWeights := make([]pickerState, 0, len(m))
	for _, ps := range m {
		switch ps.state {
		case connectivity.Ready:
			readyN++
			readyPickerWithWeights = append(readyPickerWithWeights, *ps)
		case connectivity.Connecting:
			connectingN++
		}
	}
	var aggregatedState connectivity.State
	switch {
	case readyN > 0:
		aggregatedState = connectivity.Ready
	case connectingN > 0:
		aggregatedState = connectivity.Connecting
	default:
		aggregatedState = connectivity.TransientFailure
	}
	if aggregatedState == connectivity.TransientFailure {
		return balancer.State{aggregatedState, base.NewErrPickerV2(balancer.ErrTransientFailure)}
	}
	return balancer.State{aggregatedState, newPickerGroup(readyPickerWithWeights)}
}

// RandomWRR constructor, to be modified in tests.
var newRandomWRR = wrr.NewRandom

type pickerGroup struct {
	length int
	w      wrr.WRR
}

// newPickerGroup takes pickers with weights, and group them into one picker.
//
// Note it only takes ready pickers. The map shouldn't contain non-ready
// pickers.
//
// TODO: (bg) confirm this is the expected behavior: non-ready balancers should
// be ignored when picking. Only ready balancers are picked.
func newPickerGroup(readyPickerWithWeights []pickerState) *pickerGroup {
	w := newRandomWRR()
	for _, ps := range readyPickerWithWeights {
		w.Add(ps.picker, int64(ps.weight))
	}

	return &pickerGroup{
		length: len(readyPickerWithWeights),
		w:      w,
	}
}

func (pg *pickerGroup) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	if pg.length <= 0 {
		return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
	}
	p := pg.w.Next().(balancer.V2Picker)
	return p.Pick(info)
}

const (
	serverLoadCPUName    = "cpu_utilization"
	serverLoadMemoryName = "mem_utilization"
)

type loadReportPicker struct {
	p balancer.V2Picker

	id        internal.Locality
	loadStore lrs.Store
}

func newLoadReportPicker(p balancer.V2Picker, id internal.Locality, loadStore lrs.Store) *loadReportPicker {
	return &loadReportPicker{
		p:         p,
		id:        id,
		loadStore: loadStore,
	}
}

func (lrp *loadReportPicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	res, err := lrp.p.Pick(info)
	if lrp.loadStore != nil && err == nil {
		lrp.loadStore.CallStarted(lrp.id)
		td := res.Done
		res.Done = func(info balancer.DoneInfo) {
			lrp.loadStore.CallFinished(lrp.id, info.Err)
			if load, ok := info.ServerLoad.(*orcapb.OrcaLoadReport); ok {
				lrp.loadStore.CallServerLoad(lrp.id, serverLoadCPUName, load.CpuUtilization)
				lrp.loadStore.CallServerLoad(lrp.id, serverLoadMemoryName, load.MemUtilization)
				for n, d := range load.RequestCost {
					lrp.loadStore.CallServerLoad(lrp.id, n, d)
				}
				for n, d := range load.Utilization {
					lrp.loadStore.CallServerLoad(lrp.id, n, d)
				}
			}
			if td != nil {
				td(info)
			}
		}
	}
	return res, err
}
