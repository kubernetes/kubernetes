/*
 *
 * Copyright 2020 gRPC authors.
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

package clustermanager

import (
	"fmt"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/grpclog"
)

type subBalancerState struct {
	state balancer.State
	// stateToAggregate is the connectivity state used only for state
	// aggregation. It could be different from state.ConnectivityState. For
	// example when a sub-balancer transitions from TransientFailure to
	// connecting, state.ConnectivityState is Connecting, but stateToAggregate
	// is still TransientFailure.
	stateToAggregate connectivity.State
}

func (s *subBalancerState) String() string {
	return fmt.Sprintf("picker:%p,state:%v,stateToAggregate:%v", s.state.Picker, s.state.ConnectivityState, s.stateToAggregate)
}

type balancerStateAggregator struct {
	cc     balancer.ClientConn
	logger *grpclog.PrefixLogger
	csEval *balancer.ConnectivityStateEvaluator

	mu sync.Mutex
	// This field is used to ensure that no updates are forwarded to the parent
	// CC once the aggregator is closed. A closed sub-balancer could still send
	// pickers to this aggregator.
	closed bool
	// Map from child policy name to last reported state.
	idToPickerState map[string]*subBalancerState
	// Set when UpdateState call propagation is paused.
	pauseUpdateState bool
	// Set when UpdateState call propagation is paused and an UpdateState call
	// is suppressed.
	needUpdateStateOnResume bool
}

func newBalancerStateAggregator(cc balancer.ClientConn, logger *grpclog.PrefixLogger) *balancerStateAggregator {
	return &balancerStateAggregator{
		cc:              cc,
		logger:          logger,
		csEval:          &balancer.ConnectivityStateEvaluator{},
		idToPickerState: make(map[string]*subBalancerState),
	}
}

func (bsa *balancerStateAggregator) close() {
	bsa.mu.Lock()
	defer bsa.mu.Unlock()
	bsa.closed = true
}

// add adds a sub-balancer in CONNECTING state.
//
// This is called when there's a new child.
func (bsa *balancerStateAggregator) add(id string) {
	bsa.mu.Lock()
	defer bsa.mu.Unlock()

	bsa.idToPickerState[id] = &subBalancerState{
		// Start everything in CONNECTING, so if one of the sub-balancers
		// reports TransientFailure, the RPCs will still wait for the other
		// sub-balancers.
		state: balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
		},
		stateToAggregate: connectivity.Connecting,
	}
	bsa.csEval.RecordTransition(connectivity.Shutdown, connectivity.Connecting)
	bsa.buildAndUpdateLocked()
}

// remove removes the sub-balancer state. Future updates from this sub-balancer,
// if any, will be ignored.
//
// This is called when a child is removed.
func (bsa *balancerStateAggregator) remove(id string) {
	bsa.mu.Lock()
	defer bsa.mu.Unlock()
	if _, ok := bsa.idToPickerState[id]; !ok {
		return
	}
	// Setting the state of the deleted sub-balancer to Shutdown will get
	// csEvltr to remove the previous state for any aggregated state
	// evaluations. Transitions to and from connectivity.Shutdown are ignored
	// by csEvltr.
	bsa.csEval.RecordTransition(bsa.idToPickerState[id].stateToAggregate, connectivity.Shutdown)
	// Remove id and picker from picker map. This also results in future updates
	// for this ID to be ignored.
	delete(bsa.idToPickerState, id)
	bsa.buildAndUpdateLocked()
}

// pauseStateUpdates causes UpdateState calls to not propagate to the parent
// ClientConn.  The last state will be remembered and propagated when
// ResumeStateUpdates is called.
func (bsa *balancerStateAggregator) pauseStateUpdates() {
	bsa.mu.Lock()
	defer bsa.mu.Unlock()
	bsa.pauseUpdateState = true
	bsa.needUpdateStateOnResume = false
}

// resumeStateUpdates will resume propagating UpdateState calls to the parent,
// and call UpdateState on the parent if any UpdateState call was suppressed.
func (bsa *balancerStateAggregator) resumeStateUpdates() {
	bsa.mu.Lock()
	defer bsa.mu.Unlock()
	bsa.pauseUpdateState = false
	if bsa.needUpdateStateOnResume {
		bsa.cc.UpdateState(bsa.buildLocked())
	}
}

// UpdateState is called to report a balancer state change from sub-balancer.
// It's usually called by the balancer group.
//
// It calls parent ClientConn's UpdateState with the new aggregated state.
func (bsa *balancerStateAggregator) UpdateState(id string, state balancer.State) {
	bsa.logger.Infof("State update from sub-balancer %q: %+v", id, state)

	bsa.mu.Lock()
	defer bsa.mu.Unlock()
	pickerSt, ok := bsa.idToPickerState[id]
	if !ok {
		// All state starts with an entry in pickStateMap. If ID is not in map,
		// it's either removed, or never existed.
		return
	}
	if !(pickerSt.state.ConnectivityState == connectivity.TransientFailure && state.ConnectivityState == connectivity.Connecting) {
		// If old state is TransientFailure, and new state is Connecting, don't
		// update the state, to prevent the aggregated state from being always
		// CONNECTING. Otherwise, stateToAggregate is the same as
		// state.ConnectivityState.
		bsa.csEval.RecordTransition(pickerSt.stateToAggregate, state.ConnectivityState)
		pickerSt.stateToAggregate = state.ConnectivityState
	}
	pickerSt.state = state
	bsa.buildAndUpdateLocked()
}

// buildAndUpdateLocked combines the sub-state from each sub-balancer into one
// state, and sends a picker update to the parent ClientConn.
func (bsa *balancerStateAggregator) buildAndUpdateLocked() {
	if bsa.closed {
		return
	}
	if bsa.pauseUpdateState {
		// If updates are paused, do not call UpdateState, but remember that we
		// need to call it when they are resumed.
		bsa.needUpdateStateOnResume = true
		return
	}
	bsa.cc.UpdateState(bsa.buildLocked())
}

// buildLocked combines sub-states into one.
func (bsa *balancerStateAggregator) buildLocked() balancer.State {
	// The picker's return error might not be consistent with the
	// aggregatedState. Because for this LB policy, we want to always build
	// picker with all sub-pickers (not only ready sub-pickers), so even if the
	// overall state is Ready, pick for certain RPCs can behave like Connecting
	// or TransientFailure.
	bsa.logger.Infof("Child pickers: %+v", bsa.idToPickerState)
	return balancer.State{
		ConnectivityState: bsa.csEval.CurrentState(),
		Picker:            newPickerGroup(bsa.idToPickerState),
	}
}
