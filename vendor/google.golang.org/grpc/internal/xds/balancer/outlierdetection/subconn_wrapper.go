/*
 *
 * Copyright 2022 gRPC authors.
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

package outlierdetection

import (
	"fmt"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/buffer"
	"google.golang.org/grpc/resolver"
)

// subConnWrapper wraps every created SubConn in the Outlier Detection Balancer,
// to help track the latest state update from the underlying SubConn, and also
// whether or not this SubConn is ejected.
type subConnWrapper struct {
	balancer.SubConn
	// endpointInfo is a pointer to the subConnWrapper's corresponding endpoint
	// map entry, if the map entry exists. It is accessed atomically.
	endpointInfo atomic.Pointer[endpointInfo]
	// The following fields are set during object creation and read-only after
	// that.

	listener func(balancer.SubConnState)

	scUpdateCh *buffer.Unbounded

	// The following fields are only referenced in the context of a work
	// serializing buffer and don't need to be protected by a mutex.

	// These two pieces of state will reach eventual consistency due to sync in
	// run(), and child will always have the correctly updated SubConnState.

	ejected bool

	// addresses is the list of address(es) this SubConn was created with to
	// help support any change in address(es)
	addresses []resolver.Address
	// latestHealthState is tracked to update the child policy during
	// unejection.
	latestHealthState balancer.SubConnState

	// Access to the following fields are protected by a mutex. These fields
	// should not be accessed from outside this file, instead use methods
	// defined on the struct.
	mu             sync.Mutex
	healthListener func(balancer.SubConnState)
	// latestReceivedConnectivityState is the SubConn's most recent connectivity
	// state received. It may not be delivered to the child balancer yet. It is
	// used to ensure a health listener is registered with the SubConn only when
	// the SubConn is READY.
	latestReceivedConnectivityState connectivity.State
}

// eject causes the wrapper to report a state update with the TRANSIENT_FAILURE
// state, and to stop passing along updates from the underlying subchannel.
func (scw *subConnWrapper) eject() {
	scw.scUpdateCh.Put(&ejectionUpdate{
		scw:       scw,
		isEjected: true,
	})
}

// uneject causes the wrapper to report a state update with the latest update
// from the underlying subchannel, and resume passing along updates from the
// underlying subchannel.
func (scw *subConnWrapper) uneject() {
	scw.scUpdateCh.Put(&ejectionUpdate{
		scw:       scw,
		isEjected: false,
	})
}

func (scw *subConnWrapper) String() string {
	return fmt.Sprintf("%+v", scw.addresses)
}

func (scw *subConnWrapper) RegisterHealthListener(listener func(balancer.SubConnState)) {
	// gRPC currently supports two mechanisms that provide a health signal for
	// a connection: client-side health checking and outlier detection. Earlier
	// both these mechanisms signaled unhealthiness by setting the subchannel
	// state to TRANSIENT_FAILURE. As part of the dualstack changes to make
	// pick_first the universal leaf policy (see A61), both these mechanisms
	// started using the new health listener to make health signal visible to
	// the petiole policies without affecting the underlying connectivity
	// management of the pick_first policy.
	scw.mu.Lock()
	defer scw.mu.Unlock()

	if scw.latestReceivedConnectivityState != connectivity.Ready {
		return
	}
	scw.healthListener = listener
	if listener == nil {
		scw.SubConn.RegisterHealthListener(nil)
		return
	}

	scw.SubConn.RegisterHealthListener(func(scs balancer.SubConnState) {
		scw.scUpdateCh.Put(&scHealthUpdate{
			scw:   scw,
			state: scs,
		})
	})
}

// updateSubConnHealthState stores the latest health state for unejection and
// sends updates the health listener.
func (scw *subConnWrapper) updateSubConnHealthState(scs balancer.SubConnState) {
	scw.latestHealthState = scs
	if scw.ejected {
		return
	}
	scw.mu.Lock()
	defer scw.mu.Unlock()
	if scw.healthListener != nil {
		scw.healthListener(scs)
	}
}

// updateSubConnConnectivityState stores the latest connectivity state for
// unejection and updates the raw connectivity listener.
func (scw *subConnWrapper) updateSubConnConnectivityState(scs balancer.SubConnState) {
	if scw.listener != nil {
		scw.listener(scs)
	}
}

func (scw *subConnWrapper) clearHealthListener() {
	scw.mu.Lock()
	defer scw.mu.Unlock()
	scw.healthListener = nil
}

func (scw *subConnWrapper) handleUnejection() {
	scw.ejected = false
	// If scw.latestHealthState has never been written to will use the health
	// state CONNECTING set during object creation.
	scw.updateSubConnHealthState(scw.latestHealthState)
}

func (scw *subConnWrapper) handleEjection() {
	scw.ejected = true
	stateToUpdate := balancer.SubConnState{
		ConnectivityState: connectivity.TransientFailure,
	}
	scw.mu.Lock()
	defer scw.mu.Unlock()
	if scw.healthListener != nil {
		scw.healthListener(stateToUpdate)
	}
}

func (scw *subConnWrapper) setLatestConnectivityState(state connectivity.State) {
	scw.mu.Lock()
	defer scw.mu.Unlock()
	scw.latestReceivedConnectivityState = state
}
