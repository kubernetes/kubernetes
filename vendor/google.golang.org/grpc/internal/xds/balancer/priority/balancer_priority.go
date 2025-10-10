/*
 *
 * Copyright 2021 gRPC authors.
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

package priority

import (
	"errors"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
)

var (
	// ErrAllPrioritiesRemoved is returned by the picker when there's no priority available.
	ErrAllPrioritiesRemoved = errors.New("no priority is provided, all priorities are removed")
	// DefaultPriorityInitTimeout is the timeout after which if a priority is
	// not READY, the next will be started. It's exported to be overridden by
	// tests.
	DefaultPriorityInitTimeout = 10 * time.Second
)

// syncPriority handles priority after a config update or a child balancer
// connectivity state update. It makes sure the balancer state (started or not)
// is in sync with the priorities (even in tricky cases where a child is moved
// from a priority to another).
//
// It's guaranteed that after this function returns:
//
//	If some child is READY, it is childInUse, and all lower priorities are
//	closed.
//
//	If some child is newly started(in Connecting for the first time), it is
//	childInUse, and all lower priorities are closed.
//
//	Otherwise, the lowest priority is childInUse (none of the children is
//	ready, and the overall state is not ready).
//
// Steps:
//
//	If all priorities were deleted, unset childInUse (to an empty string), and
//	set parent ClientConn to TransientFailure
//
//	Otherwise, Scan all children from p0, and check balancer stats:
//
//	  For any of the following cases:
//
//	    If balancer is not started (not built), this is either a new child with
//	    high priority, or a new builder for an existing child.
//
//	    If balancer is Connecting and has non-nil initTimer (meaning it
//	    transitioned from Ready or Idle to connecting, not from TF, so we
//	    should give it init-time to connect).
//
//	    If balancer is READY or IDLE
//
//	    If this is the lowest priority
//
//	 do the following:
//
//	    if this is not the old childInUse, override picker so old picker is no
//	    longer used.
//
//	    switch to it (because all higher priorities are neither new or Ready)
//
//	    forward the new addresses and config
//
// Caller must hold b.mu.
func (b *priorityBalancer) syncPriority(childUpdating string) {
	if b.inhibitPickerUpdates {
		if b.logger.V(2) {
			b.logger.Infof("Skipping update from child policy %q", childUpdating)
		}
		return
	}
	for p, name := range b.priorities {
		child, ok := b.children[name]
		if !ok {
			b.logger.Warningf("Priority name %q is not found in list of child policies", name)
			continue
		}

		if !child.started ||
			child.state.ConnectivityState == connectivity.Ready ||
			child.state.ConnectivityState == connectivity.Idle ||
			(child.state.ConnectivityState == connectivity.Connecting && child.initTimer != nil) ||
			p == len(b.priorities)-1 {
			if b.childInUse != child.name || child.name == childUpdating {
				if b.logger.V(2) {
					b.logger.Infof("childInUse, childUpdating: %q, %q", b.childInUse, child.name)
				}
				// If we switch children or the child in use just updated its
				// picker, push the child's picker to the parent.
				b.cc.UpdateState(child.state)
			}
			if b.logger.V(2) {
				b.logger.Infof("Switching to (%q, %v) in syncPriority", child.name, p)
			}
			b.switchToChild(child, p)
			break
		}
	}
}

// Stop priorities [p+1, lowest].
//
// Caller must hold b.mu.
func (b *priorityBalancer) stopSubBalancersLowerThanPriority(p int) {
	for i := p + 1; i < len(b.priorities); i++ {
		name := b.priorities[i]
		child, ok := b.children[name]
		if !ok {
			b.logger.Warningf("Priority name %q is not found in list of child policies", name)
			continue
		}
		child.stop()
	}
}

// switchToChild does the following:
// - stop all child with lower priorities
// - if childInUse is not this child
//   - set childInUse to this child
//   - if this child is not started, start it
//
// Note that it does NOT send the current child state (picker) to the parent
// ClientConn. The caller needs to send it if necessary.
//
// this can be called when
// 1. first update, start p0
// 2. an update moves a READY child from a lower priority to higher
// 2. a different builder is updated for this child
// 3. a high priority goes Failure, start next
// 4. a high priority init timeout, start next
//
// Caller must hold b.mu.
func (b *priorityBalancer) switchToChild(child *childBalancer, priority int) {
	// Stop lower priorities even if childInUse is same as this child. It's
	// possible this child was moved from a priority to another.
	b.stopSubBalancersLowerThanPriority(priority)

	// If this child is already in use, do nothing.
	//
	// This can happen:
	// - all priorities are not READY, an config update always triggers switch
	// to the lowest. In this case, the lowest child could still be connecting,
	// so we don't stop the init timer.
	// - a high priority is READY, an config update always triggers switch to
	// it.
	if b.childInUse == child.name && child.started {
		return
	}
	b.childInUse = child.name

	if !child.started {
		child.start()
	}
}

// handleChildStateUpdate start/close priorities based on the connectivity
// state.
func (b *priorityBalancer) handleChildStateUpdate(childName string, s balancer.State) {
	// Update state in child. The updated picker will be sent to parent later if
	// necessary.
	child, ok := b.children[childName]
	if !ok {
		b.logger.Warningf("Child policy not found for %q", childName)
		return
	}
	if !child.started {
		b.logger.Warningf("Ignoring update from child policy %q which is not in started state: %+v", childName, s)
		return
	}
	child.state = s

	// We start/stop the init timer of this child based on the new connectivity
	// state. syncPriority() later will need the init timer (to check if it's
	// nil or not) to decide which child to switch to.
	switch s.ConnectivityState {
	case connectivity.Ready, connectivity.Idle:
		child.reportedTF = false
		child.stopInitTimer()
	case connectivity.TransientFailure:
		child.reportedTF = true
		child.stopInitTimer()
	case connectivity.Connecting:
		if !child.reportedTF {
			child.startInitTimer()
		}
	default:
		// New state is Shutdown, should never happen. Don't forward.
	}

	child.parent.syncPriority(childName)
}
