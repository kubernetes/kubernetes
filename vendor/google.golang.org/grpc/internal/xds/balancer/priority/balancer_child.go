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
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

type childBalancer struct {
	name         string
	parent       *priorityBalancer
	parentCC     balancer.ClientConn
	balancerName string
	cc           *ignoreResolveNowClientConn

	ignoreReresolutionRequests bool
	config                     serviceconfig.LoadBalancingConfig
	rState                     resolver.State

	started bool
	// This is set when the child reports TransientFailure, and unset when it
	// reports Ready or Idle. It is used to decide whether the failover timer
	// should start when the child is transitioning into Connecting. The timer
	// will be restarted if the child has not reported TF more recently than it
	// reported Ready or Idle.
	reportedTF bool
	// The latest state the child balancer provided.
	state balancer.State
	// The timer to give a priority some time to connect. And if the priority
	// doesn't go into Ready/Failure, the next priority will be started.
	initTimer *timerWrapper
}

// newChildBalancer creates a child balancer place holder, but doesn't
// build/start the child balancer.
func newChildBalancer(name string, parent *priorityBalancer, balancerName string, cc balancer.ClientConn) *childBalancer {
	return &childBalancer{
		name:         name,
		parent:       parent,
		parentCC:     cc,
		balancerName: balancerName,
		cc:           newIgnoreResolveNowClientConn(cc, false),
		started:      false,
		// Start with the connecting state and picker with re-pick error, so
		// that when a priority switch causes this child picked before it's
		// balancing policy is created, a re-pick will happen.
		state: balancer.State{
			ConnectivityState: connectivity.Connecting,
			Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
		},
	}
}

// updateBalancerName updates balancer name for the child, but doesn't build a
// new one. The parent priority LB always closes the child policy before
// updating the balancer name, and the new balancer is built when it gets added
// to the balancergroup as part of start().
func (cb *childBalancer) updateBalancerName(balancerName string) {
	cb.balancerName = balancerName
	cb.cc = newIgnoreResolveNowClientConn(cb.parentCC, cb.ignoreReresolutionRequests)
}

// updateConfig sets childBalancer's config and state, but doesn't send update to
// the child balancer unless it is started.
func (cb *childBalancer) updateConfig(child *Child, rState resolver.State) {
	cb.ignoreReresolutionRequests = child.IgnoreReresolutionRequests
	cb.config = child.Config.Config
	cb.rState = rState
	if cb.started {
		cb.sendUpdate()
	}
}

// start builds the child balancer if it's not already started.
//
// It doesn't do it directly. It asks the balancer group to build it.
func (cb *childBalancer) start() {
	if cb.started {
		return
	}
	cb.started = true
	cb.parent.bg.AddWithClientConn(cb.name, cb.balancerName, cb.cc)
	cb.startInitTimer()
	cb.sendUpdate()
}

// sendUpdate sends the addresses and config to the child balancer.
func (cb *childBalancer) sendUpdate() {
	cb.cc.updateIgnoreResolveNow(cb.ignoreReresolutionRequests)
	// TODO: return and aggregate the returned error in the parent.
	err := cb.parent.bg.UpdateClientConnState(cb.name, balancer.ClientConnState{
		ResolverState:  cb.rState,
		BalancerConfig: cb.config,
	})
	if err != nil {
		cb.parent.logger.Warningf("Failed to update state for child policy %q: %v", cb.name, err)
	}
}

// stop stops the child balancer and resets the state.
//
// It doesn't do it directly. It asks the balancer group to remove it.
//
// Note that the underlying balancer group could keep the child in a cache.
func (cb *childBalancer) stop() {
	if !cb.started {
		return
	}
	cb.stopInitTimer()
	cb.parent.bg.Remove(cb.name)
	cb.started = false
	cb.state = balancer.State{
		ConnectivityState: connectivity.Connecting,
		Picker:            base.NewErrPicker(balancer.ErrNoSubConnAvailable),
	}
	// Clear child.reportedTF, so that if this child is started later, it will
	// be given time to connect.
	cb.reportedTF = false
}

func (cb *childBalancer) startInitTimer() {
	if cb.initTimer != nil {
		return
	}
	// Need this local variable to capture timerW in the AfterFunc closure
	// to check the stopped boolean.
	timerW := &timerWrapper{}
	cb.initTimer = timerW
	timerW.timer = time.AfterFunc(DefaultPriorityInitTimeout, func() {
		cb.parent.mu.Lock()
		defer cb.parent.mu.Unlock()
		if timerW.stopped {
			return
		}
		cb.initTimer = nil
		// Re-sync the priority. This will switch to the next priority if
		// there's any. Note that it's important sync() is called after setting
		// initTimer to nil.
		cb.parent.syncPriority("")
	})
}

func (cb *childBalancer) stopInitTimer() {
	timerW := cb.initTimer
	if timerW == nil {
		return
	}
	cb.initTimer = nil
	timerW.stopped = true
	timerW.timer.Stop()
}
