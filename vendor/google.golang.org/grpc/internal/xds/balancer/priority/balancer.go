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

// Package priority implements the priority balancer.
//
// This balancer will be kept in internal until we use it in the xds balancers,
// and are confident its functionalities are stable. It will then be exported
// for more users.
package priority

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/balancergroup"
	"google.golang.org/grpc/internal/buffer"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/internal/hierarchy"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

// Name is the name of the priority balancer.
const Name = "priority_experimental"

// DefaultSubBalancerCloseTimeout is defined as a variable instead of const for
// testing.
var DefaultSubBalancerCloseTimeout = 15 * time.Minute

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, bOpts balancer.BuildOptions) balancer.Balancer {
	b := &priorityBalancer{
		cc:                       cc,
		done:                     grpcsync.NewEvent(),
		children:                 make(map[string]*childBalancer),
		childBalancerStateUpdate: buffer.NewUnbounded(),
	}

	b.logger = prefixLogger(b)
	b.bg = balancergroup.New(balancergroup.Options{
		CC:                      cc,
		BuildOpts:               bOpts,
		StateAggregator:         b,
		Logger:                  b.logger,
		SubBalancerCloseTimeout: DefaultSubBalancerCloseTimeout,
	})
	go b.run()
	b.logger.Infof("Created")
	return b
}

func (b bb) ParseConfig(s json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return parseConfig(s)
}

func (bb) Name() string {
	return Name
}

// timerWrapper wraps a timer with a boolean. So that when a race happens
// between AfterFunc and Stop, the func is guaranteed to not execute.
type timerWrapper struct {
	stopped bool
	timer   *time.Timer
}

type priorityBalancer struct {
	logger                   *grpclog.PrefixLogger
	cc                       balancer.ClientConn
	bg                       *balancergroup.BalancerGroup
	done                     *grpcsync.Event
	childBalancerStateUpdate *buffer.Unbounded

	mu         sync.Mutex
	childInUse string
	// priorities is a list of child names from higher to lower priority.
	priorities []string
	// children is a map from child name to sub-balancers.
	children map[string]*childBalancer

	// Set during UpdateClientConnState when calling into sub-balancers.
	// Prevents child updates from recomputing the active priority or sending
	// an update of the aggregated picker to the parent.  Cleared after all
	// sub-balancers have finished UpdateClientConnState, after which
	// syncPriority is called manually.
	inhibitPickerUpdates bool
}

func (b *priorityBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received an update with balancer config: %+v", pretty.ToJSON(s.BalancerConfig))
	}
	newConfig, ok := s.BalancerConfig.(*LBConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", s.BalancerConfig)
	}
	addressesSplit := hierarchy.Group(s.ResolverState.Addresses)
	endpointsSplit := hierarchy.GroupEndpoints(s.ResolverState.Endpoints)

	b.mu.Lock()
	// Create and remove children, since we know all children from the config
	// are used by some priority.
	for name, newSubConfig := range newConfig.Children {
		bb := balancer.Get(newSubConfig.Config.Name)
		if bb == nil {
			b.logger.Errorf("balancer name %v from config is not registered", newSubConfig.Config.Name)
			continue
		}

		currentChild, ok := b.children[name]
		if !ok {
			// This is a new child, add it to the children list. But note that
			// the balancer isn't built, because this child can be a low
			// priority. If necessary, it will be built when syncing priorities.
			cb := newChildBalancer(name, b, bb.Name(), b.cc)
			cb.updateConfig(newSubConfig, resolver.State{
				Addresses:     addressesSplit[name],
				Endpoints:     endpointsSplit[name],
				ServiceConfig: s.ResolverState.ServiceConfig,
				Attributes:    s.ResolverState.Attributes,
			})
			b.children[name] = cb
			continue
		}

		// This is not a new child. But the config/addresses could change.

		// The balancing policy name is changed, close the old child. But don't
		// rebuild, rebuild will happen when syncing priorities.
		if currentChild.balancerName != bb.Name() {
			currentChild.stop()
			currentChild.updateBalancerName(bb.Name())
		}

		// Update config and address, but note that this doesn't send the
		// updates to non-started child balancers (the child balancer might not
		// be built, if it's a low priority).
		currentChild.updateConfig(newSubConfig, resolver.State{
			Addresses:     addressesSplit[name],
			Endpoints:     endpointsSplit[name],
			ServiceConfig: s.ResolverState.ServiceConfig,
			Attributes:    s.ResolverState.Attributes,
		})
	}
	// Cleanup resources used by children removed from the config.
	for name, oldChild := range b.children {
		if _, ok := newConfig.Children[name]; !ok {
			oldChild.stop()
			delete(b.children, name)
		}
	}

	// Update priorities and handle priority changes.
	b.priorities = newConfig.Priorities

	// Everything was removed by the update.
	if len(b.priorities) == 0 {
		b.childInUse = ""
		b.cc.UpdateState(balancer.State{
			ConnectivityState: connectivity.TransientFailure,
			Picker:            base.NewErrPicker(ErrAllPrioritiesRemoved),
		})
		b.mu.Unlock()
		return nil
	}

	// This will sync the states of all children to the new updated
	// priorities. Includes starting/stopping child balancers when necessary.
	// Block picker updates until all children have had a chance to call
	// UpdateState to prevent races where, e.g., the active priority reports
	// transient failure but a higher priority may have reported something that
	// made it active, and if the transient failure update is handled first,
	// RPCs could fail.
	b.inhibitPickerUpdates = true
	// Add an item to queue to notify us when the current items in the queue
	// are done and syncPriority has been called.
	done := make(chan struct{})
	b.childBalancerStateUpdate.Put(resumePickerUpdates{done: done})
	b.mu.Unlock()
	<-done

	return nil
}

func (b *priorityBalancer) ResolverError(err error) {
	if b.logger.V(2) {
		b.logger.Infof("Received error from the resolver: %v", err)
	}
	b.bg.ResolverError(err)
}

func (b *priorityBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *priorityBalancer) Close() {
	b.bg.Close()
	b.childBalancerStateUpdate.Close()

	b.mu.Lock()
	defer b.mu.Unlock()
	b.done.Fire()
	// Clear states of the current child in use, so if there's a race in picker
	// update, it will be dropped.
	b.childInUse = ""
	// Stop the child policies, this is necessary to stop the init timers in the
	// children.
	for _, child := range b.children {
		child.stop()
	}
}

func (b *priorityBalancer) ExitIdle() {
	b.bg.ExitIdle()
}

// UpdateState implements balancergroup.BalancerStateAggregator interface. The
// balancer group sends new connectivity state and picker here.
func (b *priorityBalancer) UpdateState(childName string, state balancer.State) {
	b.childBalancerStateUpdate.Put(childBalancerState{
		name: childName,
		s:    state,
	})
}

type childBalancerState struct {
	name string
	s    balancer.State
}

type resumePickerUpdates struct {
	done chan struct{}
}

// run handles child update in a separate goroutine, so if the child sends
// updates inline (when called by parent), it won't cause deadlocks (by trying
// to hold the same mutex).
func (b *priorityBalancer) run() {
	for {
		select {
		case u, ok := <-b.childBalancerStateUpdate.Get():
			if !ok {
				return
			}
			b.childBalancerStateUpdate.Load()
			// Needs to handle state update in a goroutine, because each state
			// update needs to start/close child policy, could result in
			// deadlock.
			b.mu.Lock()
			if b.done.HasFired() {
				b.mu.Unlock()
				return
			}
			switch s := u.(type) {
			case childBalancerState:
				b.handleChildStateUpdate(s.name, s.s)
			case resumePickerUpdates:
				b.inhibitPickerUpdates = false
				b.syncPriority(b.childInUse)
				close(s.done)
			}
			b.mu.Unlock()
		case <-b.done.Done():
			return
		}
	}
}
