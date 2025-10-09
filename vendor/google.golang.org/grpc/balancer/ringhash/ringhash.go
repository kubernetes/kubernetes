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

// Package ringhash implements the ringhash balancer. See the following
// gRFCs for details:
// - https://github.com/grpc/proposal/blob/master/A42-xds-ring-hash-lb-policy.md
// - https://github.com/grpc/proposal/blob/master/A61-IPv4-IPv6-dualstack-backends.md#ring-hash
// - https://github.com/grpc/proposal/blob/master/A76-ring-hash-improvements.md
//
// # Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
package ringhash

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/balancer/endpointsharding"
	"google.golang.org/grpc/balancer/lazy"
	"google.golang.org/grpc/balancer/pickfirst/pickfirstleaf"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/balancer/weight"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/pretty"
	iringhash "google.golang.org/grpc/internal/ringhash"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/ringhash"
	"google.golang.org/grpc/serviceconfig"
)

// Name is the name of the ring_hash balancer.
const Name = "ring_hash_experimental"

func lazyPickFirstBuilder(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	return lazy.NewBalancer(cc, opts, balancer.Get(pickfirstleaf.Name).Build)
}

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	b := &ringhashBalancer{
		ClientConn:     cc,
		endpointStates: resolver.NewEndpointMap[*endpointState](),
	}
	esOpts := endpointsharding.Options{DisableAutoReconnect: true}
	b.child = endpointsharding.NewBalancer(b, opts, lazyPickFirstBuilder, esOpts)
	b.logger = prefixLogger(b)
	b.logger.Infof("Created")
	return b
}

func (bb) Name() string {
	return Name
}

func (bb) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return parseConfig(c)
}

type ringhashBalancer struct {
	// The following fields are initialized at build time and read-only after
	// that and therefore do not need to be guarded by a mutex.

	// ClientConn is embedded to intercept UpdateState calls from the child
	// endpointsharding balancer.
	balancer.ClientConn
	logger *grpclog.PrefixLogger
	child  balancer.Balancer

	mu                   sync.Mutex
	config               *iringhash.LBConfig
	inhibitChildUpdates  bool
	shouldRegenerateRing bool
	endpointStates       *resolver.EndpointMap[*endpointState]

	// ring is always in sync with endpoints. When endpoints change, a new ring
	// is generated. Note that address weights updates also regenerates the
	// ring.
	ring *ring
}

// hashKey returns the hash key to use for an endpoint. Per gRFC A61, each entry
// in the ring is a hash of the endpoint's hash key concatenated with a
// per-entry unique suffix.
func hashKey(endpoint resolver.Endpoint) string {
	if hk := ringhash.HashKey(endpoint); hk != "" {
		return hk
	}
	// If no hash key is set, use the endpoint's first address as the hash key.
	// This is the default behavior when no hash key is set.
	return endpoint.Addresses[0].Addr
}

// UpdateState intercepts child balancer state updates. It updates the
// per-endpoint state stored in the ring, and also the aggregated state based on
// the child picker. It also reconciles the endpoint list. It sets
// `b.shouldRegenerateRing` to true if the new endpoint list is different from
// the previous, i.e. any of the following is true:
// - an endpoint was added
// - an endpoint was removed
// - an endpoint's weight was updated
// - the first addresses of the endpoint has changed
func (b *ringhashBalancer) UpdateState(state balancer.State) {
	b.mu.Lock()
	defer b.mu.Unlock()
	childStates := endpointsharding.ChildStatesFromPicker(state.Picker)
	// endpointsSet is the set converted from endpoints, used for quick lookup.
	endpointsSet := resolver.NewEndpointMap[bool]()

	for _, childState := range childStates {
		endpoint := childState.Endpoint
		endpointsSet.Set(endpoint, true)
		newWeight := getWeightAttribute(endpoint)
		hk := hashKey(endpoint)
		es, ok := b.endpointStates.Get(endpoint)
		if !ok {
			es := &endpointState{
				balancer: childState.Balancer,
				hashKey:  hk,
				weight:   newWeight,
				state:    childState.State,
			}
			b.endpointStates.Set(endpoint, es)
			b.shouldRegenerateRing = true
		} else {
			// We have seen this endpoint before and created a `endpointState`
			// object for it. If the weight or the hash key of the endpoint has
			// changed, update the endpoint state map with the new weight or
			// hash key. This will be used when a new ring is created.
			if oldWeight := es.weight; oldWeight != newWeight {
				b.shouldRegenerateRing = true
				es.weight = newWeight
			}
			if es.hashKey != hk {
				b.shouldRegenerateRing = true
				es.hashKey = hk
			}
			es.state = childState.State
		}
	}

	for _, endpoint := range b.endpointStates.Keys() {
		if _, ok := endpointsSet.Get(endpoint); ok {
			continue
		}
		// endpoint was removed by resolver.
		b.endpointStates.Delete(endpoint)
		b.shouldRegenerateRing = true
	}

	b.updatePickerLocked()
}

func (b *ringhashBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received update from resolver, balancer config: %+v", pretty.ToJSON(ccs.BalancerConfig))
	}

	newConfig, ok := ccs.BalancerConfig.(*iringhash.LBConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", ccs.BalancerConfig)
	}

	b.mu.Lock()
	b.inhibitChildUpdates = true
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		b.inhibitChildUpdates = false
		b.updatePickerLocked()
		b.mu.Unlock()
	}()

	if err := b.child.UpdateClientConnState(balancer.ClientConnState{
		// Make pickfirst children use health listeners for outlier detection
		// and health checking to work.
		ResolverState: pickfirstleaf.EnableHealthListener(ccs.ResolverState),
	}); err != nil {
		return err
	}

	b.mu.Lock()
	// Ring updates can happen due to the following:
	// 1. Addition or deletion of endpoints: The synchronous picker update from
	//    the child endpointsharding balancer would contain the list of updated
	//    endpoints.  Updates triggered by the child after handling the
	//    `UpdateClientConnState` call will not change the endpoint list.
	// 2. Change in the `LoadBalancerConfig`: Ring config such as max/min ring
	//    size.
	// To avoid extra ring updates, a boolean is used to track the need for a
	// ring update and the update is done only once at the end.
	//
	// If the ring configuration has changed, we need to regenerate the ring
	// while sending a new picker.
	if b.config == nil || b.config.MinRingSize != newConfig.MinRingSize || b.config.MaxRingSize != newConfig.MaxRingSize {
		b.shouldRegenerateRing = true
	}
	b.config = newConfig
	b.mu.Unlock()
	return nil
}

func (b *ringhashBalancer) ResolverError(err error) {
	b.child.ResolverError(err)
}

func (b *ringhashBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *ringhashBalancer) updatePickerLocked() {
	state := b.aggregatedStateLocked()
	// Start connecting to new endpoints if necessary.
	if state == connectivity.Connecting || state == connectivity.TransientFailure {
		// When overall state is TransientFailure, we need to make sure at least
		// one endpoint is attempting to connect, otherwise this balancer may
		// never get picks if the parent is priority.
		//
		// Because we report Connecting as the overall state when only one
		// endpoint is in TransientFailure, we do the same check for Connecting
		// here.
		//
		// Note that this check also covers deleting endpoints. E.g. if the
		// endpoint attempting to connect is deleted, and the overall state is
		// TF. Since there must be at least one endpoint attempting to connect,
		// we need to trigger one.
		//
		// After calling `ExitIdle` on a child balancer, the child will send a
		// picker update asynchronously. A race condition may occur if another
		// picker update from endpointsharding arrives before the child's
		// picker update. The received picker may trigger a re-execution of the
		// loop below to find an idle child. Since map iteration order is
		// non-deterministic, the list of `endpointState`s must be sorted to
		// ensure `ExitIdle` is called on the same child, preventing unnecessary
		// connections.
		var endpointStates = make([]*endpointState, b.endpointStates.Len())
		for i, s := range b.endpointStates.Values() {
			endpointStates[i] = s
		}
		sort.Slice(endpointStates, func(i, j int) bool {
			return endpointStates[i].hashKey < endpointStates[j].hashKey
		})
		var idleBalancer endpointsharding.ExitIdler
		for _, es := range endpointStates {
			connState := es.state.ConnectivityState
			if connState == connectivity.Connecting {
				idleBalancer = nil
				break
			}
			if idleBalancer == nil && connState == connectivity.Idle {
				idleBalancer = es.balancer
			}
		}
		if idleBalancer != nil {
			idleBalancer.ExitIdle()
		}
	}

	if b.inhibitChildUpdates {
		return
	}

	// Update the channel.
	if b.endpointStates.Len() > 0 && b.shouldRegenerateRing {
		// with a non-empty list of endpoints.
		b.ring = newRing(b.endpointStates, b.config.MinRingSize, b.config.MaxRingSize, b.logger)
	}
	b.shouldRegenerateRing = false
	var newPicker balancer.Picker
	if b.endpointStates.Len() == 0 {
		newPicker = base.NewErrPicker(errors.New("produced zero addresses"))
	} else {
		newPicker = b.newPickerLocked()
	}
	b.ClientConn.UpdateState(balancer.State{
		ConnectivityState: state,
		Picker:            newPicker,
	})
}

func (b *ringhashBalancer) Close() {
	b.logger.Infof("Shutdown")
	b.child.Close()
}

func (b *ringhashBalancer) ExitIdle() {
	// ExitIdle implementation is a no-op because connections are either
	// triggers from picks or from child balancer state changes.
}

// newPickerLocked generates a picker. The picker copies the endpoint states
// over to avoid locking the mutex at RPC time. The picker should be
// re-generated every time an endpoint state is updated.
func (b *ringhashBalancer) newPickerLocked() *picker {
	states := make(map[string]endpointState)
	hasEndpointConnecting := false
	for _, epState := range b.endpointStates.Values() {
		// Copy the endpoint state to avoid races, since ring hash
		// mutates the state, weight and hash key in place.
		states[epState.hashKey] = *epState
		if epState.state.ConnectivityState == connectivity.Connecting {
			hasEndpointConnecting = true
		}
	}
	return &picker{
		ring:                         b.ring,
		endpointStates:               states,
		requestHashHeader:            b.config.RequestHashHeader,
		hasEndpointInConnectingState: hasEndpointConnecting,
		randUint64:                   rand.Uint64,
	}
}

// aggregatedStateLocked returns the aggregated child balancers state
// based on the following rules.
//   - If there is at least one endpoint in READY state, report READY.
//   - If there are 2 or more endpoints in TRANSIENT_FAILURE state, report
//     TRANSIENT_FAILURE.
//   - If there is at least one endpoint in CONNECTING state, report CONNECTING.
//   - If there is one endpoint in TRANSIENT_FAILURE and there is more than one
//     endpoint, report state CONNECTING.
//   - If there is at least one endpoint in Idle state, report Idle.
//   - Otherwise, report TRANSIENT_FAILURE.
//
// Note that if there are 1 connecting, 2 transient failure, the overall state
// is transient failure. This is because the second transient failure is a
// fallback of the first failing endpoint, and we want to report transient
// failure to failover to the lower priority.
func (b *ringhashBalancer) aggregatedStateLocked() connectivity.State {
	var nums [5]int
	for _, es := range b.endpointStates.Values() {
		nums[es.state.ConnectivityState]++
	}

	if nums[connectivity.Ready] > 0 {
		return connectivity.Ready
	}
	if nums[connectivity.TransientFailure] > 1 {
		return connectivity.TransientFailure
	}
	if nums[connectivity.Connecting] > 0 {
		return connectivity.Connecting
	}
	if nums[connectivity.TransientFailure] == 1 && b.endpointStates.Len() > 1 {
		return connectivity.Connecting
	}
	if nums[connectivity.Idle] > 0 {
		return connectivity.Idle
	}
	return connectivity.TransientFailure
}

// getWeightAttribute is a convenience function which returns the value of the
// weight endpoint Attribute.
//
// When used in the xDS context, the weight attribute is guaranteed to be
// non-zero. But, when used in a non-xDS context, the weight attribute could be
// unset. A Default of 1 is used in the latter case.
func getWeightAttribute(e resolver.Endpoint) uint32 {
	w := weight.FromEndpoint(e).Weight
	if w == 0 {
		return 1
	}
	return w
}

type endpointState struct {
	// hashKey is the hash key of the endpoint. Per gRFC A61, each entry in the
	// ring is an endpoint, positioned based on the hash of the endpoint's first
	// address by default. Per gRFC A76, the hash key of an endpoint may be
	// overridden, for example based on EDS endpoint metadata.
	hashKey  string
	weight   uint32
	balancer endpointsharding.ExitIdler

	// state is updated by the balancer while receiving resolver updates from
	// the channel and picker updates from its children. Access to it is guarded
	// by ringhashBalancer.mu.
	state balancer.State
}
