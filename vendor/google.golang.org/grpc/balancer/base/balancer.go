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

package base

import (
	"context"
	"errors"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
)

type baseBuilder struct {
	name            string
	pickerBuilder   PickerBuilder
	v2PickerBuilder V2PickerBuilder
	config          Config
}

func (bb *baseBuilder) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	bal := &baseBalancer{
		cc:              cc,
		pickerBuilder:   bb.pickerBuilder,
		v2PickerBuilder: bb.v2PickerBuilder,

		subConns: make(map[resolver.Address]balancer.SubConn),
		scStates: make(map[balancer.SubConn]connectivity.State),
		csEvltr:  &balancer.ConnectivityStateEvaluator{},
		config:   bb.config,
	}
	// Initialize picker to a picker that always returns
	// ErrNoSubConnAvailable, because when state of a SubConn changes, we
	// may call UpdateState with this picker.
	if bb.pickerBuilder != nil {
		bal.picker = NewErrPicker(balancer.ErrNoSubConnAvailable)
	} else {
		bal.v2Picker = NewErrPickerV2(balancer.ErrNoSubConnAvailable)
	}
	return bal
}

func (bb *baseBuilder) Name() string {
	return bb.name
}

var _ balancer.V2Balancer = (*baseBalancer)(nil) // Assert that we implement V2Balancer

type baseBalancer struct {
	cc              balancer.ClientConn
	pickerBuilder   PickerBuilder
	v2PickerBuilder V2PickerBuilder

	csEvltr *balancer.ConnectivityStateEvaluator
	state   connectivity.State

	subConns map[resolver.Address]balancer.SubConn
	scStates map[balancer.SubConn]connectivity.State
	picker   balancer.Picker
	v2Picker balancer.V2Picker
	config   Config
}

func (b *baseBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	panic("not implemented")
}

func (b *baseBalancer) ResolverError(err error) {
	switch b.state {
	case connectivity.TransientFailure, connectivity.Idle, connectivity.Connecting:
		if b.picker != nil {
			b.picker = NewErrPicker(err)
		} else {
			b.v2Picker = NewErrPickerV2(err)
		}
	}
}

func (b *baseBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	// TODO: handle s.ResolverState.Err (log if not nil) once implemented.
	// TODO: handle s.ResolverState.ServiceConfig?
	if grpclog.V(2) {
		grpclog.Infoln("base.baseBalancer: got new ClientConn state: ", s)
	}
	// addrsSet is the set converted from addrs, it's used for quick lookup of an address.
	addrsSet := make(map[resolver.Address]struct{})
	for _, a := range s.ResolverState.Addresses {
		addrsSet[a] = struct{}{}
		if _, ok := b.subConns[a]; !ok {
			// a is a new address (not existing in b.subConns).
			sc, err := b.cc.NewSubConn([]resolver.Address{a}, balancer.NewSubConnOptions{HealthCheckEnabled: b.config.HealthCheck})
			if err != nil {
				grpclog.Warningf("base.baseBalancer: failed to create new SubConn: %v", err)
				continue
			}
			b.subConns[a] = sc
			b.scStates[sc] = connectivity.Idle
			sc.Connect()
		}
	}
	for a, sc := range b.subConns {
		// a was removed by resolver.
		if _, ok := addrsSet[a]; !ok {
			b.cc.RemoveSubConn(sc)
			delete(b.subConns, a)
			// Keep the state of this sc in b.scStates until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
		}
	}
	return nil
}

// regeneratePicker takes a snapshot of the balancer, and generates a picker
// from it. The picker is
//  - errPicker with ErrTransientFailure if the balancer is in TransientFailure,
//  - built by the pickerBuilder with all READY SubConns otherwise.
func (b *baseBalancer) regeneratePicker(err error) {
	if b.state == connectivity.TransientFailure {
		if b.pickerBuilder != nil {
			b.picker = NewErrPicker(balancer.ErrTransientFailure)
		} else {
			if err != nil {
				b.v2Picker = NewErrPickerV2(balancer.TransientFailureError(err))
			} else {
				// This means the last subchannel transition was not to
				// TransientFailure (otherwise err must be set), but the
				// aggregate state of the balancer is TransientFailure, meaning
				// there are no other addresses.
				b.v2Picker = NewErrPickerV2(balancer.TransientFailureError(errors.New("resolver returned no addresses")))
			}
		}
		return
	}
	if b.pickerBuilder != nil {
		readySCs := make(map[resolver.Address]balancer.SubConn)

		// Filter out all ready SCs from full subConn map.
		for addr, sc := range b.subConns {
			if st, ok := b.scStates[sc]; ok && st == connectivity.Ready {
				readySCs[addr] = sc
			}
		}
		b.picker = b.pickerBuilder.Build(readySCs)
	} else {
		readySCs := make(map[balancer.SubConn]SubConnInfo)

		// Filter out all ready SCs from full subConn map.
		for addr, sc := range b.subConns {
			if st, ok := b.scStates[sc]; ok && st == connectivity.Ready {
				readySCs[sc] = SubConnInfo{Address: addr}
			}
		}
		b.v2Picker = b.v2PickerBuilder.Build(PickerBuildInfo{ReadySCs: readySCs})
	}
}

func (b *baseBalancer) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	panic("not implemented")
}

func (b *baseBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	s := state.ConnectivityState
	if grpclog.V(2) {
		grpclog.Infof("base.baseBalancer: handle SubConn state change: %p, %v", sc, s)
	}
	oldS, ok := b.scStates[sc]
	if !ok {
		if grpclog.V(2) {
			grpclog.Infof("base.baseBalancer: got state changes for an unknown SubConn: %p, %v", sc, s)
		}
		return
	}
	b.scStates[sc] = s
	switch s {
	case connectivity.Idle:
		sc.Connect()
	case connectivity.Shutdown:
		// When an address was removed by resolver, b called RemoveSubConn but
		// kept the sc's state in scStates. Remove state for this sc here.
		delete(b.scStates, sc)
	}

	oldAggrState := b.state
	b.state = b.csEvltr.RecordTransition(oldS, s)

	// Regenerate picker when one of the following happens:
	//  - this sc became ready from not-ready
	//  - this sc became not-ready from ready
	//  - the aggregated state of balancer became TransientFailure from non-TransientFailure
	//  - the aggregated state of balancer became non-TransientFailure from TransientFailure
	if (s == connectivity.Ready) != (oldS == connectivity.Ready) ||
		(b.state == connectivity.TransientFailure) != (oldAggrState == connectivity.TransientFailure) {
		b.regeneratePicker(state.ConnectionError)
	}

	if b.picker != nil {
		b.cc.UpdateBalancerState(b.state, b.picker)
	} else {
		b.cc.UpdateState(balancer.State{ConnectivityState: b.state, Picker: b.v2Picker})
	}
}

// Close is a nop because base balancer doesn't have internal state to clean up,
// and it doesn't need to call RemoveSubConn for the SubConns.
func (b *baseBalancer) Close() {
}

// NewErrPicker returns a picker that always returns err on Pick().
func NewErrPicker(err error) balancer.Picker {
	return &errPicker{err: err}
}

type errPicker struct {
	err error // Pick() always returns this err.
}

func (p *errPicker) Pick(context.Context, balancer.PickInfo) (balancer.SubConn, func(balancer.DoneInfo), error) {
	return nil, nil, p.err
}

// NewErrPickerV2 returns a V2Picker that always returns err on Pick().
func NewErrPickerV2(err error) balancer.V2Picker {
	return &errPickerV2{err: err}
}

type errPickerV2 struct {
	err error // Pick() always returns this err.
}

func (p *errPickerV2) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	return balancer.PickResult{}, p.err
}
