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
	"errors"
	"fmt"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
)

// PickFirstBalancerName is the name of the pick_first balancer.
const PickFirstBalancerName = "pick_first"

func newPickfirstBuilder() balancer.Builder {
	return &pickfirstBuilder{}
}

type pickfirstBuilder struct{}

func (*pickfirstBuilder) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	return &pickfirstBalancer{cc: cc}
}

func (*pickfirstBuilder) Name() string {
	return PickFirstBalancerName
}

type pickfirstBalancer struct {
	state   connectivity.State
	cc      balancer.ClientConn
	subConn balancer.SubConn
}

func (b *pickfirstBalancer) ResolverError(err error) {
	if logger.V(2) {
		logger.Infof("pickfirstBalancer: ResolverError called with error %v", err)
	}
	if b.subConn == nil {
		b.state = connectivity.TransientFailure
	}

	if b.state != connectivity.TransientFailure {
		// The picker will not change since the balancer does not currently
		// report an error.
		return
	}
	b.cc.UpdateState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            &picker{err: fmt.Errorf("name resolver error: %v", err)},
	})
}

func (b *pickfirstBalancer) UpdateClientConnState(state balancer.ClientConnState) error {
	if len(state.ResolverState.Addresses) == 0 {
		// The resolver reported an empty address list. Treat it like an error by
		// calling b.ResolverError.
		if b.subConn != nil {
			// Remove the old subConn. All addresses were removed, so it is no longer
			// valid.
			b.cc.RemoveSubConn(b.subConn)
			b.subConn = nil
		}
		b.ResolverError(errors.New("produced zero addresses"))
		return balancer.ErrBadResolverState
	}

	if b.subConn != nil {
		b.cc.UpdateAddresses(b.subConn, state.ResolverState.Addresses)
		return nil
	}

	subConn, err := b.cc.NewSubConn(state.ResolverState.Addresses, balancer.NewSubConnOptions{})
	if err != nil {
		if logger.V(2) {
			logger.Errorf("pickfirstBalancer: failed to NewSubConn: %v", err)
		}
		b.state = connectivity.TransientFailure
		b.cc.UpdateState(balancer.State{
			ConnectivityState: connectivity.TransientFailure,
			Picker:            &picker{err: fmt.Errorf("error creating connection: %v", err)},
		})
		return balancer.ErrBadResolverState
	}
	b.subConn = subConn
	b.state = connectivity.Idle
	b.cc.UpdateState(balancer.State{
		ConnectivityState: connectivity.Connecting,
		Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
	})
	b.subConn.Connect()
	return nil
}

func (b *pickfirstBalancer) UpdateSubConnState(subConn balancer.SubConn, state balancer.SubConnState) {
	if logger.V(2) {
		logger.Infof("pickfirstBalancer: UpdateSubConnState: %p, %v", subConn, state)
	}
	if b.subConn != subConn {
		if logger.V(2) {
			logger.Infof("pickfirstBalancer: ignored state change because subConn is not recognized")
		}
		return
	}
	b.state = state.ConnectivityState
	if state.ConnectivityState == connectivity.Shutdown {
		b.subConn = nil
		return
	}

	switch state.ConnectivityState {
	case connectivity.Ready:
		b.cc.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            &picker{result: balancer.PickResult{SubConn: subConn}},
		})
	case connectivity.Connecting:
		b.cc.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
	case connectivity.Idle:
		b.cc.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            &idlePicker{subConn: subConn},
		})
	case connectivity.TransientFailure:
		b.cc.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            &picker{err: state.ConnectionError},
		})
	}
}

func (b *pickfirstBalancer) Close() {
}

func (b *pickfirstBalancer) ExitIdle() {
	if b.subConn != nil && b.state == connectivity.Idle {
		b.subConn.Connect()
	}
}

type picker struct {
	result balancer.PickResult
	err    error
}

func (p *picker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	return p.result, p.err
}

// idlePicker is used when the SubConn is IDLE and kicks the SubConn into
// CONNECTING when Pick is called.
type idlePicker struct {
	subConn balancer.SubConn
}

func (i *idlePicker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	i.subConn.Connect()
	return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
}

func init() {
	balancer.Register(newPickfirstBuilder())
}
