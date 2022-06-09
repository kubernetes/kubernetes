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
	state connectivity.State
	cc    balancer.ClientConn
	sc    balancer.SubConn
}

func (b *pickfirstBalancer) ResolverError(err error) {
	switch b.state {
	case connectivity.TransientFailure, connectivity.Idle, connectivity.Connecting:
		// Set a failing picker if we don't have a good picker.
		b.cc.UpdateState(balancer.State{ConnectivityState: connectivity.TransientFailure,
			Picker: &picker{err: fmt.Errorf("name resolver error: %v", err)},
		})
	}
	if logger.V(2) {
		logger.Infof("pickfirstBalancer: ResolverError called with error %v", err)
	}
}

func (b *pickfirstBalancer) UpdateClientConnState(cs balancer.ClientConnState) error {
	if len(cs.ResolverState.Addresses) == 0 {
		b.ResolverError(errors.New("produced zero addresses"))
		return balancer.ErrBadResolverState
	}
	if b.sc == nil {
		var err error
		b.sc, err = b.cc.NewSubConn(cs.ResolverState.Addresses, balancer.NewSubConnOptions{})
		if err != nil {
			if logger.V(2) {
				logger.Errorf("pickfirstBalancer: failed to NewSubConn: %v", err)
			}
			b.state = connectivity.TransientFailure
			b.cc.UpdateState(balancer.State{ConnectivityState: connectivity.TransientFailure,
				Picker: &picker{err: fmt.Errorf("error creating connection: %v", err)},
			})
			return balancer.ErrBadResolverState
		}
		b.state = connectivity.Idle
		b.cc.UpdateState(balancer.State{ConnectivityState: connectivity.Idle, Picker: &picker{result: balancer.PickResult{SubConn: b.sc}}})
		b.sc.Connect()
	} else {
		b.cc.UpdateAddresses(b.sc, cs.ResolverState.Addresses)
		b.sc.Connect()
	}
	return nil
}

func (b *pickfirstBalancer) UpdateSubConnState(sc balancer.SubConn, s balancer.SubConnState) {
	if logger.V(2) {
		logger.Infof("pickfirstBalancer: UpdateSubConnState: %p, %v", sc, s)
	}
	if b.sc != sc {
		if logger.V(2) {
			logger.Infof("pickfirstBalancer: ignored state change because sc is not recognized")
		}
		return
	}
	b.state = s.ConnectivityState
	if s.ConnectivityState == connectivity.Shutdown {
		b.sc = nil
		return
	}

	switch s.ConnectivityState {
	case connectivity.Ready:
		b.cc.UpdateState(balancer.State{ConnectivityState: s.ConnectivityState, Picker: &picker{result: balancer.PickResult{SubConn: sc}}})
	case connectivity.Connecting:
		b.cc.UpdateState(balancer.State{ConnectivityState: s.ConnectivityState, Picker: &picker{err: balancer.ErrNoSubConnAvailable}})
	case connectivity.Idle:
		b.cc.UpdateState(balancer.State{ConnectivityState: s.ConnectivityState, Picker: &idlePicker{sc: sc}})
	case connectivity.TransientFailure:
		b.cc.UpdateState(balancer.State{
			ConnectivityState: s.ConnectivityState,
			Picker:            &picker{err: s.ConnectionError},
		})
	}
}

func (b *pickfirstBalancer) Close() {
}

func (b *pickfirstBalancer) ExitIdle() {
	if b.sc != nil && b.state == connectivity.Idle {
		b.sc.Connect()
	}
}

type picker struct {
	result balancer.PickResult
	err    error
}

func (p *picker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	return p.result, p.err
}

// idlePicker is used when the SubConn is IDLE and kicks the SubConn into
// CONNECTING when Pick is called.
type idlePicker struct {
	sc balancer.SubConn
}

func (i *idlePicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	i.sc.Connect()
	return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
}

func init() {
	balancer.Register(newPickfirstBuilder())
}
