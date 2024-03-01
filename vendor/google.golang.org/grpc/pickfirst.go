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
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	internalgrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcrand"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

const (
	// PickFirstBalancerName is the name of the pick_first balancer.
	PickFirstBalancerName = "pick_first"
	logPrefix             = "[pick-first-lb %p] "
)

func newPickfirstBuilder() balancer.Builder {
	return &pickfirstBuilder{}
}

type pickfirstBuilder struct{}

func (*pickfirstBuilder) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	b := &pickfirstBalancer{cc: cc}
	b.logger = internalgrpclog.NewPrefixLogger(logger, fmt.Sprintf(logPrefix, b))
	return b
}

func (*pickfirstBuilder) Name() string {
	return PickFirstBalancerName
}

type pfConfig struct {
	serviceconfig.LoadBalancingConfig `json:"-"`

	// If set to true, instructs the LB policy to shuffle the order of the list
	// of addresses received from the name resolver before attempting to
	// connect to them.
	ShuffleAddressList bool `json:"shuffleAddressList"`
}

func (*pickfirstBuilder) ParseConfig(js json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	var cfg pfConfig
	if err := json.Unmarshal(js, &cfg); err != nil {
		return nil, fmt.Errorf("pickfirst: unable to unmarshal LB policy config: %s, error: %v", string(js), err)
	}
	return cfg, nil
}

type pickfirstBalancer struct {
	logger  *internalgrpclog.PrefixLogger
	state   connectivity.State
	cc      balancer.ClientConn
	subConn balancer.SubConn
}

func (b *pickfirstBalancer) ResolverError(err error) {
	if b.logger.V(2) {
		b.logger.Infof("Received error from the name resolver: %v", err)
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
	addrs := state.ResolverState.Addresses
	if len(addrs) == 0 {
		// The resolver reported an empty address list. Treat it like an error by
		// calling b.ResolverError.
		if b.subConn != nil {
			// Shut down the old subConn. All addresses were removed, so it is
			// no longer valid.
			b.subConn.Shutdown()
			b.subConn = nil
		}
		b.ResolverError(errors.New("produced zero addresses"))
		return balancer.ErrBadResolverState
	}

	// We don't have to guard this block with the env var because ParseConfig
	// already does so.
	cfg, ok := state.BalancerConfig.(pfConfig)
	if state.BalancerConfig != nil && !ok {
		return fmt.Errorf("pickfirst: received illegal BalancerConfig (type %T): %v", state.BalancerConfig, state.BalancerConfig)
	}
	if cfg.ShuffleAddressList {
		addrs = append([]resolver.Address{}, addrs...)
		grpcrand.Shuffle(len(addrs), func(i, j int) { addrs[i], addrs[j] = addrs[j], addrs[i] })
	}

	if b.logger.V(2) {
		b.logger.Infof("Received new config %s, resolver state %s", pretty.ToJSON(cfg), pretty.ToJSON(state.ResolverState))
	}

	if b.subConn != nil {
		b.cc.UpdateAddresses(b.subConn, addrs)
		return nil
	}

	var subConn balancer.SubConn
	subConn, err := b.cc.NewSubConn(addrs, balancer.NewSubConnOptions{
		StateListener: func(state balancer.SubConnState) {
			b.updateSubConnState(subConn, state)
		},
	})
	if err != nil {
		if b.logger.V(2) {
			b.logger.Infof("Failed to create new SubConn: %v", err)
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

// UpdateSubConnState is unused as a StateListener is always registered when
// creating SubConns.
func (b *pickfirstBalancer) UpdateSubConnState(subConn balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", subConn, state)
}

func (b *pickfirstBalancer) updateSubConnState(subConn balancer.SubConn, state balancer.SubConnState) {
	if b.logger.V(2) {
		b.logger.Infof("Received SubConn state update: %p, %+v", subConn, state)
	}
	if b.subConn != subConn {
		if b.logger.V(2) {
			b.logger.Infof("Ignored state change because subConn is not recognized")
		}
		return
	}
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
		if b.state == connectivity.TransientFailure {
			// We stay in TransientFailure until we are Ready. See A62.
			return
		}
		b.cc.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            &picker{err: balancer.ErrNoSubConnAvailable},
		})
	case connectivity.Idle:
		if b.state == connectivity.TransientFailure {
			// We stay in TransientFailure until we are Ready. Also kick the
			// subConn out of Idle into Connecting. See A62.
			b.subConn.Connect()
			return
		}
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
	b.state = state.ConnectivityState
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
