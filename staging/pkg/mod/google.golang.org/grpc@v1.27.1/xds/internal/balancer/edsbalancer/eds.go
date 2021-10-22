/*
 *
 * Copyright 2019 gRPC authors.
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

// Package edsbalancer contains EDS balancer implementation.
package edsbalancer

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
	xdsclient "google.golang.org/grpc/xds/internal/client"
)

const (
	defaultTimeout = 10 * time.Second
	edsName        = "experimental_eds"
)

var (
	newEDSBalancer = func(cc balancer.ClientConn, loadStore lrs.Store) edsBalancerImplInterface {
		return newEDSBalancerImpl(cc, loadStore)
	}
)

func init() {
	balancer.Register(&edsBalancerBuilder{})
}

type edsBalancerBuilder struct{}

// Build helps implement the balancer.Builder interface.
func (b *edsBalancerBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	ctx, cancel := context.WithCancel(context.Background())
	x := &edsBalancer{
		ctx:             ctx,
		cancel:          cancel,
		cc:              cc,
		buildOpts:       opts,
		grpcUpdate:      make(chan interface{}),
		xdsClientUpdate: make(chan interface{}),
	}
	loadStore := lrs.NewStore()
	x.edsImpl = newEDSBalancer(x.cc, loadStore)
	x.client = newXDSClientWrapper(x.handleEDSUpdate, x.loseContact, x.buildOpts, loadStore)
	go x.run()
	return x
}

func (b *edsBalancerBuilder) Name() string {
	return edsName
}

func (b *edsBalancerBuilder) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	var cfg EDSConfig
	if err := json.Unmarshal(c, &cfg); err != nil {
		return nil, fmt.Errorf("unable to unmarshal balancer config %s into EDSConfig, error: %v", string(c), err)
	}
	return &cfg, nil
}

// edsBalancerImplInterface defines the interface that edsBalancerImpl must
// implement to communicate with edsBalancer.
//
// It's implemented by the real eds balancer and a fake testing eds balancer.
type edsBalancerImplInterface interface {
	// HandleEDSResponse passes the received EDS message from traffic director to eds balancer.
	HandleEDSResponse(edsResp *xdsclient.EDSUpdate)
	// HandleChildPolicy updates the eds balancer the intra-cluster load balancing policy to use.
	HandleChildPolicy(name string, config json.RawMessage)
	// HandleSubConnStateChange handles state change for SubConn.
	HandleSubConnStateChange(sc balancer.SubConn, state connectivity.State)
	// Close closes the eds balancer.
	Close()
}

var _ balancer.V2Balancer = (*edsBalancer)(nil) // Assert that we implement V2Balancer

// edsBalancer manages xdsClient and the actual EDS balancer implementation that
// does load balancing.
//
// It currently has only an edsBalancer. Later, we may add fallback.
type edsBalancer struct {
	cc        balancer.ClientConn // *xdsClientConn
	buildOpts balancer.BuildOptions
	ctx       context.Context
	cancel    context.CancelFunc

	// edsBalancer continuously monitor the channels below, and will handle events from them in sync.
	grpcUpdate      chan interface{}
	xdsClientUpdate chan interface{}

	client  *xdsclientWrapper // may change when passed a different service config
	config  *EDSConfig        // may change when passed a different service config
	edsImpl edsBalancerImplInterface
}

// run gets executed in a goroutine once edsBalancer is created. It monitors updates from grpc,
// xdsClient and load balancer. It synchronizes the operations that happen inside edsBalancer. It
// exits when edsBalancer is closed.
func (x *edsBalancer) run() {
	for {
		select {
		case update := <-x.grpcUpdate:
			x.handleGRPCUpdate(update)
		case update := <-x.xdsClientUpdate:
			x.handleXDSClientUpdate(update)
		case <-x.ctx.Done():
			if x.client != nil {
				x.client.close()
			}
			if x.edsImpl != nil {
				x.edsImpl.Close()
			}
			return
		}
	}
}

func (x *edsBalancer) handleGRPCUpdate(update interface{}) {
	switch u := update.(type) {
	case *subConnStateUpdate:
		if x.edsImpl != nil {
			x.edsImpl.HandleSubConnStateChange(u.sc, u.state.ConnectivityState)
		}
	case *balancer.ClientConnState:
		cfg, _ := u.BalancerConfig.(*EDSConfig)
		if cfg == nil {
			// service config parsing failed. should never happen.
			return
		}

		x.client.handleUpdate(cfg, u.ResolverState.Attributes)

		if x.config == nil {
			x.config = cfg
			return
		}

		// We will update the edsImpl with the new child policy, if we got a
		// different one.
		if x.edsImpl != nil && !reflect.DeepEqual(cfg.ChildPolicy, x.config.ChildPolicy) {
			if cfg.ChildPolicy != nil {
				x.edsImpl.HandleChildPolicy(cfg.ChildPolicy.Name, cfg.ChildPolicy.Config)
			} else {
				x.edsImpl.HandleChildPolicy(roundrobin.Name, nil)
			}
		}

		x.config = cfg
	default:
		// unreachable path
		panic("wrong update type")
	}
}

func (x *edsBalancer) handleXDSClientUpdate(update interface{}) {
	switch u := update.(type) {
	// TODO: this func should accept (*xdsclient.EDSUpdate, error), and process
	// the error, instead of having a separate loseContact signal.
	case *xdsclient.EDSUpdate:
		x.edsImpl.HandleEDSResponse(u)
	case *loseContact:
		// loseContact can be useful for going into fallback.
	default:
		panic("unexpected xds client update type")
	}
}

type subConnStateUpdate struct {
	sc    balancer.SubConn
	state balancer.SubConnState
}

func (x *edsBalancer) HandleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	grpclog.Error("UpdateSubConnState should be called instead of HandleSubConnStateChange")
}

func (x *edsBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	grpclog.Error("UpdateResolverState should be called instead of HandleResolvedAddrs")
}

func (x *edsBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	update := &subConnStateUpdate{
		sc:    sc,
		state: state,
	}
	select {
	case x.grpcUpdate <- update:
	case <-x.ctx.Done():
	}
}

func (x *edsBalancer) ResolverError(error) {
	// TODO: Need to distinguish between connection errors and resource removed
	// errors. For the former, we will need to handle it later on for fallback.
	// For the latter, handle it by stopping the watch, closing sub-balancers
	// and pickers.
}

func (x *edsBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	select {
	case x.grpcUpdate <- &s:
	case <-x.ctx.Done():
	}
	return nil
}

func (x *edsBalancer) handleEDSUpdate(resp *xdsclient.EDSUpdate) error {
	// TODO: this function should take (resp, error), and send them together on
	// the channel. There doesn't need to be a separate `loseContact` function.
	select {
	case x.xdsClientUpdate <- resp:
	case <-x.ctx.Done():
	}

	return nil
}

type loseContact struct {
}

// TODO: delete loseContact when handleEDSUpdate takes (resp, error).
func (x *edsBalancer) loseContact() {
	select {
	case x.xdsClientUpdate <- &loseContact{}:
	case <-x.ctx.Done():
	}
}

func (x *edsBalancer) Close() {
	x.cancel()
}
