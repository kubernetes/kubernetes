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

package grpc

import (
	"fmt"
	"net"
	"testing"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
)

var _ balancer.V2Balancer = &funcBalancer{}

type funcBalancer struct {
	updateClientConnState func(s balancer.ClientConnState) error
}

func (*funcBalancer) HandleSubConnStateChange(balancer.SubConn, connectivity.State) {
	panic("unimplemented") // v1 API
}
func (*funcBalancer) HandleResolvedAddrs([]resolver.Address, error) {
	panic("unimplemented") // v1 API
}
func (b *funcBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	return b.updateClientConnState(s)
}
func (*funcBalancer) ResolverError(error) {}
func (*funcBalancer) UpdateSubConnState(balancer.SubConn, balancer.SubConnState) {
	panic("unimplemented") // we never have sub-conns
}
func (*funcBalancer) Close() {}

type funcBalancerBuilder struct {
	name     string
	instance *funcBalancer
}

func (b *funcBalancerBuilder) Build(balancer.ClientConn, balancer.BuildOptions) balancer.Balancer {
	return b.instance
}
func (b *funcBalancerBuilder) Name() string { return b.name }

// TestBalancerErrorResolverPolling injects balancer errors and verifies
// ResolveNow is called on the resolver with the appropriate backoff strategy
// being consulted between ResolveNow calls.
func (s) TestBalancerErrorResolverPolling(t *testing.T) {
	// The test balancer will return ErrBadResolverState iff the
	// ClientConnState contains no addresses.
	fb := &funcBalancer{
		updateClientConnState: func(s balancer.ClientConnState) error {
			if len(s.ResolverState.Addresses) == 0 {
				return balancer.ErrBadResolverState
			}
			return nil
		},
	}
	const balName = "BalancerErrorResolverPolling"
	balancer.Register(&funcBalancerBuilder{name: balName, instance: fb})

	testResolverErrorPolling(t,
		func(r *manual.Resolver) {
			// No addresses so the balancer will fail.
			r.CC.UpdateState(resolver.State{})
		}, func(r *manual.Resolver) {
			// UpdateState will block if ResolveNow is being called (which blocks on
			// rn), so call it in a goroutine.  Include some address so the balancer
			// will be happy.
			go r.CC.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "x"}}})
		},
		WithDefaultServiceConfig(fmt.Sprintf(`{ "loadBalancingConfig": [{"%v": {}}] }`, balName)))
}

// TestRoundRobinZeroAddressesResolverPolling reports no addresses to the round
// robin balancer and verifies ResolveNow is called on the resolver with the
// appropriate backoff strategy being consulted between ResolveNow calls.
func (s) TestRoundRobinZeroAddressesResolverPolling(t *testing.T) {
	// We need to start a real server or else the connecting loop will call
	// ResolveNow after every iteration, even after a valid resolver result is
	// returned.
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis.Close()
	s := NewServer()
	defer s.Stop()
	go s.Serve(lis)

	testResolverErrorPolling(t,
		func(r *manual.Resolver) {
			// No addresses so the balancer will fail.
			r.CC.UpdateState(resolver.State{})
		}, func(r *manual.Resolver) {
			// UpdateState will block if ResolveNow is being called (which
			// blocks on rn), so call it in a goroutine.  Include a valid
			// address so the balancer will be happy.
			go r.CC.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: lis.Addr().String()}}})
		},
		WithDefaultServiceConfig(fmt.Sprintf(`{ "loadBalancingConfig": [{"%v": {}}] }`, roundrobin.Name)))
}
