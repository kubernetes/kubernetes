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
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/serviceconfig"
)

var _ balancer.Builder = &magicalLB{}
var _ balancer.Balancer = &magicalLB{}

// magicalLB is a ringer for grpclb.  It is used to avoid circular dependencies on the grpclb package
type magicalLB struct{}

func (b *magicalLB) Name() string {
	return "grpclb"
}

func (b *magicalLB) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	return b
}

func (b *magicalLB) HandleSubConnStateChange(balancer.SubConn, connectivity.State) {}

func (b *magicalLB) HandleResolvedAddrs([]resolver.Address, error) {}

func (b *magicalLB) Close() {}

func init() {
	balancer.Register(&magicalLB{})
}

func checkPickFirst(cc *ClientConn, servers []*server) error {
	var (
		req   = "port"
		reply string
		err   error
	)
	connected := false
	for i := 0; i < 5000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); errorDesc(err) == servers[0].port {
			if connected {
				// connected is set to false if peer is not server[0]. So if
				// connected is true here, this is the second time we saw
				// server[0] in a row. Break because pickfirst is in effect.
				break
			}
			connected = true
		} else {
			connected = false
		}
		time.Sleep(time.Millisecond)
	}
	if !connected {
		return fmt.Errorf("pickfirst is not in effect after 5 second, EmptyCall() = _, %v, want _, %v", err, servers[0].port)
	}
	// The following RPCs should all succeed with the first server.
	for i := 0; i < 3; i++ {
		err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply)
		if errorDesc(err) != servers[0].port {
			return fmt.Errorf("index %d: want peer %v, got peer %v", i, servers[0].port, err)
		}
	}
	return nil
}

func checkRoundRobin(cc *ClientConn, servers []*server) error {
	var (
		req   = "port"
		reply string
		err   error
	)

	// Make sure connections to all servers are up.
	for i := 0; i < 2; i++ {
		// Do this check twice, otherwise the first RPC's transport may still be
		// picked by the closing pickfirst balancer, and the test becomes flaky.
		for _, s := range servers {
			var up bool
			for i := 0; i < 5000; i++ {
				if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); errorDesc(err) == s.port {
					up = true
					break
				}
				time.Sleep(time.Millisecond)
			}
			if !up {
				return fmt.Errorf("server %v is not up within 5 second", s.port)
			}
		}
	}

	serverCount := len(servers)
	for i := 0; i < 3*serverCount; i++ {
		err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply)
		if errorDesc(err) != servers[i%serverCount].port {
			return fmt.Errorf("index %d: want peer %v, got peer %v", i, servers[i%serverCount].port, err)
		}
	}
	return nil
}

func (s) TestSwitchBalancer(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	const numServers = 2
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	addrs := []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}}
	r.UpdateState(resolver.State{Addresses: addrs})
	// The default balancer is pickfirst.
	if err := checkPickFirst(cc, servers); err != nil {
		t.Fatalf("check pickfirst returned non-nil error: %v", err)
	}
	// Switch to roundrobin.
	cc.updateResolverState(resolver.State{ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`), Addresses: addrs}, nil)
	if err := checkRoundRobin(cc, servers); err != nil {
		t.Fatalf("check roundrobin returned non-nil error: %v", err)
	}
	// Switch to pickfirst.
	cc.updateResolverState(resolver.State{ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "pick_first"}`), Addresses: addrs}, nil)
	if err := checkPickFirst(cc, servers); err != nil {
		t.Fatalf("check pickfirst returned non-nil error: %v", err)
	}
}

// Test that balancer specified by dial option will not be overridden.
func (s) TestBalancerDialOption(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	const numServers = 2
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}), WithBalancerName(roundrobin.Name))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	addrs := []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}}
	r.UpdateState(resolver.State{Addresses: addrs})
	// The init balancer is roundrobin.
	if err := checkRoundRobin(cc, servers); err != nil {
		t.Fatalf("check roundrobin returned non-nil error: %v", err)
	}
	// Switch to pickfirst.
	cc.updateResolverState(resolver.State{ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "pick_first"}`), Addresses: addrs}, nil)
	// Balancer is still roundrobin.
	if err := checkRoundRobin(cc, servers); err != nil {
		t.Fatalf("check roundrobin returned non-nil error: %v", err)
	}
}

// First addr update contains grpclb.
func (s) TestSwitchBalancerGRPCLBFirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()

	// ClientConn will switch balancer to grpclb when receives an address of
	// type GRPCLB.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}, {Addr: "grpclb", Type: resolver.GRPCLB}}})
	var isGRPCLB bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not grpclb", cc.curBalancerName)
	}

	// New update containing new backend and new grpclb. Should not switch
	// balancer.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend2"}, {Addr: "grpclb2", Type: resolver.GRPCLB}}})
	for i := 0; i < 200; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if !isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("within 200 ms, cc.balancer switched to !grpclb, want grpclb")
	}

	var isPickFirst bool
	// Switch balancer to pickfirst.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}})
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isPickFirst = cc.curBalancerName == PickFirstBalancerName
		cc.mu.Unlock()
		if isPickFirst {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isPickFirst {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not pick_first", cc.curBalancerName)
	}
}

// First addr update does not contain grpclb.
func (s) TestSwitchBalancerGRPCLBSecond(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}})
	var isPickFirst bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isPickFirst = cc.curBalancerName == PickFirstBalancerName
		cc.mu.Unlock()
		if isPickFirst {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isPickFirst {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not pick_first", cc.curBalancerName)
	}

	// ClientConn will switch balancer to grpclb when receives an address of
	// type GRPCLB.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}, {Addr: "grpclb", Type: resolver.GRPCLB}}})
	var isGRPCLB bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not grpclb", cc.curBalancerName)
	}

	// New update containing new backend and new grpclb. Should not switch
	// balancer.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend2"}, {Addr: "grpclb2", Type: resolver.GRPCLB}}})
	for i := 0; i < 200; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if !isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("within 200 ms, cc.balancer switched to !grpclb, want grpclb")
	}

	// Switch balancer back.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}})
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isPickFirst = cc.curBalancerName == PickFirstBalancerName
		cc.mu.Unlock()
		if isPickFirst {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isPickFirst {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not pick_first", cc.curBalancerName)
	}
}

// Test that if the current balancer is roundrobin, after switching to grpclb,
// when the resolved address doesn't contain grpclb addresses, balancer will be
// switched back to roundrobin.
func (s) TestSwitchBalancerGRPCLBRoundRobin(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()

	sc := parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}, ServiceConfig: sc})
	var isRoundRobin bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isRoundRobin = cc.curBalancerName == "round_robin"
		cc.mu.Unlock()
		if isRoundRobin {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isRoundRobin {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not round_robin", cc.curBalancerName)
	}

	// ClientConn will switch balancer to grpclb when receives an address of
	// type GRPCLB.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "grpclb", Type: resolver.GRPCLB}}, ServiceConfig: sc})
	var isGRPCLB bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not grpclb", cc.curBalancerName)
	}

	// Switch balancer back.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}, ServiceConfig: sc})
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isRoundRobin = cc.curBalancerName == "round_robin"
		cc.mu.Unlock()
		if isRoundRobin {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isRoundRobin {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not round_robin", cc.curBalancerName)
	}
}

// Test that if resolved address list contains grpclb, the balancer option in
// service config won't take effect. But when there's no grpclb address in a new
// resolved address list, balancer will be switched to the new one.
func (s) TestSwitchBalancerGRPCLBServiceConfig(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}})
	var isPickFirst bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isPickFirst = cc.curBalancerName == PickFirstBalancerName
		cc.mu.Unlock()
		if isPickFirst {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isPickFirst {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not pick_first", cc.curBalancerName)
	}

	// ClientConn will switch balancer to grpclb when receives an address of
	// type GRPCLB.
	addrs := []resolver.Address{{Addr: "grpclb", Type: resolver.GRPCLB}}
	r.UpdateState(resolver.State{Addresses: addrs})
	var isGRPCLB bool
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isGRPCLB = cc.curBalancerName == "grpclb"
		cc.mu.Unlock()
		if isGRPCLB {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isGRPCLB {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not grpclb", cc.curBalancerName)
	}

	sc := parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)
	r.UpdateState(resolver.State{Addresses: addrs, ServiceConfig: sc})
	var isRoundRobin bool
	for i := 0; i < 200; i++ {
		cc.mu.Lock()
		isRoundRobin = cc.curBalancerName == "round_robin"
		cc.mu.Unlock()
		if isRoundRobin {
			break
		}
		time.Sleep(time.Millisecond)
	}
	// Balancer should NOT switch to round_robin because resolved list contains
	// grpclb.
	if isRoundRobin {
		t.Fatalf("within 200 ms, cc.balancer switched to round_robin, want grpclb")
	}

	// Switch balancer back.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "backend"}}, ServiceConfig: sc})
	for i := 0; i < 5000; i++ {
		cc.mu.Lock()
		isRoundRobin = cc.curBalancerName == "round_robin"
		cc.mu.Unlock()
		if isRoundRobin {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !isRoundRobin {
		t.Fatalf("after 5 second, cc.balancer is of type %v, not round_robin", cc.curBalancerName)
	}
}

// Test that when switching to grpclb fails because grpclb is not registered,
// the fallback balancer will only get backend addresses, not the grpclb server
// address.
//
// The tests sends 3 server addresses (all backends) as resolved addresses, but
// claim the first one is grpclb server. The all RPCs should all be send to the
// other addresses, not the first one.
func (s) TestSwitchBalancerGRPCLBWithGRPCLBNotRegistered(t *testing.T) {
	internal.BalancerUnregister("grpclb")
	defer balancer.Register(&magicalLB{})

	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	const numServers = 3
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[1].addr}, {Addr: servers[2].addr}}})
	// The default balancer is pickfirst.
	if err := checkPickFirst(cc, servers[1:]); err != nil {
		t.Fatalf("check pickfirst returned non-nil error: %v", err)
	}
	// Try switching to grpclb by sending servers[0] as grpclb address. It's
	// expected that servers[0] will be filtered out, so it will not be used by
	// the balancer.
	//
	// If the filtering failed, servers[0] will be used for RPCs and the RPCs
	// will succeed. The following checks will catch this and fail.
	addrs := []resolver.Address{
		{Addr: servers[0].addr, Type: resolver.GRPCLB},
		{Addr: servers[1].addr}, {Addr: servers[2].addr}}
	r.UpdateState(resolver.State{Addresses: addrs})
	// Still check for pickfirst, but only with server[1] and server[2].
	if err := checkPickFirst(cc, servers[1:]); err != nil {
		t.Fatalf("check pickfirst returned non-nil error: %v", err)
	}
	// Switch to roundrobin, and check against server[1] and server[2].
	cc.updateResolverState(resolver.State{ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`), Addresses: addrs}, nil)
	if err := checkRoundRobin(cc, servers[1:]); err != nil {
		t.Fatalf("check roundrobin returned non-nil error: %v", err)
	}
}

func parseCfg(r *manual.Resolver, s string) *serviceconfig.ParseResult {
	scpr := r.CC.ParseServiceConfig(s)
	if scpr.Err != nil {
		panic(fmt.Sprintf("Error parsing config %q: %v", s, scpr.Err))
	}
	return scpr
}
