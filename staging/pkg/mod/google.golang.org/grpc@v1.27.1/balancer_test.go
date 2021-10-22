/*
 *
 * Copyright 2016 gRPC authors.
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
	"strconv"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/naming"
	"google.golang.org/grpc/status"
)

func pickFirstBalancerV1(r naming.Resolver) Balancer {
	return &pickFirst{&roundRobin{r: r}}
}

type testWatcher struct {
	// the channel to receives name resolution updates
	update chan *naming.Update
	// the side channel to get to know how many updates in a batch
	side chan int
	// the channel to notify update injector that the update reading is done
	readDone chan int
}

func (w *testWatcher) Next() (updates []*naming.Update, err error) {
	n := <-w.side
	if n == 0 {
		return nil, fmt.Errorf("w.side is closed")
	}
	for i := 0; i < n; i++ {
		u := <-w.update
		if u != nil {
			updates = append(updates, u)
		}
	}
	w.readDone <- 0
	return
}

func (w *testWatcher) Close() {
	close(w.side)
}

// Inject naming resolution updates to the testWatcher.
func (w *testWatcher) inject(updates []*naming.Update) {
	w.side <- len(updates)
	for _, u := range updates {
		w.update <- u
	}
	<-w.readDone
}

type testNameResolver struct {
	w    *testWatcher
	addr string
}

func (r *testNameResolver) Resolve(target string) (naming.Watcher, error) {
	r.w = &testWatcher{
		update:   make(chan *naming.Update, 1),
		side:     make(chan int, 1),
		readDone: make(chan int),
	}
	r.w.side <- 1
	r.w.update <- &naming.Update{
		Op:   naming.Add,
		Addr: r.addr,
	}
	go func() {
		<-r.w.readDone
	}()
	return r.w, nil
}

func startServers(t *testing.T, numServers int, maxStreams uint32) ([]*server, *testNameResolver, func()) {
	var servers []*server
	for i := 0; i < numServers; i++ {
		s := newTestServer()
		servers = append(servers, s)
		go s.start(t, 0, maxStreams)
		s.wait(t, 2*time.Second)
	}
	// Point to server[0]
	addr := "localhost:" + servers[0].port
	return servers, &testNameResolver{
			addr: addr,
		}, func() {
			for i := 0; i < numServers; i++ {
				servers[i].stop()
			}
		}
}

func (s) TestNameDiscovery(t *testing.T) {
	// Start 2 servers on 2 ports.
	numServers := 2
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	req := "port"
	var reply string
	if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want %s", err, servers[0].port)
	}
	// Inject the name resolution change to remove servers[0] and add servers[1].
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	})
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	})
	r.w.inject(updates)
	// Loop until the rpcs in flight talks to servers[1].
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s) TestEmptyAddrs(t *testing.T) {
	servers, r, cleanup := startServers(t, 1, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	var reply string
	if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, reply = %q, want %q, <nil>", err, reply, expectedResponse)
	}
	// Inject name resolution change to remove the server so that there is no address
	// available after that.
	u := &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until the above updates apply.
	for {
		time.Sleep(10 * time.Millisecond)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := cc.Invoke(ctx, "/foo/bar", &expectedRequest, &reply); err != nil {
			cancel()
			break
		}
		cancel()
	}
}

func (s) TestRoundRobin(t *testing.T) {
	// Start 3 servers on 3 ports.
	numServers := 3
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] to the service discovery.
	u := &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Add server2[2] to the service discovery.
	u = &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[2].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until both servers[2] are up.
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[2].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Check the incoming RPCs served in a round-robin manner.
	for i := 0; i < 10; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[i%numServers].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", i, err, servers[i%numServers].port)
		}
	}
}

func (s) TestCloseWithPendingRPC(t *testing.T) {
	servers, r, cleanup := startServers(t, 1, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	var reply string
	if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err != nil {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want %s", err, servers[0].port)
	}
	// Remove the server.
	updates := []*naming.Update{{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}}
	r.w.inject(updates)
	// Loop until the above update applies.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := cc.Invoke(ctx, "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); status.Code(err) == codes.DeadlineExceeded {
			cancel()
			break
		}
		time.Sleep(10 * time.Millisecond)
		cancel()
	}
	// Issue 2 RPCs which should be completed with error status once cc is closed.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		var reply string
		if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	go func() {
		defer wg.Done()
		var reply string
		time.Sleep(5 * time.Millisecond)
		if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	time.Sleep(5 * time.Millisecond)
	cc.Close()
	wg.Wait()
}

func (s) TestGetOnWaitChannel(t *testing.T) {
	servers, r, cleanup := startServers(t, 1, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Remove all servers so that all upcoming RPCs will block on waitCh.
	updates := []*naming.Update{{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}}
	r.w.inject(updates)
	for {
		var reply string
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := cc.Invoke(ctx, "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); status.Code(err) == codes.DeadlineExceeded {
			cancel()
			break
		}
		cancel()
		time.Sleep(10 * time.Millisecond)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var reply string
		if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err != nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
		}
	}()
	// Add a connected server to get the above RPC through.
	updates = []*naming.Update{{
		Op:   naming.Add,
		Addr: "localhost:" + servers[0].port,
	}}
	r.w.inject(updates)
	// Wait until the above RPC succeeds.
	wg.Wait()
}

func (s) TestOneServerDown(t *testing.T) {
	// Start 2 servers.
	numServers := 2
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] to the service discovery.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	})
	r.w.inject(updates)
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	var wg sync.WaitGroup
	numRPC := 100
	sleepDuration := 10 * time.Millisecond
	wg.Add(1)
	go func() {
		time.Sleep(sleepDuration)
		// After sleepDuration, kill server[0].
		servers[0].stop()
		wg.Done()
	}()

	// All non-failfast RPCs should not block because there's at least one connection available.
	for i := 0; i < numRPC; i++ {
		wg.Add(1)
		go func() {
			time.Sleep(sleepDuration)
			// After sleepDuration, invoke RPC.
			// server[0] is killed around the same time to make it racy between balancer and gRPC internals.
			cc.Invoke(context.Background(), "/foo/bar", &req, &reply, WaitForReady(true))
			wg.Done()
		}()
	}
	wg.Wait()
}

func (s) TestOneAddressRemoval(t *testing.T) {
	// Start 2 servers.
	numServers := 2
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] to the service discovery.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	})
	r.w.inject(updates)
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	var wg sync.WaitGroup
	numRPC := 100
	sleepDuration := 10 * time.Millisecond
	wg.Add(1)
	go func() {
		time.Sleep(sleepDuration)
		// After sleepDuration, delete server[0].
		var updates []*naming.Update
		updates = append(updates, &naming.Update{
			Op:   naming.Delete,
			Addr: "localhost:" + servers[0].port,
		})
		r.w.inject(updates)
		wg.Done()
	}()

	// All non-failfast RPCs should not fail because there's at least one connection available.
	for i := 0; i < numRPC; i++ {
		wg.Add(1)
		go func() {
			var reply string
			time.Sleep(sleepDuration)
			// After sleepDuration, invoke RPC.
			// server[0] is removed around the same time to make it racy between balancer and gRPC internals.
			if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err != nil {
				t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want nil", err)
			}
			wg.Done()
		}()
	}
	wg.Wait()
}

func checkServerUp(t *testing.T, currentServer *server) {
	req := "port"
	port := currentServer.port
	cc, err := Dial("passthrough:///localhost:"+port, WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	var reply string
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s) TestPickFirstEmptyAddrs(t *testing.T) {
	servers, r, cleanup := startServers(t, 1, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(pickFirstBalancerV1(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	var reply string
	if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, reply = %q, want %q, <nil>", err, reply, expectedResponse)
	}
	// Inject name resolution change to remove the server so that there is no address
	// available after that.
	u := &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until the above updates apply.
	for {
		time.Sleep(10 * time.Millisecond)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := cc.Invoke(ctx, "/foo/bar", &expectedRequest, &reply); err != nil {
			cancel()
			break
		}
		cancel()
	}
}

func (s) TestPickFirstCloseWithPendingRPC(t *testing.T) {
	servers, r, cleanup := startServers(t, 1, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(pickFirstBalancerV1(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	var reply string
	if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err != nil {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want %s", err, servers[0].port)
	}
	// Remove the server.
	updates := []*naming.Update{{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}}
	r.w.inject(updates)
	// Loop until the above update applies.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := cc.Invoke(ctx, "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); status.Code(err) == codes.DeadlineExceeded {
			cancel()
			break
		}
		time.Sleep(10 * time.Millisecond)
		cancel()
	}
	// Issue 2 RPCs which should be completed with error status once cc is closed.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		var reply string
		if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	go func() {
		defer wg.Done()
		var reply string
		time.Sleep(5 * time.Millisecond)
		if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	time.Sleep(5 * time.Millisecond)
	cc.Close()
	wg.Wait()
}

func (s) TestPickFirstOrderAllServerUp(t *testing.T) {
	// Start 3 servers on 3 ports.
	numServers := 3
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(pickFirstBalancerV1(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] and [2] to the service discovery.
	u := &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})

	u = &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[2].port,
	}
	r.w.inject([]*naming.Update{u})

	// Loop until all 3 servers are up
	checkServerUp(t, servers[0])
	checkServerUp(t, servers[1])
	checkServerUp(t, servers[2])

	// Check the incoming RPCs served in server[0]
	req := "port"
	var reply string
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Delete server[0] in the balancer, the incoming RPCs served in server[1]
	// For test addrconn, close server[0] instead
	u = &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[0].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until it changes to server[1]
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Add server[0] back to the balancer, the incoming RPCs served in server[1]
	// Add is append operation, the order of Notify now is {server[1].port server[2].port server[0].port}
	u = &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[0].port,
	}
	r.w.inject([]*naming.Update{u})
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Delete server[1] in the balancer, the incoming RPCs served in server[2]
	u = &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[2].port {
			break
		}
		time.Sleep(1 * time.Second)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[2].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 2, err, servers[2].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Delete server[2] in the balancer, the incoming RPCs served in server[0]
	u = &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[2].port,
	}
	r.w.inject([]*naming.Update{u})
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(1 * time.Second)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s) TestPickFirstOrderOneServerDown(t *testing.T) {
	// Start 3 servers on 3 ports.
	numServers := 3
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///foo.bar.com", WithBalancer(pickFirstBalancerV1(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] and [2] to the service discovery.
	u := &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})

	u = &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[2].port,
	}
	r.w.inject([]*naming.Update{u})

	// Loop until all 3 servers are up
	checkServerUp(t, servers[0])
	checkServerUp(t, servers[1])
	checkServerUp(t, servers[2])

	// Check the incoming RPCs served in server[0]
	req := "port"
	var reply string
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// server[0] down, incoming RPCs served in server[1], but the order of Notify still remains
	// {server[0] server[1] server[2]}
	servers[0].stop()
	// Loop until it changes to server[1]
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// up the server[0] back, the incoming RPCs served in server[1]
	p, _ := strconv.Atoi(servers[0].port)
	servers[0] = newTestServer()
	go servers[0].start(t, p, math.MaxUint32)
	defer servers[0].stop()
	servers[0].wait(t, 2*time.Second)
	checkServerUp(t, servers[0])

	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Delete server[1] in the balancer, the incoming RPCs served in server[0]
	u = &naming.Update{
		Op:   naming.Delete,
		Addr: "localhost:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})
	for {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(1 * time.Second)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (s) TestPickFirstOneAddressRemoval(t *testing.T) {
	// Start 2 servers.
	numServers := 2
	servers, r, cleanup := startServers(t, numServers, math.MaxUint32)
	defer cleanup()
	cc, err := Dial("passthrough:///localhost:"+servers[0].port, WithBalancer(pickFirstBalancerV1(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	defer cc.Close()
	// Add servers[1] to the service discovery.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "localhost:" + servers[1].port,
	})
	r.w.inject(updates)

	// Create a new cc to Loop until servers[1] is up
	checkServerUp(t, servers[0])
	checkServerUp(t, servers[1])

	var wg sync.WaitGroup
	numRPC := 100
	sleepDuration := 10 * time.Millisecond
	wg.Add(1)
	go func() {
		time.Sleep(sleepDuration)
		// After sleepDuration, delete server[0].
		var updates []*naming.Update
		updates = append(updates, &naming.Update{
			Op:   naming.Delete,
			Addr: "localhost:" + servers[0].port,
		})
		r.w.inject(updates)
		wg.Done()
	}()

	// All non-failfast RPCs should not fail because there's at least one connection available.
	for i := 0; i < numRPC; i++ {
		wg.Add(1)
		go func() {
			var reply string
			time.Sleep(sleepDuration)
			// After sleepDuration, invoke RPC.
			// server[0] is removed around the same time to make it racy between balancer and gRPC internals.
			if err := cc.Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, WaitForReady(true)); err != nil {
				t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want nil", err)
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
