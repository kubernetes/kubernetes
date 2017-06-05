/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/naming"
)

type testWatcher struct {
	// the channel to receives name resolution updates
	update chan *naming.Update
	// the side channel to get to know how many updates in a batch
	side chan int
	// the channel to notifiy update injector that the update reading is done
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

func startServers(t *testing.T, numServers int, maxStreams uint32) ([]*server, *testNameResolver) {
	var servers []*server
	for i := 0; i < numServers; i++ {
		s := newTestServer()
		servers = append(servers, s)
		go s.start(t, 0, maxStreams)
		s.wait(t, 2*time.Second)
	}
	// Point to server[0]
	addr := "127.0.0.1:" + servers[0].port
	return servers, &testNameResolver{
		addr: addr,
	}
}

func TestNameDiscovery(t *testing.T) {
	// Start 2 servers on 2 ports.
	numServers := 2
	servers, r := startServers(t, numServers, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	req := "port"
	var reply string
	if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err == nil || ErrorDesc(err) != servers[0].port {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want %s", err, servers[0].port)
	}
	// Inject the name resolution change to remove servers[0] and add servers[1].
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Delete,
		Addr: "127.0.0.1:" + servers[0].port,
	})
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[1].port,
	})
	r.w.inject(updates)
	// Loop until the rpcs in flight talks to servers[1].
	for {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err != nil && ErrorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	cc.Close()
	for i := 0; i < numServers; i++ {
		servers[i].stop()
	}
}

func TestEmptyAddrs(t *testing.T) {
	servers, r := startServers(t, 1, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	var reply string
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, reply = %q, want %q, <nil>", err, reply, expectedResponse)
	}
	// Inject name resolution change to remove the server so that there is no address
	// available after that.
	u := &naming.Update{
		Op:   naming.Delete,
		Addr: "127.0.0.1:" + servers[0].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until the above updates apply.
	for {
		time.Sleep(10 * time.Millisecond)
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := Invoke(ctx, "/foo/bar", &expectedRequest, &reply, cc); err != nil {
			break
		}
	}
	cc.Close()
	servers[0].stop()
}

func TestRoundRobin(t *testing.T) {
	// Start 3 servers on 3 ports.
	numServers := 3
	servers, r := startServers(t, numServers, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	// Add servers[1] to the service discovery.
	u := &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[1].port,
	}
	r.w.inject([]*naming.Update{u})
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err != nil && ErrorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Add server2[2] to the service discovery.
	u = &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[2].port,
	}
	r.w.inject([]*naming.Update{u})
	// Loop until both servers[2] are up.
	for {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err != nil && ErrorDesc(err) == servers[2].port {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Check the incoming RPCs served in a round-robin manner.
	for i := 0; i < 10; i++ {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err == nil || ErrorDesc(err) != servers[i%numServers].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", i, err, servers[i%numServers].port)
		}
	}
	cc.Close()
	for i := 0; i < numServers; i++ {
		servers[i].stop()
	}
}

func TestCloseWithPendingRPC(t *testing.T) {
	servers, r := startServers(t, 1, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	var reply string
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); err != nil {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want %s", err, servers[0].port)
	}
	// Remove the server.
	updates := []*naming.Update{{
		Op:   naming.Delete,
		Addr: "127.0.0.1:" + servers[0].port,
	}}
	r.w.inject(updates)
	// Loop until the above update applies.
	for {
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := Invoke(ctx, "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); Code(err) == codes.DeadlineExceeded {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Issue 2 RPCs which should be completed with error status once cc is closed.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		var reply string
		if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	go func() {
		defer wg.Done()
		var reply string
		time.Sleep(5 * time.Millisecond)
		if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); err == nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
		}
	}()
	time.Sleep(5 * time.Millisecond)
	cc.Close()
	wg.Wait()
	servers[0].stop()
}

func TestGetOnWaitChannel(t *testing.T) {
	servers, r := startServers(t, 1, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	// Remove all servers so that all upcoming RPCs will block on waitCh.
	updates := []*naming.Update{{
		Op:   naming.Delete,
		Addr: "127.0.0.1:" + servers[0].port,
	}}
	r.w.inject(updates)
	for {
		var reply string
		ctx, _ := context.WithTimeout(context.Background(), 10*time.Millisecond)
		if err := Invoke(ctx, "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); Code(err) == codes.DeadlineExceeded {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var reply string
		if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); err != nil {
			t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
		}
	}()
	// Add a connected server to get the above RPC through.
	updates = []*naming.Update{{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[0].port,
	}}
	r.w.inject(updates)
	// Wait until the above RPC succeeds.
	wg.Wait()
	cc.Close()
	servers[0].stop()
}

func TestOneServerDown(t *testing.T) {
	// Start 2 servers.
	numServers := 2
	servers, r := startServers(t, numServers, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	// Add servers[1] to the service discovery.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[1].port,
	})
	r.w.inject(updates)
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err != nil && ErrorDesc(err) == servers[1].port {
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
			Invoke(context.Background(), "/foo/bar", &req, &reply, cc, FailFast(false))
			wg.Done()
		}()
	}
	wg.Wait()
	cc.Close()
	for i := 0; i < numServers; i++ {
		servers[i].stop()
	}
}

func TestOneAddressRemoval(t *testing.T) {
	// Start 2 servers.
	numServers := 2
	servers, r := startServers(t, numServers, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithBalancer(RoundRobin(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	// Add servers[1] to the service discovery.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[1].port,
	})
	r.w.inject(updates)
	req := "port"
	var reply string
	// Loop until servers[1] is up
	for {
		if err := Invoke(context.Background(), "/foo/bar", &req, &reply, cc); err != nil && ErrorDesc(err) == servers[1].port {
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
			Addr: "127.0.0.1:" + servers[0].port,
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
			if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc, FailFast(false)); err != nil {
				t.Errorf("grpc.Invoke(_, _, _, _, _) = %v, want not nil", err)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	cc.Close()
	for i := 0; i < numServers; i++ {
		servers[i].stop()
	}
}
