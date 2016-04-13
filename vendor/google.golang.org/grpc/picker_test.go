/*
 *
 * Copyright 2014, Google Inc.
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
	"testing"
	"time"

	"golang.org/x/net/context"
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

func startServers(t *testing.T, numServers, port int, maxStreams uint32) ([]*server, *testNameResolver) {
	var servers []*server
	for i := 0; i < numServers; i++ {
		s := newTestServer()
		servers = append(servers, s)
		go s.start(t, port, maxStreams)
		s.wait(t, 2*time.Second)
	}
	// Point to server1
	addr := "127.0.0.1:" + servers[0].port
	return servers, &testNameResolver{
		addr: addr,
	}
}

func TestNameDiscovery(t *testing.T) {
	// Start 3 servers on 3 ports.
	servers, r := startServers(t, 3, 0, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithPicker(NewUnicastNamingPicker(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	var reply string
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
	}
	// Inject name resolution change to point to the second server now.
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
	servers[0].stop()
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
	}
	// Add another server address (server#3) to name resolution
	updates = nil
	updates = append(updates, &naming.Update{
		Op:   naming.Add,
		Addr: "127.0.0.1:" + servers[2].port,
	})
	r.w.inject(updates)
	// Stop server#2. The library should direct to server#3 automatically.
	servers[1].stop()
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
	}
	cc.Close()
	servers[2].stop()
}

func TestEmptyAddrs(t *testing.T) {
	servers, r := startServers(t, 1, 0, math.MaxUint32)
	cc, err := Dial("foo.bar.com", WithPicker(NewUnicastNamingPicker(r)), WithBlock(), WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("Failed to create ClientConn: %v", err)
	}
	var reply string
	if err := Invoke(context.Background(), "/foo/bar", &expectedRequest, &reply, cc); err != nil || reply != expectedResponse {
		t.Fatalf("grpc.Invoke(_, _, _, _, _) = %v, want <nil>", err)
	}
	// Inject name resolution change to remove the server address so that there is no address
	// available after that.
	var updates []*naming.Update
	updates = append(updates, &naming.Update{
		Op:   naming.Delete,
		Addr: "127.0.0.1:" + servers[0].port,
	})
	r.w.inject(updates)
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
