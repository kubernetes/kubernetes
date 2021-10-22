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
	"math"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/status"
)

func errorDesc(err error) string {
	if s, ok := status.FromError(err); ok {
		return s.Message()
	}
	return err.Error()
}

func (s) TestOneBackendPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 1
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}}})
	// The second RPC should succeed.
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("EmptyCall() = _, %v, want _, %v", err, servers[0].port)
}

func (s) TestBackendsPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 2
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}}})
	// The second RPC should succeed with the first server.
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("EmptyCall() = _, %v, want _, %v", err, servers[0].port)
}

func (s) TestNewAddressWhileBlockingPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 1
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// This RPC blocks until NewAddress is called.
			cc.Invoke(context.Background(), "/foo/bar", &req, &reply)
		}()
	}
	time.Sleep(50 * time.Millisecond)
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}}})
	wg.Wait()
}

func (s) TestCloseWithPendingRPCPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 1
	_, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// This RPC blocks until NewAddress is called.
			cc.Invoke(context.Background(), "/foo/bar", &req, &reply)
		}()
	}
	time.Sleep(50 * time.Millisecond)
	cc.Close()
	wg.Wait()
}

func (s) TestOneServerDownPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 2
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}}})
	// The second RPC should succeed with the first server.
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(time.Millisecond)
	}

	servers[0].stop()
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("EmptyCall() = _, %v, want _, %v", err, servers[0].port)
}

func (s) TestAllServersDownPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 2
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}}})
	// The second RPC should succeed with the first server.
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(time.Millisecond)
	}

	for i := 0; i < numServers; i++ {
		servers[i].stop()
	}
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); status.Code(err) == codes.Unavailable {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("EmptyCall() = _, %v, want _, error with code unavailable", err)
}

func (s) TestAddressesRemovedPickfirst(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	numServers := 3
	servers, _, scleanup := startServers(t, numServers, math.MaxInt32)
	defer scleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure(), WithCodec(testCodec{}))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()
	// The first RPC should fail because there's no address.
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	req := "port"
	var reply string
	if err := cc.Invoke(ctx, "/foo/bar", &req, &reply); err == nil || status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("EmptyCall() = _, %v, want _, DeadlineExceeded", err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}, {Addr: servers[1].addr}, {Addr: servers[2].addr}}})
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Remove server[0].
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[1].addr}, {Addr: servers[2].addr}}})
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[1].port {
			break
		}
		time.Sleep(time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Append server[0], nothing should change.
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[1].addr}, {Addr: servers[2].addr}, {Addr: servers[0].addr}}})
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[1].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 1, err, servers[1].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Remove server[1].
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[2].addr}, {Addr: servers[0].addr}}})
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[2].port {
			break
		}
		time.Sleep(time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[2].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 2, err, servers[2].port)
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Remove server[2].
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: servers[0].addr}}})
	for i := 0; i < 1000; i++ {
		if err = cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err != nil && errorDesc(err) == servers[0].port {
			break
		}
		time.Sleep(time.Millisecond)
	}
	for i := 0; i < 20; i++ {
		if err := cc.Invoke(context.Background(), "/foo/bar", &req, &reply); err == nil || errorDesc(err) != servers[0].port {
			t.Fatalf("Index %d: Invoke(_, _, _, _, _) = %v, want %s", 0, err, servers[0].port)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
