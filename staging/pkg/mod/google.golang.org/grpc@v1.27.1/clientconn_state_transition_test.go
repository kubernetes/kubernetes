/*
 *
 * Copyright 2018 gRPC authors.
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
	"net"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/http2"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
)

const stateRecordingBalancerName = "state_recoding_balancer"

var testBalancerBuilder = newStateRecordingBalancerBuilder()

func init() {
	balancer.Register(testBalancerBuilder)
}

// These tests use a pipeListener. This listener is similar to net.Listener
// except that it is unbuffered, so each read and write will wait for the other
// side's corresponding write or read.
func (s) TestStateTransitions_SingleAddress(t *testing.T) {
	for _, test := range []struct {
		desc   string
		want   []connectivity.State
		server func(net.Listener) net.Conn
	}{
		{
			desc: "When the server returns server preface, the client enters READY.",
			want: []connectivity.State{
				connectivity.Connecting,
				connectivity.Ready,
			},
			server: func(lis net.Listener) net.Conn {
				conn, err := lis.Accept()
				if err != nil {
					t.Error(err)
					return nil
				}

				go keepReading(conn)

				framer := http2.NewFramer(conn, conn)
				if err := framer.WriteSettings(http2.Setting{}); err != nil {
					t.Errorf("Error while writing settings frame. %v", err)
					return nil
				}

				return conn
			},
		},
		{
			desc: "When the connection is closed, the client enters TRANSIENT FAILURE.",
			want: []connectivity.State{
				connectivity.Connecting,
				connectivity.TransientFailure,
			},
			server: func(lis net.Listener) net.Conn {
				conn, err := lis.Accept()
				if err != nil {
					t.Error(err)
					return nil
				}

				conn.Close()
				return nil
			},
		},
		{
			desc: `When the server sends its connection preface, but the connection dies before the client can write its
connection preface, the client enters TRANSIENT FAILURE.`,
			want: []connectivity.State{
				connectivity.Connecting,
				connectivity.TransientFailure,
			},
			server: func(lis net.Listener) net.Conn {
				conn, err := lis.Accept()
				if err != nil {
					t.Error(err)
					return nil
				}

				framer := http2.NewFramer(conn, conn)
				if err := framer.WriteSettings(http2.Setting{}); err != nil {
					t.Errorf("Error while writing settings frame. %v", err)
					return nil
				}

				conn.Close()
				return nil
			},
		},
		{
			desc: `When the server reads the client connection preface but does not send its connection preface, the
client enters TRANSIENT FAILURE.`,
			want: []connectivity.State{
				connectivity.Connecting,
				connectivity.TransientFailure,
			},
			server: func(lis net.Listener) net.Conn {
				conn, err := lis.Accept()
				if err != nil {
					t.Error(err)
					return nil
				}

				go keepReading(conn)

				return conn
			},
		},
	} {
		t.Log(test.desc)
		testStateTransitionSingleAddress(t, test.want, test.server)
	}
}

func testStateTransitionSingleAddress(t *testing.T, want []connectivity.State, server func(net.Listener) net.Conn) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	pl := testutils.NewPipeListener()
	defer pl.Close()

	// Launch the server.
	var conn net.Conn
	var connMu sync.Mutex
	go func() {
		connMu.Lock()
		conn = server(pl)
		connMu.Unlock()
	}()

	client, err := DialContext(ctx,
		"",
		WithInsecure(),
		WithBalancerName(stateRecordingBalancerName),
		WithDialer(pl.Dialer()),
		withBackoff(noBackoff{}),
		withMinConnectDeadline(func() time.Duration { return time.Millisecond * 100 }))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	stateNotifications := testBalancerBuilder.nextStateNotifier()

	timeout := time.After(5 * time.Second)

	for i := 0; i < len(want); i++ {
		select {
		case <-timeout:
			t.Fatalf("timed out waiting for state %d (%v) in flow %v", i, want[i], want)
		case seen := <-stateNotifications:
			if seen != want[i] {
				t.Fatalf("expected to see %v at position %d in flow %v, got %v", want[i], i, want, seen)
			}
		}
	}

	connMu.Lock()
	defer connMu.Unlock()
	if conn != nil {
		err = conn.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
}

// When a READY connection is closed, the client enters CONNECTING.
func (s) TestStateTransitions_ReadyToConnecting(t *testing.T) {
	want := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
		connectivity.Connecting,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis.Close()

	sawReady := make(chan struct{})

	// Launch the server.
	go func() {
		conn, err := lis.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		go keepReading(conn)

		framer := http2.NewFramer(conn, conn)
		if err := framer.WriteSettings(http2.Setting{}); err != nil {
			t.Errorf("Error while writing settings frame. %v", err)
			return
		}

		// Prevents race between onPrefaceReceipt and onClose.
		<-sawReady

		conn.Close()
	}()

	client, err := DialContext(ctx, lis.Addr().String(), WithInsecure(), WithBalancerName(stateRecordingBalancerName))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	stateNotifications := testBalancerBuilder.nextStateNotifier()

	timeout := time.After(5 * time.Second)

	for i := 0; i < len(want); i++ {
		select {
		case <-timeout:
			t.Fatalf("timed out waiting for state %d (%v) in flow %v", i, want[i], want)
		case seen := <-stateNotifications:
			if seen == connectivity.Ready {
				close(sawReady)
			}
			if seen != want[i] {
				t.Fatalf("expected to see %v at position %d in flow %v, got %v", want[i], i, want, seen)
			}
		}
	}
}

// When the first connection is closed, the client stays in CONNECTING until it
// tries the second address (which succeeds, and then it enters READY).
func (s) TestStateTransitions_TriesAllAddrsBeforeTransientFailure(t *testing.T) {
	want := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	lis1, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis1.Close()

	lis2, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis2.Close()

	server1Done := make(chan struct{})
	server2Done := make(chan struct{})

	// Launch server 1.
	go func() {
		conn, err := lis1.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		conn.Close()
		close(server1Done)
	}()
	// Launch server 2.
	go func() {
		conn, err := lis2.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		go keepReading(conn)

		framer := http2.NewFramer(conn, conn)
		if err := framer.WriteSettings(http2.Setting{}); err != nil {
			t.Errorf("Error while writing settings frame. %v", err)
			return
		}

		close(server2Done)
	}()

	rb := manual.NewBuilderWithScheme("whatever")
	rb.InitialState(resolver.State{Addresses: []resolver.Address{
		{Addr: lis1.Addr().String()},
		{Addr: lis2.Addr().String()},
	}})
	client, err := DialContext(ctx, "whatever:///this-gets-overwritten", WithInsecure(), WithBalancerName(stateRecordingBalancerName), WithResolvers(rb))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	stateNotifications := testBalancerBuilder.nextStateNotifier()

	timeout := time.After(5 * time.Second)

	for i := 0; i < len(want); i++ {
		select {
		case <-timeout:
			t.Fatalf("timed out waiting for state %d (%v) in flow %v", i, want[i], want)
		case seen := <-stateNotifications:
			if seen != want[i] {
				t.Fatalf("expected to see %v at position %d in flow %v, got %v", want[i], i, want, seen)
			}
		}
	}
	select {
	case <-timeout:
		t.Fatal("saw the correct state transitions, but timed out waiting for client to finish interactions with server 1")
	case <-server1Done:
	}
	select {
	case <-timeout:
		t.Fatal("saw the correct state transitions, but timed out waiting for client to finish interactions with server 2")
	case <-server2Done:
	}
}

// When there are multiple addresses, and we enter READY on one of them, a
// later closure should cause the client to enter CONNECTING
func (s) TestStateTransitions_MultipleAddrsEntersReady(t *testing.T) {
	want := []connectivity.State{
		connectivity.Connecting,
		connectivity.Ready,
		connectivity.Connecting,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	lis1, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis1.Close()

	// Never actually gets used; we just want it to be alive so that the resolver has two addresses to target.
	lis2, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Error while listening. Err: %v", err)
	}
	defer lis2.Close()

	server1Done := make(chan struct{})
	sawReady := make(chan struct{})

	// Launch server 1.
	go func() {
		conn, err := lis1.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		go keepReading(conn)

		framer := http2.NewFramer(conn, conn)
		if err := framer.WriteSettings(http2.Setting{}); err != nil {
			t.Errorf("Error while writing settings frame. %v", err)
			return
		}

		<-sawReady

		conn.Close()

		_, err = lis1.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		close(server1Done)
	}()

	rb := manual.NewBuilderWithScheme("whatever")
	rb.InitialState(resolver.State{Addresses: []resolver.Address{
		{Addr: lis1.Addr().String()},
		{Addr: lis2.Addr().String()},
	}})
	client, err := DialContext(ctx, "whatever:///this-gets-overwritten", WithInsecure(), WithBalancerName(stateRecordingBalancerName), WithResolvers(rb))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	stateNotifications := testBalancerBuilder.nextStateNotifier()

	timeout := time.After(2 * time.Second)

	for i := 0; i < len(want); i++ {
		select {
		case <-timeout:
			t.Fatalf("timed out waiting for state %d (%v) in flow %v", i, want[i], want)
		case seen := <-stateNotifications:
			if seen == connectivity.Ready {
				close(sawReady)
			}
			if seen != want[i] {
				t.Fatalf("expected to see %v at position %d in flow %v, got %v", want[i], i, want, seen)
			}
		}
	}
	select {
	case <-timeout:
		t.Fatal("saw the correct state transitions, but timed out waiting for client to finish interactions with server 1")
	case <-server1Done:
	}
}

type stateRecordingBalancer struct {
	notifier chan<- connectivity.State
	balancer.Balancer
}

func (b *stateRecordingBalancer) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	b.notifier <- s
	b.Balancer.HandleSubConnStateChange(sc, s)
}

func (b *stateRecordingBalancer) ResetNotifier(r chan<- connectivity.State) {
	b.notifier = r
}

func (b *stateRecordingBalancer) Close() {
	b.Balancer.Close()
}

type stateRecordingBalancerBuilder struct {
	mu       sync.Mutex
	notifier chan connectivity.State // The notifier used in the last Balancer.
}

func newStateRecordingBalancerBuilder() *stateRecordingBalancerBuilder {
	return &stateRecordingBalancerBuilder{}
}

func (b *stateRecordingBalancerBuilder) Name() string {
	return stateRecordingBalancerName
}

func (b *stateRecordingBalancerBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	stateNotifications := make(chan connectivity.State, 10)
	b.mu.Lock()
	b.notifier = stateNotifications
	b.mu.Unlock()
	return &stateRecordingBalancer{
		notifier: stateNotifications,
		Balancer: balancer.Get(PickFirstBalancerName).Build(cc, opts),
	}
}

func (b *stateRecordingBalancerBuilder) nextStateNotifier() <-chan connectivity.State {
	b.mu.Lock()
	defer b.mu.Unlock()
	ret := b.notifier
	b.notifier = nil
	return ret
}

type noBackoff struct{}

func (b noBackoff) Backoff(int) time.Duration { return time.Duration(0) }

// Keep reading until something causes the connection to die (EOF, server
// closed, etc). Useful as a tool for mindlessly keeping the connection
// healthy, since the client will error if things like client prefaces are not
// accepted in a timely fashion.
func keepReading(conn net.Conn) {
	buf := make([]byte, 1024)
	for _, err := conn.Read(buf); err == nil; _, err = conn.Read(buf) {
	}
}
