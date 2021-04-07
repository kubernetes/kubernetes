/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package connrotation

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCloseAll(t *testing.T) {
	closed := make(chan struct{}, 50)
	dialFn := func(ctx context.Context, network, address string) (net.Conn, error) {
		return closeOnlyConn{onClose: func() { closed <- struct{}{} }}, nil
	}
	dialer := NewDialer(dialFn)

	const numConns = 10

	// Outer loop to ensure Dialer is re-usable after CloseAll.
	for i := 0; i < 5; i++ {
		for j := 0; j < numConns; j++ {
			if _, err := dialer.Dial("", ""); err != nil {
				t.Fatal(err)
			}
		}
		dialer.CloseAll()
		deadline := time.After(time.Second)
		for j := 0; j < numConns; j++ {
			select {
			case <-closed:
			case <-deadline:
				t.Fatalf("iteration %d: 1s after CloseAll only %d/%d connections closed", i, j, numConns)
			}
		}
	}
}

// TestCloseAllRace ensures CloseAll works with connections being simultaneously dialed
func TestCloseAllRace(t *testing.T) {
	conns := int64(0)
	dialer := NewDialer(func(ctx context.Context, network, address string) (net.Conn, error) {
		return closeOnlyConn{onClose: func() { atomic.AddInt64(&conns, -1) }}, nil
	})

	const raceCount = 5000
	begin := &sync.WaitGroup{}
	begin.Add(1)

	wg := &sync.WaitGroup{}

	// Close all as fast as we can
	wg.Add(1)
	go func() {
		begin.Wait()
		defer wg.Done()
		for i := 0; i < raceCount; i++ {
			dialer.CloseAll()
		}
	}()

	// Dial as fast as we can
	wg.Add(1)
	go func() {
		begin.Wait()
		defer wg.Done()
		for i := 0; i < raceCount; i++ {
			if _, err := dialer.Dial("", ""); err != nil {
				t.Error(err)
				return
			}
			atomic.AddInt64(&conns, 1)
		}
	}()

	// Trigger both goroutines as close to the same time as possible
	begin.Done()

	// Wait for goroutines
	wg.Wait()

	// Ensure CloseAll ran after all dials
	dialer.CloseAll()

	// Expect all connections to close within 5 seconds
	for start := time.Now(); time.Now().Sub(start) < 5*time.Second; time.Sleep(10 * time.Millisecond) {
		// Ensure all connections were closed
		if c := atomic.LoadInt64(&conns); c == 0 {
			break
		} else {
			t.Logf("got %d open connections, want 0, will retry", c)
		}
	}
	// Ensure all connections were closed
	if c := atomic.LoadInt64(&conns); c != 0 {
		t.Fatalf("got %d open connections, want 0", c)
	}
}

type closeOnlyConn struct {
	net.Conn
	onClose func()
}

func (c closeOnlyConn) Close() error {
	go c.onClose()
	return nil
}
