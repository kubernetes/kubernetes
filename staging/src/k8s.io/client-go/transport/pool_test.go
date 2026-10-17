/*
Copyright 2026 The Kubernetes Authors.

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

package transport

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

type mockRoundTripper struct {
	roundTripFn func(*http.Request) (*http.Response, error)
	calls       atomic.Int64
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	m.calls.Add(1)
	if m.roundTripFn != nil {
		return m.roundTripFn(req)
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader([]byte("ok"))),
	}, nil
}

func TestNewPoolRoundTripperValidation(t *testing.T) {
	// Zero transports
	_, err := NewPoolRoundTripper(nil, PowerOfTwoChoices)
	if err == nil {
		t.Fatal("expected error with zero transports, got nil")
	}

	// Nil transport in slice
	_, err = NewPoolRoundTripper([]http.RoundTripper{&mockRoundTripper{}, nil}, PowerOfTwoChoices)
	if err == nil {
		t.Fatal("expected error with nil transport element, got nil")
	}

	// Invalid strategy
	_, err = NewPoolRoundTripper([]http.RoundTripper{&mockRoundTripper{}}, BalancingStrategy("invalid"))
	if err == nil {
		t.Fatal("expected error with invalid strategy, got nil")
	}

	// Default strategy
	p, err := NewPoolRoundTripper([]http.RoundTripper{&mockRoundTripper{}}, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Strategy() != PowerOfTwoChoices {
		t.Fatalf("expected default strategy %q, got %q", PowerOfTwoChoices, p.Strategy())
	}
	if p.Size() != 1 {
		t.Fatalf("expected size 1, got %d", p.Size())
	}
	if p.WrappedRoundTripper() == nil {
		t.Fatal("expected non-nil WrappedRoundTripper")
	}
}

func TestRoundRobinBalancing(t *testing.T) {
	numConns := 3
	mocks := make([]*mockRoundTripper, numConns)
	rts := make([]http.RoundTripper, numConns)
	for i := 0; i < numConns; i++ {
		mocks[i] = &mockRoundTripper{}
		rts[i] = mocks[i]
	}

	pool, err := NewPoolRoundTripper(rts, RoundRobin)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	numRequests := 9
	for i := 0; i < numRequests; i++ {
		req, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)
		resp, err := pool.RoundTrip(req)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		_ = resp.Body.Close()
	}

	expectedPerConn := int64(numRequests / numConns)
	for i, m := range mocks {
		if got := m.calls.Load(); got != expectedPerConn {
			t.Errorf("connection %d: expected %d calls, got %d", i, expectedPerConn, got)
		}
	}
}

func TestPowerOfTwoChoicesInflightAvoidance(t *testing.T) {
	numConns := 4
	mocks := make([]*mockRoundTripper, numConns)
	rts := make([]http.RoundTripper, numConns)
	for i := 0; i < numConns; i++ {
		mocks[i] = &mockRoundTripper{}
		rts[i] = mocks[i]
	}

	pRaw, err := NewPoolRoundTripper(rts, PowerOfTwoChoices)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pool := pRaw.(*poolRoundTripper)

	// Artificially simulate high in-flight load on connection 0
	pool.conns[0].inflight.Store(50)

	// Send 200 requests. Since connection 0 has 50 inflight, P2C should almost never pick connection 0
	for i := 0; i < 200; i++ {
		req, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)
		resp, err := pool.RoundTrip(req)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		_ = resp.Body.Close()
	}

	c0Calls := mocks[0].calls.Load()
	if c0Calls > 5 {
		t.Errorf("expected connection 0 with high inflight to receive <= 5 calls, but got %d", c0Calls)
	}

	totalOther := mocks[1].calls.Load() + mocks[2].calls.Load() + mocks[3].calls.Load()
	if totalOther < 195 {
		t.Errorf("expected other connections to handle the bulk of requests, got %d", totalOther)
	}
}

func TestBlackholeReplicaSelfHealing(t *testing.T) {
	// Simulate 5 connections.
	// Connection 0 is connected to an unresponsive/blackholed backend replica (blocks indefinitely).
	// Connections 1..4 are healthy (respond immediately).
	numConns := 5
	mocks := make([]*mockRoundTripper, numConns)
	rts := make([]http.RoundTripper, numConns)

	unblockCh := make(chan struct{})
	defer close(unblockCh)

	for i := 0; i < numConns; i++ {
		idx := i
		if idx == 0 {
			mocks[idx] = &mockRoundTripper{
				roundTripFn: func(req *http.Request) (*http.Response, error) {
					select {
					case <-req.Context().Done():
						return nil, req.Context().Err()
					case <-unblockCh:
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(bytes.NewReader([]byte("unblocked"))),
						}, nil
					}
				},
			}
		} else {
			mocks[idx] = &mockRoundTripper{
				roundTripFn: func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(bytes.NewReader([]byte("healthy"))),
					}, nil
				},
			}
		}
		rts[idx] = mocks[idx]
	}

	pool, err := NewPoolRoundTripper(rts, PowerOfTwoChoices)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Launch concurrent traffic
	var wg sync.WaitGroup
	var successCount atomic.Int64
	var failureCount atomic.Int64

	numRequests := 100
	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			// Short timeout to represent client timeout on blackhole
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			req, _ := http.NewRequestWithContext(ctx, "GET", "http://example.com", nil)
			resp, err := pool.RoundTrip(req)
			if err != nil {
				failureCount.Add(1)
				return
			}
			_ = resp.Body.Close()
			successCount.Add(1)
		}(i)
		// Small stagger to simulate realistic client request arrival
		time.Sleep(1 * time.Millisecond)
	}

	wg.Wait()

	// With PoolSize=1 stuck on connection 0, success rate would be 0%.
	// With PoolSize=5 and P2C self-healing, success rate should be high (>80%).
	success := successCount.Load()
	t.Logf("Blackhole test results: %d successes, %d failures (out of %d total requests)",
		success, failureCount.Load(), numRequests)

	if success < 75 {
		t.Errorf("expected high success rate despite blackholed connection, got %d/%d", success, numRequests)
	}

	// Verify all healthy connections processed traffic
	for i := 1; i < numConns; i++ {
		if mocks[i].calls.Load() == 0 {
			t.Errorf("healthy connection %d did not receive any requests", i)
		}
	}
}

func TestInflightTrackingAndContextCancellation(t *testing.T) {
	mock := &mockRoundTripper{}
	pool, err := NewPoolRoundTripper([]http.RoundTripper{mock}, PowerOfTwoChoices)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if pool.Inflight(0) != 0 {
		t.Fatalf("expected initial inflight 0, got %d", pool.Inflight(0))
	}

	// Case 1: Normal request & response close
	req1, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)
	resp1, err := pool.RoundTrip(req1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pool.Inflight(0) != 1 {
		t.Fatalf("expected inflight 1 while body is open, got %d", pool.Inflight(0))
	}
	_ = resp1.Body.Close()
	if pool.Inflight(0) != 0 {
		t.Fatalf("expected inflight 0 after body closed, got %d", pool.Inflight(0))
	}

	// Case 2: Context cancellation while body open
	ctx, cancel := context.WithCancel(context.Background())
	req2, _ := http.NewRequestWithContext(ctx, "GET", "http://example.com", nil)
	resp2, err := pool.RoundTrip(req2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pool.Inflight(0) != 1 {
		t.Fatalf("expected inflight 1 while body is open, got %d", pool.Inflight(0))
	}
	cancel() // Cancel context before Close()
	// context.AfterFunc triggers asynchronously or synchronously on cancel
	time.Sleep(10 * time.Millisecond)
	if pool.Inflight(0) != 0 {
		t.Fatalf("expected inflight 0 after context cancellation, got %d", pool.Inflight(0))
	}
	// Calling Close() after cancellation should be idempotent
	_ = resp2.Body.Close()
	if pool.Inflight(0) != 0 {
		t.Fatalf("expected inflight 0 after idempotent body close, got %d", pool.Inflight(0))
	}

	// Case 3: RoundTrip error
	errorMock := &mockRoundTripper{
		roundTripFn: func(req *http.Request) (*http.Response, error) {
			return nil, errors.New("network dial failure")
		},
	}
	errPool, _ := NewPoolRoundTripper([]http.RoundTripper{errorMock}, PowerOfTwoChoices)
	req3, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)
	_, err = errPool.RoundTrip(req3)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if errPool.Inflight(0) != 0 {
		t.Fatalf("expected inflight 0 after error, got %d", errPool.Inflight(0))
	}
}

func TestTransportNewWithConnectionPool(t *testing.T) {
	cfg := &Config{
		ConnectionPool: &ConnectionPoolConfig{
			Size:     3,
			Strategy: PowerOfTwoChoices,
		},
	}

	rt, err := New(cfg)
	if err != nil {
		t.Fatalf("unexpected error creating transport with ConnectionPool: %v", err)
	}

	// Find the PoolRoundTripper within the wrapped chain
	var pool PoolRoundTripper
	curr := rt
	for curr != nil {
		if p, ok := curr.(PoolRoundTripper); ok {
			pool = p
			break
		}
		if wrapper, ok := curr.(utilnet.RoundTripperWrapper); ok {
			curr = wrapper.WrappedRoundTripper()
		} else {
			break
		}
	}

	if pool == nil {
		t.Fatal("expected transport chain to contain PoolRoundTripper")
	}
	if pool.Size() != 3 {
		t.Fatalf("expected pool size 3, got %d", pool.Size())
	}
	if pool.Strategy() != PowerOfTwoChoices {
		t.Fatalf("expected strategy %q, got %q", PowerOfTwoChoices, pool.Strategy())
	}
}

func BenchmarkRoundRobin(b *testing.B) {
	numConns := 10
	rts := make([]http.RoundTripper, numConns)
	for i := 0; i < numConns; i++ {
		rts[i] = &mockRoundTripper{}
	}
	pool, _ := NewPoolRoundTripper(rts, RoundRobin)
	req, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			resp, _ := pool.RoundTrip(req)
			_ = resp.Body.Close()
		}
	})
}

func BenchmarkPowerOfTwoChoices(b *testing.B) {
	numConns := 10
	rts := make([]http.RoundTripper, numConns)
	for i := 0; i < numConns; i++ {
		rts[i] = &mockRoundTripper{}
	}
	pool, _ := NewPoolRoundTripper(rts, PowerOfTwoChoices)
	req, _ := http.NewRequestWithContext(context.Background(), "GET", "http://example.com", nil)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			resp, _ := pool.RoundTrip(req)
			_ = resp.Body.Close()
		}
	})
}

