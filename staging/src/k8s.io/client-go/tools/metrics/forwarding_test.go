/*
Copyright The Kubernetes Authors.

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

package metrics

import (
	"context"
	"net/url"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type captureExpiry struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureExpiry) Set(expiry *time.Time) { *c.calls = append(*c.calls, []any{expiry}) }

type captureDuration struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureDuration) Observe(duration time.Duration) {
	*c.calls = append(*c.calls, []any{duration})
}

type captureLatency struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureLatency) Observe(ctx context.Context, verb string, u url.URL, latency time.Duration) {
	*c.calls = append(*c.calls, []any{ctx, verb, u, latency})
}

type captureResolverLatency struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureResolverLatency) Observe(ctx context.Context, host string, latency time.Duration) {
	*c.calls = append(*c.calls, []any{ctx, host, latency})
}

type captureSize struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureSize) Observe(ctx context.Context, verb string, host string, size float64) {
	*c.calls = append(*c.calls, []any{ctx, verb, host, size})
}

type captureResult struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureResult) Increment(ctx context.Context, code string, method string, host string) {
	*c.calls = append(*c.calls, []any{ctx, code, method, host})
}

type captureCalls struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureCalls) Increment(exitCode int, callStatus string) {
	*c.calls = append(*c.calls, []any{exitCode, callStatus})
}

type capturePolicy struct {
	calls        *[][]any
	uncomparable []int
}

func (c capturePolicy) Increment(status string) { *c.calls = append(*c.calls, []any{status}) }

type captureRetry struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureRetry) IncrementRetry(ctx context.Context, code string, method string, host string) {
	*c.calls = append(*c.calls, []any{ctx, code, method, host})
}

type captureTransportCache struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureTransportCache) Observe(value int) { *c.calls = append(*c.calls, []any{value}) }

type captureTransportCreateCalls struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureTransportCreateCalls) Increment(result string) {
	*c.calls = append(*c.calls, []any{result})
}

type captureTransportCAReloads struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureTransportCAReloads) Increment(result, reason string) {
	*c.calls = append(*c.calls, []any{result, reason})
}

type captureTransportCertRotationGCCalls struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureTransportCertRotationGCCalls) Increment() { *c.calls = append(*c.calls, []any{}) }

type captureTransportCacheGCCalls struct {
	calls        *[][]any
	uncomparable []int
}

func (c captureTransportCacheGCCalls) Increment(result string) {
	*c.calls = append(*c.calls, []any{result})
}

// Counts alone would not catch a fan-out that drops a label, context or value.
func TestFanoutPreservesArguments(t *testing.T) {
	type contextKey struct{}
	ctx := context.WithValue(context.Background(), contextKey{}, "request")
	u := url.URL{Scheme: "https", Host: "example.org", Path: "/api/v1/pods"}
	duration := 137 * time.Millisecond
	instant := time.Unix(123, 456)
	expiry := &instant
	t.Run("Expiry", func(t *testing.T) {
		var got [][]any
		var hook ExpiryMetric
		for range 3 {
			hook = combineExpiry(hook, captureExpiry{calls: &got, uncomparable: []int{1}})
		}
		hook.Set(expiry)
		want := []any{expiry}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Duration", func(t *testing.T) {
		var got [][]any
		var hook DurationMetric
		for range 3 {
			hook = combineDuration(hook, captureDuration{calls: &got, uncomparable: []int{1}})
		}
		hook.Observe(duration)
		want := []any{duration}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Latency", func(t *testing.T) {
		var got [][]any
		var hook LatencyMetric
		for range 3 {
			hook = combineLatency(hook, captureLatency{calls: &got, uncomparable: []int{1}})
		}
		hook.Observe(ctx, "PATCH", u, duration)
		want := []any{ctx, "PATCH", u, duration}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("ResolverLatency", func(t *testing.T) {
		var got [][]any
		var hook ResolverLatencyMetric
		for range 3 {
			hook = combineResolverLatency(hook, captureResolverLatency{calls: &got, uncomparable: []int{1}})
		}
		hook.Observe(ctx, "example.org", duration)
		want := []any{ctx, "example.org", duration}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Size", func(t *testing.T) {
		var got [][]any
		var hook SizeMetric
		for range 3 {
			hook = combineSize(hook, captureSize{calls: &got, uncomparable: []int{1}})
		}
		hook.Observe(ctx, "PATCH", "example.org", float64(123))
		want := []any{ctx, "PATCH", "example.org", float64(123)}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Result", func(t *testing.T) {
		var got [][]any
		var hook ResultMetric
		for range 3 {
			hook = combineResult(hook, captureResult{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment(ctx, "429", "PATCH", "example.org")
		want := []any{ctx, "429", "PATCH", "example.org"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Calls", func(t *testing.T) {
		var got [][]any
		var hook CallsMetric
		for range 3 {
			hook = combineCalls(hook, captureCalls{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment(17, "failure")
		want := []any{17, "failure"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Policy", func(t *testing.T) {
		var got [][]any
		var hook PolicyCallsMetric
		for range 3 {
			hook = combinePolicy(hook, capturePolicy{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment("denied")
		want := []any{"denied"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("Retry", func(t *testing.T) {
		var got [][]any
		var hook RetryMetric
		for range 3 {
			hook = combineRetry(hook, captureRetry{calls: &got, uncomparable: []int{1}})
		}
		hook.IncrementRetry(ctx, "429", "PATCH", "example.org")
		want := []any{ctx, "429", "PATCH", "example.org"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("TransportCache", func(t *testing.T) {
		var got [][]any
		var hook TransportCacheMetric
		for range 3 {
			hook = combineTransportCache(hook, captureTransportCache{calls: &got, uncomparable: []int{1}})
		}
		hook.Observe(37)
		want := []any{37}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("TransportCreateCalls", func(t *testing.T) {
		var got [][]any
		var hook TransportCreateCallsMetric
		for range 3 {
			hook = combineTransportCreateCalls(hook, captureTransportCreateCalls{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment("miss")
		want := []any{"miss"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("TransportCAReloads", func(t *testing.T) {
		var got [][]any
		var hook TransportCAReloadsMetric
		for range 3 {
			hook = combineTransportCAReloads(hook, captureTransportCAReloads{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment("failure", "changed")
		want := []any{"failure", "changed"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("TransportCertRotationGCCalls", func(t *testing.T) {
		var got [][]any
		var hook TransportCertRotationGCCallsMetric
		for range 3 {
			hook = combineTransportCertRotationGCCalls(hook, captureTransportCertRotationGCCalls{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment()
		want := []any{}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
	t.Run("TransportCacheGCCalls", func(t *testing.T) {
		var got [][]any
		var hook TransportCacheGCCallsMetric
		for range 3 {
			hook = combineTransportCacheGCCalls(hook, captureTransportCacheGCCalls{calls: &got, uncomparable: []int{1}})
		}
		hook.Increment("deleted")
		want := []any{"deleted"}
		if len(got) != 3 {
			t.Fatalf("got %d observations, want 3", len(got))
		}
		for i, args := range got {
			if !reflect.DeepEqual(args, want) {
				t.Errorf("provider %d got %#v, want %#v", i, args, want)
			}
		}
	})
}
func TestEnsureRegisteredWaitsForConcurrentCallback(t *testing.T) {
	resetForTest(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	Register(RegisterOpts{RegisterFn: func() { calls.Add(1); close(entered); <-release }})
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(EnsureRegistered)
	}
	<-entered
	if !registerFnsPending.Load() {
		t.Error("registration marked complete while callback is blocked")
	}
	close(release)
	wg.Wait()
	if calls.Load() != 1 {
		t.Errorf("callback calls = %d, want 1", calls.Load())
	}
	if registerFnsPending.Load() {
		t.Error("registration still pending after all callers returned")
	}
}

// Callback-only registration does not write hooks and may race with EnsureRegistered.
func TestRegisterCallbacksDuringEnsure(t *testing.T) {
	resetForTest(t)
	entered, release := make(chan struct{}), make(chan struct{})
	Register(RegisterOpts{RegisterFn: func() { close(entered); <-release }})
	var wg sync.WaitGroup
	wg.Go(EnsureRegistered)
	<-entered
	var calls atomic.Int32
	for range 20 {
		wg.Go(func() { Register(RegisterOpts{RegisterFn: func() { calls.Add(1) }}) })
	}
	close(release)
	wg.Wait()
	EnsureRegistered()
	if calls.Load() != 20 {
		t.Errorf("callback calls = %d, want 20", calls.Load())
	}
}

func TestCapturedFanoutIsNotChangedByRegistration(t *testing.T) {
	resetForTest(t)
	var a, b, c atomic.Int32
	// Spare capacity makes an accidental append-in-place observable.
	captured := make(multiLatency, 2, 4)
	captured[0], captured[1] = countLatency{&a}, countLatency{&b}
	RequestLatency = captured
	Register(RegisterOpts{RequestLatency: countLatency{&c}})
	captured.Observe(context.Background(), "GET", url.URL{}, time.Second)
	if c.Load() != 0 || captured[:3][2] != nil {
		t.Fatal("registration mutated a previously captured fan-out")
	}
	RequestLatency.Observe(context.Background(), "GET", url.URL{}, time.Second)
	if a.Load() != 2 || b.Load() != 2 || c.Load() != 1 {
		t.Fatalf("observations = (%d, %d, %d), want (2, 2, 1)", a.Load(), b.Load(), c.Load())
	}
}

func TestDuplicateRegistrationDeliversPerRegistration(t *testing.T) {
	resetForTest(t)
	opts, c := newCountingOpts()
	Register(opts)
	Register(opts)
	invokeAllHooks()
	for name, got := range c.metrics() {
		if got != 2 {
			t.Errorf("%s observed %d times, want 2", name, got)
		}
	}
}
