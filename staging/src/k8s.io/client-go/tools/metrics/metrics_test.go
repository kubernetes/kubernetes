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
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These tests mutate package-level state, so none of them call t.Parallel.

// Each metric interface needs its own counting implementation: Observe and
// Increment are overloaded across the interfaces with different signatures, so
// a single fake cannot satisfy all of them.

type countExpiry struct{ n *atomic.Int32 }

func (c countExpiry) Set(*time.Time) { c.n.Add(1) }

type countDuration struct{ n *atomic.Int32 }

func (c countDuration) Observe(time.Duration) { c.n.Add(1) }

type countLatency struct{ n *atomic.Int32 }

func (c countLatency) Observe(context.Context, string, url.URL, time.Duration) { c.n.Add(1) }

type countResolverLatency struct{ n *atomic.Int32 }

func (c countResolverLatency) Observe(context.Context, string, time.Duration) { c.n.Add(1) }

type countSize struct{ n *atomic.Int32 }

func (c countSize) Observe(context.Context, string, string, float64) { c.n.Add(1) }

type countResult struct{ n *atomic.Int32 }

func (c countResult) Increment(context.Context, string, string, string) { c.n.Add(1) }

type countCalls struct{ n *atomic.Int32 }

func (c countCalls) Increment(int, string) { c.n.Add(1) }

type countPolicy struct{ n *atomic.Int32 }

func (c countPolicy) Increment(string) { c.n.Add(1) }

type countRetry struct{ n *atomic.Int32 }

func (c countRetry) IncrementRetry(context.Context, string, string, string) { c.n.Add(1) }

type countTransportCache struct{ n *atomic.Int32 }

func (c countTransportCache) Observe(int) { c.n.Add(1) }

type countTransportCreateCalls struct{ n *atomic.Int32 }

func (c countTransportCreateCalls) Increment(string) { c.n.Add(1) }

type countTransportCAReloads struct{ n *atomic.Int32 }

func (c countTransportCAReloads) Increment(string, string) { c.n.Add(1) }

type countTransportCertRotationGCCalls struct{ n *atomic.Int32 }

func (c countTransportCertRotationGCCalls) Increment() { c.n.Add(1) }

type countTransportCacheGCCalls struct{ n *atomic.Int32 }

func (c countTransportCacheGCCalls) Increment(string) { c.n.Add(1) }

// counters holds one counter per exported hook, plus one for RegisterFn.
type counters struct {
	clientCertExpiry             atomic.Int32
	clientCertRotationAge        atomic.Int32
	requestLatency               atomic.Int32
	resolverLatency              atomic.Int32
	requestSize                  atomic.Int32
	responseSize                 atomic.Int32
	rateLimiterLatency           atomic.Int32
	requestResult                atomic.Int32
	execPluginCalls              atomic.Int32
	execPluginPolicyCalls        atomic.Int32
	requestRetry                 atomic.Int32
	transportCacheEntries        atomic.Int32
	transportCreateCalls         atomic.Int32
	transportCAReloads           atomic.Int32
	transportCertRotationGCCalls atomic.Int32
	transportCacheGCCalls        atomic.Int32

	registerFn atomic.Int32
}

// metrics returns the per-hook counts, excluding registerFn.
func (c *counters) metrics() map[string]int32 {
	return map[string]int32{
		"ClientCertExpiry":             c.clientCertExpiry.Load(),
		"ClientCertRotationAge":        c.clientCertRotationAge.Load(),
		"RequestLatency":               c.requestLatency.Load(),
		"ResolverLatency":              c.resolverLatency.Load(),
		"RequestSize":                  c.requestSize.Load(),
		"ResponseSize":                 c.responseSize.Load(),
		"RateLimiterLatency":           c.rateLimiterLatency.Load(),
		"RequestResult":                c.requestResult.Load(),
		"ExecPluginCalls":              c.execPluginCalls.Load(),
		"ExecPluginPolicyCalls":        c.execPluginPolicyCalls.Load(),
		"RequestRetry":                 c.requestRetry.Load(),
		"TransportCacheEntries":        c.transportCacheEntries.Load(),
		"TransportCreateCalls":         c.transportCreateCalls.Load(),
		"TransportCAReloads":           c.transportCAReloads.Load(),
		"TransportCertRotationGCCalls": c.transportCertRotationGCCalls.Load(),
		"TransportCacheGCCalls":        c.transportCacheGCCalls.Load(),
	}
}

// newCountingOpts returns a RegisterOpts with every field populated by a
// distinct counting metric, along with the counters they increment.
func newCountingOpts() (RegisterOpts, *counters) {
	c := &counters{}
	return RegisterOpts{
		ClientCertExpiry:             countExpiry{&c.clientCertExpiry},
		ClientCertRotationAge:        countDuration{&c.clientCertRotationAge},
		RequestLatency:               countLatency{&c.requestLatency},
		ResolverLatency:              countResolverLatency{&c.resolverLatency},
		RequestSize:                  countSize{&c.requestSize},
		ResponseSize:                 countSize{&c.responseSize},
		RateLimiterLatency:           countLatency{&c.rateLimiterLatency},
		RequestResult:                countResult{&c.requestResult},
		ExecPluginCalls:              countCalls{&c.execPluginCalls},
		ExecPluginPolicyCalls:        countPolicy{&c.execPluginPolicyCalls},
		RequestRetry:                 countRetry{&c.requestRetry},
		TransportCacheEntries:        countTransportCache{&c.transportCacheEntries},
		TransportCreateCalls:         countTransportCreateCalls{&c.transportCreateCalls},
		TransportCAReloads:           countTransportCAReloads{&c.transportCAReloads},
		TransportCertRotationGCCalls: countTransportCertRotationGCCalls{&c.transportCertRotationGCCalls},
		TransportCacheGCCalls:        countTransportCacheGCCalls{&c.transportCacheGCCalls},
		RegisterFn:                   func() { c.registerFn.Add(1) },
	}, c
}

// invokeAllHooks calls every exported metric hook exactly once.
func invokeAllHooks() {
	ctx := context.Background()
	now := time.Now()
	u := url.URL{Scheme: "https", Host: "example.com"}

	ClientCertExpiry.Set(&now)
	ClientCertRotationAge.Observe(time.Second)
	RequestLatency.Observe(ctx, "GET", u, time.Second)
	ResolverLatency.Observe(ctx, "example.com", time.Second)
	RequestSize.Observe(ctx, "GET", "example.com", 1)
	ResponseSize.Observe(ctx, "GET", "example.com", 1)
	RateLimiterLatency.Observe(ctx, "GET", u, time.Second)
	RequestResult.Increment(ctx, "200", "GET", "example.com")
	ExecPluginCalls.Increment(0, "no_error")
	ExecPluginPolicyCalls.Increment("allowed")
	RequestRetry.IncrementRetry(ctx, "429", "GET", "example.com")
	TransportCacheEntries.Observe(1)
	TransportCreateCalls.Increment("hit")
	TransportCAReloads.Increment("success", "reload")
	TransportCertRotationGCCalls.Increment()
	TransportCacheGCCalls.Increment("deleted")
}

// resetForTest returns the package to its pristine state and restores whatever
// was there when the test finishes, so these tests neither see nor leak the
// registrations any other test in this binary may have made.
func resetForTest(tb testing.TB) {
	tb.Helper()

	registerLock.Lock()
	var (
		origClientCertExpiry             = ClientCertExpiry
		origClientCertRotationAge        = ClientCertRotationAge
		origRequestLatency               = RequestLatency
		origResolverLatency              = ResolverLatency
		origRequestSize                  = RequestSize
		origResponseSize                 = ResponseSize
		origRateLimiterLatency           = RateLimiterLatency
		origRequestResult                = RequestResult
		origExecPluginCalls              = ExecPluginCalls
		origExecPluginPolicyCalls        = ExecPluginPolicyCalls
		origRequestRetry                 = RequestRetry
		origTransportCacheEntries        = TransportCacheEntries
		origTransportCreateCalls         = TransportCreateCalls
		origTransportCAReloads           = TransportCAReloads
		origTransportCertRotationGCCalls = TransportCertRotationGCCalls
		origTransportCacheGCCalls        = TransportCacheGCCalls

		origRegisterFns             = registerFns
		origEnsureRegisteredStarted = ensureRegisteredStarted.Load()
		origRegisterFnsPending      = registerFnsPending.Load()
	)

	ClientCertExpiry = noopExpiry{}
	ClientCertRotationAge = noopDuration{}
	RequestLatency = noopLatency{}
	ResolverLatency = noopResolverLatency{}
	RequestSize = noopSize{}
	ResponseSize = noopSize{}
	RateLimiterLatency = noopLatency{}
	RequestResult = noopResult{}
	ExecPluginCalls = noopCalls{}
	ExecPluginPolicyCalls = noopPolicy{}
	RequestRetry = noopRetry{}
	TransportCacheEntries = noopTransportCache{}
	TransportCreateCalls = noopTransportCreateCalls{}
	TransportCAReloads = noopTransportCAReloads{}
	TransportCertRotationGCCalls = noopTransportCertRotationGCCalls{}
	TransportCacheGCCalls = noopTransportCacheGCCalls{}

	registerFns = nil
	ensureRegisteredStarted.Store(false)
	registerFnsPending.Store(false)
	registerLock.Unlock()

	tb.Cleanup(func() {
		registerLock.Lock()
		defer registerLock.Unlock()

		ClientCertExpiry = origClientCertExpiry
		ClientCertRotationAge = origClientCertRotationAge
		RequestLatency = origRequestLatency
		ResolverLatency = origResolverLatency
		RequestSize = origRequestSize
		ResponseSize = origResponseSize
		RateLimiterLatency = origRateLimiterLatency
		RequestResult = origRequestResult
		ExecPluginCalls = origExecPluginCalls
		ExecPluginPolicyCalls = origExecPluginPolicyCalls
		RequestRetry = origRequestRetry
		TransportCacheEntries = origTransportCacheEntries
		TransportCreateCalls = origTransportCreateCalls
		TransportCAReloads = origTransportCAReloads
		TransportCertRotationGCCalls = origTransportCertRotationGCCalls
		TransportCacheGCCalls = origTransportCacheGCCalls

		registerFns = origRegisterFns
		ensureRegisteredStarted.Store(origEnsureRegisteredStarted)
		registerFnsPending.Store(origRegisterFnsPending)
	})
}

// TestRegisterSingleProviderStoredDirectly guards the hot path: with exactly
// one provider - the overwhelmingly common case - each hook must hold that
// provider's metric itself, with no fan-out wrapper to iterate on every
// request.
func TestRegisterSingleProviderStoredDirectly(t *testing.T) {
	resetForTest(t)

	opts, _ := newCountingOpts()
	Register(opts)

	stored := map[string][2]any{
		"ClientCertExpiry":             {ClientCertExpiry, opts.ClientCertExpiry},
		"ClientCertRotationAge":        {ClientCertRotationAge, opts.ClientCertRotationAge},
		"RequestLatency":               {RequestLatency, opts.RequestLatency},
		"ResolverLatency":              {ResolverLatency, opts.ResolverLatency},
		"RequestSize":                  {RequestSize, opts.RequestSize},
		"ResponseSize":                 {ResponseSize, opts.ResponseSize},
		"RateLimiterLatency":           {RateLimiterLatency, opts.RateLimiterLatency},
		"RequestResult":                {RequestResult, opts.RequestResult},
		"ExecPluginCalls":              {ExecPluginCalls, opts.ExecPluginCalls},
		"ExecPluginPolicyCalls":        {ExecPluginPolicyCalls, opts.ExecPluginPolicyCalls},
		"RequestRetry":                 {RequestRetry, opts.RequestRetry},
		"TransportCacheEntries":        {TransportCacheEntries, opts.TransportCacheEntries},
		"TransportCreateCalls":         {TransportCreateCalls, opts.TransportCreateCalls},
		"TransportCAReloads":           {TransportCAReloads, opts.TransportCAReloads},
		"TransportCertRotationGCCalls": {TransportCertRotationGCCalls, opts.TransportCertRotationGCCalls},
		"TransportCacheGCCalls":        {TransportCacheGCCalls, opts.TransportCacheGCCalls},
	}
	for name, pair := range stored {
		if got, want := pair[0], pair[1]; got != want {
			t.Errorf("%s = %#v, want the registered metric itself (%#v); a single provider must not be wrapped", name, got, want)
		}
	}
}

// TestRegisterTwoProvidersBothObserve is the bug this change fixes: before it,
// the second registrant was silently dropped.
func TestRegisterTwoProvidersBothObserve(t *testing.T) {
	resetForTest(t)

	optsA, a := newCountingOpts()
	optsB, b := newCountingOpts()
	Register(optsA)
	Register(optsB)

	invokeAllHooks()

	for name, got := range a.metrics() {
		if got != 1 {
			t.Errorf("first provider: %s observed %d times, want 1", name, got)
		}
	}
	for name, got := range b.metrics() {
		if got != 1 {
			t.Errorf("second provider: %s observed %d times, want 1", name, got)
		}
	}
}

// TestRegisterThreeProvidersStayFlat checks that repeated registration extends
// one fan-out rather than nesting them, which would make dispatch cost grow
// with registration depth.
func TestRegisterThreeProvidersStayFlat(t *testing.T) {
	resetForTest(t)

	optsA, a := newCountingOpts()
	optsB, b := newCountingOpts()
	optsC, c := newCountingOpts()
	Register(optsA)
	Register(optsB)
	Register(optsC)

	fanout, ok := RequestLatency.(multiLatency)
	if !ok {
		t.Fatalf("RequestLatency is %T, want multiLatency", RequestLatency)
	}
	if len(fanout) != 3 {
		t.Errorf("fan-out holds %d metrics, want 3 (nested rather than flattened?)", len(fanout))
	}

	invokeAllHooks()
	for i, counter := range []*counters{a, b, c} {
		if got := counter.requestLatency.Load(); got != 1 {
			t.Errorf("provider %d: RequestLatency observed %d times, want 1", i, got)
		}
	}
}

func TestRegisterNilFieldsIgnored(t *testing.T) {
	resetForTest(t)

	c := &counters{}
	Register(RegisterOpts{RequestResult: countResult{&c.requestResult}})

	if _, ok := RequestResult.(countResult); !ok {
		t.Errorf("RequestResult = %T, want the registered countResult", RequestResult)
	}

	// Every other hook must be untouched, still holding its noop default.
	untouched := map[string]any{
		"ClientCertExpiry":             ClientCertExpiry,
		"ClientCertRotationAge":        ClientCertRotationAge,
		"RequestLatency":               RequestLatency,
		"ResolverLatency":              ResolverLatency,
		"RequestSize":                  RequestSize,
		"ResponseSize":                 ResponseSize,
		"RateLimiterLatency":           RateLimiterLatency,
		"ExecPluginCalls":              ExecPluginCalls,
		"ExecPluginPolicyCalls":        ExecPluginPolicyCalls,
		"RequestRetry":                 RequestRetry,
		"TransportCacheEntries":        TransportCacheEntries,
		"TransportCreateCalls":         TransportCreateCalls,
		"TransportCAReloads":           TransportCAReloads,
		"TransportCertRotationGCCalls": TransportCertRotationGCCalls,
		"TransportCacheGCCalls":        TransportCacheGCCalls,
	}
	for name, got := range untouched {
		switch got.(type) {
		case noopExpiry, noopDuration, noopLatency, noopResolverLatency, noopSize,
			noopResult, noopCalls, noopPolicy, noopRetry, noopTransportCache,
			noopTransportCreateCalls, noopTransportCAReloads,
			noopTransportCertRotationGCCalls, noopTransportCacheGCCalls:
		default:
			t.Errorf("%s = %T, want its noop default; a nil field must not register anything", name, got)
		}
	}
}

// TestRegisterPreservesDirectAssignment covers callers that assign the exported
// variables directly, which several in-tree tests and downstream projects do.
func TestRegisterPreservesDirectAssignment(t *testing.T) {
	resetForTest(t)

	direct := &counters{}
	RequestLatency = countLatency{&direct.requestLatency}

	opts, registered := newCountingOpts()
	Register(opts)

	RequestLatency.Observe(context.Background(), "GET", url.URL{}, time.Second)

	if got := direct.requestLatency.Load(); got != 1 {
		t.Errorf("directly assigned metric observed %d times, want 1; Register must not clobber it", got)
	}
	if got := registered.requestLatency.Load(); got != 1 {
		t.Errorf("registered metric observed %d times, want 1", got)
	}
}

// TestEnsureRegisteredFansOutAndStaysLazy pins the property that is easiest to
// break: RegisterFn must not fire at Register time, because adapters install it
// precisely so registration happens once feature gates are set.
func TestEnsureRegisteredFansOutAndStaysLazy(t *testing.T) {
	resetForTest(t)

	optsA, a := newCountingOpts()
	optsB, b := newCountingOpts()
	Register(optsA)
	Register(optsB)

	if got := a.registerFn.Load(); got != 0 {
		t.Errorf("first RegisterFn ran %d times before EnsureRegistered, want 0", got)
	}
	if got := b.registerFn.Load(); got != 0 {
		t.Errorf("second RegisterFn ran %d times before EnsureRegistered, want 0", got)
	}

	EnsureRegistered()

	if got := a.registerFn.Load(); got != 1 {
		t.Errorf("first RegisterFn ran %d times, want 1", got)
	}
	if got := b.registerFn.Load(); got != 1 {
		t.Errorf("second RegisterFn ran %d times, want 1", got)
	}

	EnsureRegistered()
	EnsureRegistered()

	if got := a.registerFn.Load(); got != 1 {
		t.Errorf("first RegisterFn ran %d times after repeat calls, want 1", got)
	}
	if got := b.registerFn.Load(); got != 1 {
		t.Errorf("second RegisterFn ran %d times after repeat calls, want 1", got)
	}
}

// TestEnsureRegisteredLateRegistrant covers a provider that registers after the
// first client was already built: nothing guarantees another EnsureRegistered
// call, so Register has to run the callback itself.
func TestEnsureRegisteredLateRegistrant(t *testing.T) {
	resetForTest(t)

	optsA, a := newCountingOpts()
	Register(optsA)
	EnsureRegistered()

	optsB, b := newCountingOpts()
	Register(optsB)

	if got := b.registerFn.Load(); got != 1 {
		t.Errorf("late RegisterFn ran %d times, want 1; it would otherwise never run", got)
	}
	if got := a.registerFn.Load(); got != 1 {
		t.Errorf("earlier RegisterFn ran %d times, want 1", got)
	}

	EnsureRegistered()

	if got := a.registerFn.Load(); got != 1 {
		t.Errorf("earlier RegisterFn ran %d times after a further EnsureRegistered, want 1", got)
	}
	if got := b.registerFn.Load(); got != 1 {
		t.Errorf("late RegisterFn ran %d times after a further EnsureRegistered, want 1", got)
	}
}

func TestEnsureRegisteredWithoutRegistrants(t *testing.T) {
	resetForTest(t)

	// Must not panic, and must stay cheap: nothing was ever registered.
	EnsureRegistered()
	EnsureRegistered()
}

// TestRegisterConcurrent checks that concurrent registrants do not lose each
// other, which is the failure the write lock exists to prevent. It deliberately
// does not observe while registering: the read path is documented as
// unsynchronized, with registration expected to happen before any client is
// built.
func TestRegisterConcurrent(t *testing.T) {
	resetForTest(t)

	const providers = 20
	all := make([]*counters, providers)

	var wg sync.WaitGroup
	for i := range providers {
		opts, c := newCountingOpts()
		all[i] = c
		wg.Add(1)
		go func() {
			defer wg.Done()
			Register(opts)
		}()
	}
	wg.Wait()

	invokeAllHooks()
	EnsureRegistered()

	for i, c := range all {
		for name, got := range c.metrics() {
			if got != 1 {
				t.Errorf("provider %d: %s observed %d times, want 1", i, name, got)
			}
		}
		if got := c.registerFn.Load(); got != 1 {
			t.Errorf("provider %d: RegisterFn ran %d times, want 1", i, got)
		}
	}
}

func BenchmarkRequestLatencyOneProvider(b *testing.B) {
	resetForTest(b)

	opts, _ := newCountingOpts()
	Register(opts)

	ctx := context.Background()
	u := url.URL{Scheme: "https", Host: "example.com"}
	b.ResetTimer()
	for range b.N {
		RequestLatency.Observe(ctx, "GET", u, time.Second)
	}
}

func BenchmarkRequestLatencyTwoProviders(b *testing.B) {
	resetForTest(b)

	optsA, _ := newCountingOpts()
	optsB, _ := newCountingOpts()
	Register(optsA)
	Register(optsB)

	ctx := context.Background()
	u := url.URL{Scheme: "https", Host: "example.com"}
	b.ResetTimer()
	for range b.N {
		RequestLatency.Observe(ctx, "GET", u, time.Second)
	}
}

func BenchmarkEnsureRegisteredSteadyState(b *testing.B) {
	resetForTest(b)

	opts, _ := newCountingOpts()
	Register(opts)
	EnsureRegistered()

	b.ResetTimer()
	for range b.N {
		EnsureRegistered()
	}
}

// TestEnsureRegisteredLateRegistrantAfterEmptyEnsure covers a provider that
// registers a callback after EnsureRegistered has already run with nothing
// installed, which is what happens when a client is built before any provider
// registers. The callback still has to run: nothing guarantees a further
// EnsureRegistered call.
func TestEnsureRegisteredLateRegistrantAfterEmptyEnsure(t *testing.T) {
	resetForTest(t)

	EnsureRegistered()

	opts, c := newCountingOpts()
	Register(opts)

	if got := c.registerFn.Load(); got != 1 {
		t.Errorf("late RegisterFn ran %d times, want 1; EnsureRegistered had already run, so no further call is guaranteed", got)
	}
}

// TestRegisterNilFieldsDoNotWriteHooks guards against Register storing into the
// exported hooks when it has nothing to add. They are read from the request
// path without synchronization, so writing back an identical value is still a
// data race. Only meaningful under -race.
func TestRegisterNilFieldsDoNotWriteHooks(t *testing.T) {
	resetForTest(t)

	opts, _ := newCountingOpts()
	Register(opts)

	ctx := context.Background()
	u := url.URL{Scheme: "https", Host: "example.com"}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				RequestLatency.Observe(ctx, "GET", u, time.Second)
			}
		}
	}()

	for range 100 {
		Register(RegisterOpts{})
	}
	close(stop)
	wg.Wait()
}
