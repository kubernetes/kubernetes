/*
Copyright 2015 The Kubernetes Authors.

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

// Package metrics provides abstractions for registering which metrics
// to record.
package metrics

import (
	"context"
	"net/url"
	"sync"
	"sync/atomic"
	"time"
)

// registerFnEntry pairs a RegisterFn installed via RegisterOpts with its own
// sync.Once, so each callback runs exactly once even when it is registered
// after EnsureRegistered has already run. Running one more than once would
// re-register with legacyregistry, which panics on duplicates.
type registerFnEntry struct {
	once sync.Once
	fn   func()
}

var (
	// registerLock guards the write path of Register and the registration
	// state below. The exported metric hooks are read from the request path
	// without synchronization, so Register must be called before any rest
	// client is constructed; see the Register documentation.
	registerLock sync.Mutex

	// registerFns holds every callback installed via RegisterOpts.RegisterFn.
	// Adapter packages (e.g. k8s.io/component-base/metrics/prometheus/restclient)
	// install these in their init() so that the actual registration with
	// legacyregistry — and the metric Create() that reads feature-gate-derived
	// options like NativeHistograms — happens at runtime rather than at init()
	// time. See EnsureRegistered for the caller-side contract.
	//
	// It is only ever replaced, never appended to in place, so EnsureRegistered
	// can range over a snapshot of it without holding registerLock.
	registerFns []*registerFnEntry

	// ensureRegisteredStarted records whether EnsureRegistered has run at least
	// once. Once it has, a callback registered afterwards is invoked
	// immediately rather than waiting for a call that may never come. It is
	// atomic because the lock-free fast path in EnsureRegistered has to record
	// it without taking registerLock.
	ensureRegisteredStarted atomic.Bool

	// registerFnsPending is the lock-free fast path for EnsureRegistered, which
	// adapters call on every observation. Its zero value, false, means there is
	// nothing left to invoke.
	registerFnsPending atomic.Bool
)

// EnsureRegistered invokes the callbacks installed via RegisterOpts.RegisterFn,
// each exactly once. Callers should treat it as idempotent; subsequent calls are
// effectively free.
//
// New public constructors or entry points for packages in client-go that create
// a REST client, HTTP transport, or credential provider should invoke EnsureRegistered()
// at the very beginning of the function. Adapter Observe methods
// also call EnsureRegistered(), so no observations are lost if a constructor
// forgets to invoke it, but if invoked, the entrypoint call shifts registration from
// "first-observation" to "first client construction", meaning the metric
// series is visible to Promteheus scrapes from process startup rather than
// appearing only after the first request.
//
// A RegisterFn must not itself call Register or EnsureRegistered.
func EnsureRegistered() {
	// Record that registration time has passed even when there is nothing to
	// invoke, so that a provider registering later knows no further
	// EnsureRegistered call is guaranteed. Tested before storing to keep the
	// steady-state path read-only rather than writing the same cache line from
	// every request on every core.
	if !ensureRegisteredStarted.Load() {
		ensureRegisteredStarted.Store(true)
	}

	if !registerFnsPending.Load() {
		return
	}

	registerLock.Lock()
	entries := registerFns
	registerLock.Unlock()

	// Callbacks run outside registerLock: they call back into metric
	// registries and must not be serialized behind the registration state.
	for _, entry := range entries {
		entry.once.Do(entry.fn)
	}

	registerLock.Lock()
	// Anything registered while the callbacks above were running is not
	// covered by this pass, so leave the fast path armed for it.
	if len(registerFns) == len(entries) {
		registerFnsPending.Store(false)
	}
	registerLock.Unlock()
}

// DurationMetric is a measurement of some amount of time.
type DurationMetric interface {
	Observe(duration time.Duration)
}

// ExpiryMetric sets some time of expiry. If nil, assume not relevant.
type ExpiryMetric interface {
	Set(expiry *time.Time)
}

// LatencyMetric observes client latency partitioned by verb and url.
type LatencyMetric interface {
	Observe(ctx context.Context, verb string, u url.URL, latency time.Duration)
}

type ResolverLatencyMetric interface {
	Observe(ctx context.Context, host string, latency time.Duration)
}

// SizeMetric observes client response size partitioned by verb and host.
type SizeMetric interface {
	Observe(ctx context.Context, verb string, host string, size float64)
}

// ResultMetric counts response codes partitioned by method and host.
type ResultMetric interface {
	Increment(ctx context.Context, code string, method string, host string)
}

// CallsMetric counts calls that take place for a specific exec plugin.
type CallsMetric interface {
	// Increment increments a counter per exitCode and callStatus.
	Increment(exitCode int, callStatus string)
}

// CallsMetric counts the success or failure of execution for exec plugins.
type PolicyCallsMetric interface {
	// Increment increments a counter per status { "allowed", "denied" }
	Increment(status string)
}

// RetryMetric counts the number of retries sent to the server
// partitioned by code, method, and host.
type RetryMetric interface {
	IncrementRetry(ctx context.Context, code string, method string, host string)
}

// TransportCacheMetric shows the number of entries in the internal transport cache
type TransportCacheMetric interface {
	Observe(value int)
}

// TransportCreateCallsMetric counts the number of times a transport is created
// partitioned by the result of the cache: hit, miss, miss-gc, uncacheable
type TransportCreateCallsMetric interface {
	Increment(result string)
}

// TransportCAReloadsMetric counts the number of times a CA reload is attempted,
// partitioned by the result and reason.
type TransportCAReloadsMetric interface {
	Increment(result, reason string)
}

// TransportCertRotationGCCallsMetric counts the number of times a cert rotation
// goroutine cancel func is called via GC cleanup.
type TransportCertRotationGCCallsMetric interface {
	Increment()
}

// TransportCacheGCCallsMetric counts the number of times a GC cleanup
// attempts to delete a cache entry, partitioned by the result: deleted, skipped.
type TransportCacheGCCallsMetric interface {
	Increment(result string)
}

var (
	// ClientCertExpiry is the expiry time of a client certificate
	ClientCertExpiry ExpiryMetric = noopExpiry{}
	// ClientCertRotationAge is the age of a certificate that has just been rotated.
	ClientCertRotationAge DurationMetric = noopDuration{}
	// RequestLatency is the latency metric that rest clients will update.
	RequestLatency LatencyMetric = noopLatency{}
	// ResolverLatency is the latency metric that DNS resolver will update
	ResolverLatency ResolverLatencyMetric = noopResolverLatency{}
	// RequestSize is the request size metric that rest clients will update.
	RequestSize SizeMetric = noopSize{}
	// ResponseSize is the response size metric that rest clients will update.
	ResponseSize SizeMetric = noopSize{}
	// RateLimiterLatency is the client side rate limiter latency metric.
	RateLimiterLatency LatencyMetric = noopLatency{}
	// RequestResult is the result metric that rest clients will update.
	RequestResult ResultMetric = noopResult{}
	// ExecPluginCalls is the number of calls made to an exec plugin, partitioned by
	// exit code and call status.
	ExecPluginCalls CallsMetric = noopCalls{}
	// ExecPluginPolicyCalls is the number of plugin policy check calls, partitioned
	// by {"allowed", "denied"}
	ExecPluginPolicyCalls PolicyCallsMetric = noopPolicy{}
	// RequestRetry is the retry metric that tracks the number of
	// retries sent to the server.
	RequestRetry RetryMetric = noopRetry{}
	// TransportCacheEntries is the metric that tracks the number of entries in the
	// internal transport cache.
	TransportCacheEntries TransportCacheMetric = noopTransportCache{}
	// TransportCreateCalls is the metric that counts the number of times a new transport
	// is created
	TransportCreateCalls TransportCreateCallsMetric = noopTransportCreateCalls{}
	// TransportCAReloads is the metric that counts the number of times a CA reload is attempted
	TransportCAReloads TransportCAReloadsMetric = noopTransportCAReloads{}
	// TransportCertRotationGCCalls counts the number of times a cert rotation goroutine
	// cancel func is called via GC cleanup
	TransportCertRotationGCCalls TransportCertRotationGCCallsMetric = noopTransportCertRotationGCCalls{}
	// TransportCacheGCCalls counts the number of times a GC cleanup attempts
	// to delete a transport cache entry, partitioned by result: deleted, skipped.
	TransportCacheGCCalls TransportCacheGCCallsMetric = noopTransportCacheGCCalls{}
)

// RegisterOpts contains all the metrics to register. Metrics may be nil.
type RegisterOpts struct {
	ClientCertExpiry             ExpiryMetric
	ClientCertRotationAge        DurationMetric
	RequestLatency               LatencyMetric
	ResolverLatency              ResolverLatencyMetric
	RequestSize                  SizeMetric
	ResponseSize                 SizeMetric
	RateLimiterLatency           LatencyMetric
	RequestResult                ResultMetric
	ExecPluginCalls              CallsMetric
	ExecPluginPolicyCalls        PolicyCallsMetric
	RequestRetry                 RetryMetric
	TransportCacheEntries        TransportCacheMetric
	TransportCreateCalls         TransportCreateCallsMetric
	TransportCAReloads           TransportCAReloadsMetric
	TransportCertRotationGCCalls TransportCertRotationGCCallsMetric
	TransportCacheGCCalls        TransportCacheGCCallsMetric

	// RegisterFn, if non-nil, is invoked exactly once by EnsureRegistered().
	// before the first rest client is constructed. Adapters use this to defer
	// registrations that depend on runtime state (eg., feature gates read my metric
	// Create() without changing the import side contract of the adapter package.)
	//
	// Callbacks from multiple Register calls are all retained and each is
	// invoked exactly once. A callback registered after EnsureRegistered has
	// already run is invoked by Register itself, since no further
	// EnsureRegistered call is guaranteed. It must not call Register or
	// EnsureRegistered.
	RegisterFn func()
}

// Register adds the metrics in opts to those the rest client and its transports
// update. It may be called by more than one provider: every registered metric
// receives every observation. Nil fields in opts are ignored.
//
// Earlier releases applied only the first call to Register and silently
// ignored every later caller.
//
// Register should be called before any REST client, HTTP transport, or
// credential provider is constructed, typically from an init function: the
// metrics it installs are read from the request path without synchronization,
// and values already captured by existing clients are not updated
// retroactively. Registering the same metric more than once records each
// observation once per registration.
func Register(opts RegisterOpts) {
	registerLock.Lock()

	// Each hook is written only when there is something to add. The combine
	// helpers already ignore a nil addition, but the assignment itself would
	// still be a write, and these are read from the request path without
	// synchronization.
	if opts.ClientCertExpiry != nil {
		ClientCertExpiry = combineExpiry(ClientCertExpiry, opts.ClientCertExpiry)
	}
	if opts.ClientCertRotationAge != nil {
		ClientCertRotationAge = combineDuration(ClientCertRotationAge, opts.ClientCertRotationAge)
	}
	if opts.RequestLatency != nil {
		RequestLatency = combineLatency(RequestLatency, opts.RequestLatency)
	}
	if opts.ResolverLatency != nil {
		ResolverLatency = combineResolverLatency(ResolverLatency, opts.ResolverLatency)
	}
	if opts.RequestSize != nil {
		RequestSize = combineSize(RequestSize, opts.RequestSize)
	}
	if opts.ResponseSize != nil {
		ResponseSize = combineSize(ResponseSize, opts.ResponseSize)
	}
	if opts.RateLimiterLatency != nil {
		RateLimiterLatency = combineLatency(RateLimiterLatency, opts.RateLimiterLatency)
	}
	if opts.RequestResult != nil {
		RequestResult = combineResult(RequestResult, opts.RequestResult)
	}
	if opts.ExecPluginCalls != nil {
		ExecPluginCalls = combineCalls(ExecPluginCalls, opts.ExecPluginCalls)
	}
	if opts.ExecPluginPolicyCalls != nil {
		ExecPluginPolicyCalls = combinePolicy(ExecPluginPolicyCalls, opts.ExecPluginPolicyCalls)
	}
	if opts.RequestRetry != nil {
		RequestRetry = combineRetry(RequestRetry, opts.RequestRetry)
	}
	if opts.TransportCacheEntries != nil {
		TransportCacheEntries = combineTransportCache(TransportCacheEntries, opts.TransportCacheEntries)
	}
	if opts.TransportCreateCalls != nil {
		TransportCreateCalls = combineTransportCreateCalls(TransportCreateCalls, opts.TransportCreateCalls)
	}
	if opts.TransportCAReloads != nil {
		TransportCAReloads = combineTransportCAReloads(TransportCAReloads, opts.TransportCAReloads)
	}
	if opts.TransportCertRotationGCCalls != nil {
		TransportCertRotationGCCalls = combineTransportCertRotationGCCalls(TransportCertRotationGCCalls, opts.TransportCertRotationGCCalls)
	}
	if opts.TransportCacheGCCalls != nil {
		TransportCacheGCCalls = combineTransportCacheGCCalls(TransportCacheGCCalls, opts.TransportCacheGCCalls)
	}

	fireNow := false
	if opts.RegisterFn != nil {
		// Copy rather than append in place: EnsureRegistered ranges over the
		// current slice without holding registerLock.
		next := make([]*registerFnEntry, len(registerFns)+1)
		copy(next, registerFns)
		next[len(registerFns)] = &registerFnEntry{fn: opts.RegisterFn}
		registerFns = next
		registerFnsPending.Store(true)

		// RegisterFn is deliberately not invoked here. Adapters install it so
		// that registration happens at first client construction, once
		// feature gates are set; firing it now would defeat that. Fire it
		// only if that moment has already passed, because no further
		// EnsureRegistered call is guaranteed.
		fireNow = ensureRegisteredStarted.Load()
	}
	registerLock.Unlock()

	if fireNow {
		EnsureRegistered()
	}
}

type noopDuration struct{}

func (noopDuration) Observe(time.Duration) {}

type noopExpiry struct{}

func (noopExpiry) Set(*time.Time) {}

type noopLatency struct{}

func (noopLatency) Observe(context.Context, string, url.URL, time.Duration) {}

type noopResolverLatency struct{}

func (n noopResolverLatency) Observe(ctx context.Context, host string, latency time.Duration) {
}

type noopSize struct{}

func (noopSize) Observe(context.Context, string, string, float64) {}

type noopResult struct{}

func (noopResult) Increment(context.Context, string, string, string) {}

type noopCalls struct{}

func (noopCalls) Increment(int, string) {}

type noopPolicy struct{}

func (noopPolicy) Increment(string) {}

type noopRetry struct{}

func (noopRetry) IncrementRetry(context.Context, string, string, string) {}

type noopTransportCache struct{}

func (noopTransportCache) Observe(int) {}

type noopTransportCreateCalls struct{}

func (noopTransportCreateCalls) Increment(string) {}

type noopTransportCAReloads struct{}

func (noopTransportCAReloads) Increment(result, reason string) {}

type noopTransportCertRotationGCCalls struct{}

func (noopTransportCertRotationGCCalls) Increment() {}

type noopTransportCacheGCCalls struct{}

func (noopTransportCacheGCCalls) Increment(string) {}
