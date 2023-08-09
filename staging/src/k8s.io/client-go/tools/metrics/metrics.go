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
	"time"
)

var registerMetrics sync.Once

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
// partitioned by the result of the cache: hit, miss, uncacheable
type TransportCreateCallsMetric interface {
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
	// RequestRetry is the retry metric that tracks the number of
	// retries sent to the server.
	RequestRetry RetryMetric = noopRetry{}
	// TransportCacheEntries is the metric that tracks the number of entries in the
	// internal transport cache.
	TransportCacheEntries TransportCacheMetric = noopTransportCache{}
	// TransportCreateCalls is the metric that counts the number of times a new transport
	// is created
	TransportCreateCalls TransportCreateCallsMetric = noopTransportCreateCalls{}
)

// RegisterOpts contains all the metrics to register. Metrics may be nil.
type RegisterOpts struct {
	ClientCertExpiry      ExpiryMetric
	ClientCertRotationAge DurationMetric
	RequestLatency        LatencyMetric
	ResolverLatency       ResolverLatencyMetric
	RequestSize           SizeMetric
	ResponseSize          SizeMetric
	RateLimiterLatency    LatencyMetric
	RequestResult         ResultMetric
	ExecPluginCalls       CallsMetric
	RequestRetry          RetryMetric
	TransportCacheEntries TransportCacheMetric
	TransportCreateCalls  TransportCreateCallsMetric
}

// Register registers metrics for the rest client to use. This can
// only be called once.
func Register(opts RegisterOpts) {
	registerMetrics.Do(func() {
		if opts.ClientCertExpiry != nil {
			ClientCertExpiry = opts.ClientCertExpiry
		}
		if opts.ClientCertRotationAge != nil {
			ClientCertRotationAge = opts.ClientCertRotationAge
		}
		if opts.RequestLatency != nil {
			RequestLatency = opts.RequestLatency
		}
		if opts.ResolverLatency != nil {
			ResolverLatency = opts.ResolverLatency
		}
		if opts.RequestSize != nil {
			RequestSize = opts.RequestSize
		}
		if opts.ResponseSize != nil {
			ResponseSize = opts.ResponseSize
		}
		if opts.RateLimiterLatency != nil {
			RateLimiterLatency = opts.RateLimiterLatency
		}
		if opts.RequestResult != nil {
			RequestResult = opts.RequestResult
		}
		if opts.ExecPluginCalls != nil {
			ExecPluginCalls = opts.ExecPluginCalls
		}
		if opts.RequestRetry != nil {
			RequestRetry = opts.RequestRetry
		}
		if opts.TransportCacheEntries != nil {
			TransportCacheEntries = opts.TransportCacheEntries
		}
		if opts.TransportCreateCalls != nil {
			TransportCreateCalls = opts.TransportCreateCalls
		}
	})
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

type noopRetry struct{}

func (noopRetry) IncrementRetry(context.Context, string, string, string) {}

type noopTransportCache struct{}

func (noopTransportCache) Observe(int) {}

type noopTransportCreateCalls struct{}

func (noopTransportCreateCalls) Increment(string) {}
