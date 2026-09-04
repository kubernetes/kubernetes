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
	"time"
)

// This file lets more than one provider register the same metric hook. Each
// multi* type fans an observation out to every metric it holds, and each
// combine* function folds a newly registered metric into whatever the hook
// currently holds:
//
//   - a nil addition leaves the hook untouched, so nil fields in RegisterOpts
//     are ignored exactly as they were before;
//   - registering into an unset hook (the noop default, or nil) stores the
//     metric directly, so the single-provider case - by far the common one -
//     carries no wrapper and no loop on the read path;
//   - registering into a hook that already fans out extends it, keeping the
//     fan-out flat rather than nesting;
//   - anything else, including a value assigned directly to the exported
//     variable by a caller, becomes a two-element fan-out, so such an
//     assignment is preserved rather than dropped.
//
// Extending a fan-out always copies: a value already published to the exported
// variable may be being ranged over by an in-flight request, so its backing
// array must never be written in place.
//
// Registering the same metric twice records the observation twice. Suppressing
// that would mean comparing metrics for equality, which panics when the dynamic
// type is uncomparable, so it is deliberately not attempted.

type multiExpiry []ExpiryMetric

func (m multiExpiry) Set(expiry *time.Time) {
	for _, metric := range m {
		metric.Set(expiry)
	}
}

func combineExpiry(cur, add ExpiryMetric) ExpiryMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopExpiry:
		return add
	case multiExpiry:
		next := make(multiExpiry, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiExpiry{cur, add}
	}
}

type multiDuration []DurationMetric

func (m multiDuration) Observe(duration time.Duration) {
	for _, metric := range m {
		metric.Observe(duration)
	}
}

func combineDuration(cur, add DurationMetric) DurationMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopDuration:
		return add
	case multiDuration:
		next := make(multiDuration, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiDuration{cur, add}
	}
}

type multiLatency []LatencyMetric

func (m multiLatency) Observe(ctx context.Context, verb string, u url.URL, latency time.Duration) {
	for _, metric := range m {
		metric.Observe(ctx, verb, u, latency)
	}
}

func combineLatency(cur, add LatencyMetric) LatencyMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopLatency:
		return add
	case multiLatency:
		next := make(multiLatency, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiLatency{cur, add}
	}
}

type multiResolverLatency []ResolverLatencyMetric

func (m multiResolverLatency) Observe(ctx context.Context, host string, latency time.Duration) {
	for _, metric := range m {
		metric.Observe(ctx, host, latency)
	}
}

func combineResolverLatency(cur, add ResolverLatencyMetric) ResolverLatencyMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopResolverLatency:
		return add
	case multiResolverLatency:
		next := make(multiResolverLatency, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiResolverLatency{cur, add}
	}
}

type multiSize []SizeMetric

func (m multiSize) Observe(ctx context.Context, verb string, host string, size float64) {
	for _, metric := range m {
		metric.Observe(ctx, verb, host, size)
	}
}

func combineSize(cur, add SizeMetric) SizeMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopSize:
		return add
	case multiSize:
		next := make(multiSize, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiSize{cur, add}
	}
}

type multiResult []ResultMetric

func (m multiResult) Increment(ctx context.Context, code string, method string, host string) {
	for _, metric := range m {
		metric.Increment(ctx, code, method, host)
	}
}

func combineResult(cur, add ResultMetric) ResultMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopResult:
		return add
	case multiResult:
		next := make(multiResult, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiResult{cur, add}
	}
}

type multiCalls []CallsMetric

func (m multiCalls) Increment(exitCode int, callStatus string) {
	for _, metric := range m {
		metric.Increment(exitCode, callStatus)
	}
}

func combineCalls(cur, add CallsMetric) CallsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopCalls:
		return add
	case multiCalls:
		next := make(multiCalls, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiCalls{cur, add}
	}
}

type multiPolicy []PolicyCallsMetric

func (m multiPolicy) Increment(status string) {
	for _, metric := range m {
		metric.Increment(status)
	}
}

func combinePolicy(cur, add PolicyCallsMetric) PolicyCallsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopPolicy:
		return add
	case multiPolicy:
		next := make(multiPolicy, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiPolicy{cur, add}
	}
}

type multiRetry []RetryMetric

func (m multiRetry) IncrementRetry(ctx context.Context, code string, method string, host string) {
	for _, metric := range m {
		metric.IncrementRetry(ctx, code, method, host)
	}
}

func combineRetry(cur, add RetryMetric) RetryMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopRetry:
		return add
	case multiRetry:
		next := make(multiRetry, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiRetry{cur, add}
	}
}

type multiTransportCache []TransportCacheMetric

func (m multiTransportCache) Observe(value int) {
	for _, metric := range m {
		metric.Observe(value)
	}
}

func combineTransportCache(cur, add TransportCacheMetric) TransportCacheMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopTransportCache:
		return add
	case multiTransportCache:
		next := make(multiTransportCache, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiTransportCache{cur, add}
	}
}

type multiTransportCreateCalls []TransportCreateCallsMetric

func (m multiTransportCreateCalls) Increment(result string) {
	for _, metric := range m {
		metric.Increment(result)
	}
}

func combineTransportCreateCalls(cur, add TransportCreateCallsMetric) TransportCreateCallsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopTransportCreateCalls:
		return add
	case multiTransportCreateCalls:
		next := make(multiTransportCreateCalls, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiTransportCreateCalls{cur, add}
	}
}

type multiTransportCAReloads []TransportCAReloadsMetric

func (m multiTransportCAReloads) Increment(result, reason string) {
	for _, metric := range m {
		metric.Increment(result, reason)
	}
}

func combineTransportCAReloads(cur, add TransportCAReloadsMetric) TransportCAReloadsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopTransportCAReloads:
		return add
	case multiTransportCAReloads:
		next := make(multiTransportCAReloads, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiTransportCAReloads{cur, add}
	}
}

type multiTransportCertRotationGCCalls []TransportCertRotationGCCallsMetric

func (m multiTransportCertRotationGCCalls) Increment() {
	for _, metric := range m {
		metric.Increment()
	}
}

func combineTransportCertRotationGCCalls(cur, add TransportCertRotationGCCallsMetric) TransportCertRotationGCCallsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopTransportCertRotationGCCalls:
		return add
	case multiTransportCertRotationGCCalls:
		next := make(multiTransportCertRotationGCCalls, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiTransportCertRotationGCCalls{cur, add}
	}
}

type multiTransportCacheGCCalls []TransportCacheGCCallsMetric

func (m multiTransportCacheGCCalls) Increment(result string) {
	for _, metric := range m {
		metric.Increment(result)
	}
}

func combineTransportCacheGCCalls(cur, add TransportCacheGCCallsMetric) TransportCacheGCCallsMetric {
	if add == nil {
		return cur
	}
	switch c := cur.(type) {
	case nil, noopTransportCacheGCCalls:
		return add
	case multiTransportCacheGCCalls:
		next := make(multiTransportCacheGCCalls, len(c)+1)
		copy(next, c)
		next[len(c)] = add
		return next
	default:
		return multiTransportCacheGCCalls{cur, add}
	}
}
