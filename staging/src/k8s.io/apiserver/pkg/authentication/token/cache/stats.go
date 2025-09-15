/*
Copyright 2019 The Kubernetes Authors.

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

package cache

import (
	"context"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	requestLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      "authentication",
			Subsystem:      "token_cache",
			Name:           "request_duration_seconds",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"},
	)
	requestCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      "authentication",
			Subsystem:      "token_cache",
			Name:           "request_total",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"},
	)
	fetchCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      "authentication",
			Subsystem:      "token_cache",
			Name:           "fetch_total",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"},
	)
	activeFetchCount = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      "authentication",
			Subsystem:      "token_cache",
			Name:           "active_fetch_count",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"},
	)
)

func init() {
	legacyregistry.MustRegister(
		requestLatency,
		requestCount,
		fetchCount,
		activeFetchCount,
	)
}

const (
	hitTag  = "hit"
	missTag = "miss"

	fetchFailedTag = "error"
	fetchOkTag     = "ok"

	fetchInFlightTag = "in_flight"
	fetchBlockedTag  = "blocked"
)

type statsCollector struct{}

var stats = statsCollector{}

func (statsCollector) authenticating(ctx context.Context) func(hit bool) {
	start := time.Now()
	return func(hit bool) {
		var tag string
		if hit {
			tag = hitTag
		} else {
			tag = missTag
		}

		latency := time.Since(start)

		requestCount.WithContext(ctx).WithLabelValues(tag).Inc()
		requestLatency.WithContext(ctx).WithLabelValues(tag).Observe(float64(latency.Milliseconds()) / 1000)
	}
}

func (statsCollector) blocking(ctx context.Context) func() {
	activeFetchCount.WithContext(ctx).WithLabelValues(fetchBlockedTag).Inc()
	return activeFetchCount.WithContext(ctx).WithLabelValues(fetchBlockedTag).Dec
}

func (statsCollector) fetching(ctx context.Context) func(ok bool) {
	activeFetchCount.WithContext(ctx).WithLabelValues(fetchInFlightTag).Inc()
	return func(ok bool) {
		var tag string
		if ok {
			tag = fetchOkTag
		} else {
			tag = fetchFailedTag
		}

		fetchCount.WithContext(ctx).WithLabelValues(tag).Inc()

		activeFetchCount.WithContext(ctx).WithLabelValues(fetchInFlightTag).Dec()
	}
}
