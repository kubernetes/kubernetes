/*
Copyright 2017 The Kubernetes Authors.

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

package winkernel

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const kubeProxySubsystem = "kubeproxy"

var (
	// SyncProxyRulesLatency is the latency of one round of kube-proxy syncing proxy rules.
	SyncProxyRulesLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: kubeProxySubsystem,
			Name:      "sync_proxy_rules_latency_microseconds",
			Help:      "SyncProxyRules latency",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)

	// SyncProxyRulesCount is the number of times kube-proxy has synced proxy rules.
	SyncProxyRulesCount = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: kubeProxySubsystem,
			Name:      "sync_proxy_rules_count",
			Help:      "SyncProxyRules count",
		},
	)
)

var registerMetricsOnce sync.Once

func RegisterMetrics() {
	registerMetricsOnce.Do(func() {
		prometheus.MustRegister(SyncProxyRulesLatency)
		prometheus.MustRegister(SyncProxyRulesCount)
	})
}

// Gets the time since the specified start in microseconds.
func sinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
