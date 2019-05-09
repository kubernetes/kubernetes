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

package metrics

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
			Name:      "sync_proxy_rules_duration_seconds",
			Help:      "SyncProxyRules latency in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)

	// DeprecatedSyncProxyRulesLatency is the latency of one round of kube-proxy syncing proxy rules.
	DeprecatedSyncProxyRulesLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: kubeProxySubsystem,
			Name:      "sync_proxy_rules_latency_microseconds",
			Help:      "(Deprecated) SyncProxyRules latency in microseconds",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)

	// NetworkProgrammingLatency is defined as the time it took to program the network - from the time
	// the service or pod has changed to the time the change was propagated and the proper kube-proxy
	// rules were synced. Exported for each endpoints object that were part of the rules sync.
	// See https://github.com/kubernetes/community/blob/master/sig-scalability/slos/network_programming_latency.md
	// Note that the metrics is partially based on the time exported by the endpoints controller on
	// the master machine. The measurement may be inaccurate if there is a clock drift between the
	// node and master machine.
	NetworkProgrammingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: kubeProxySubsystem,
			Name:      "network_programming_duration_seconds",
			Help:      "In Cluster Network Programming Latency in seconds",
			// TODO(mm4tt): Reevaluate buckets before 1.14 release.
			// The last bucket will be [0.001s*2^20 ~= 17min, +inf)
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 20),
		},
	)
)

var registerMetricsOnce sync.Once

// RegisterMetrics registers kube-proxy metrics.
func RegisterMetrics() {
	registerMetricsOnce.Do(func() {
		prometheus.MustRegister(SyncProxyRulesLatency)
		prometheus.MustRegister(DeprecatedSyncProxyRulesLatency)
		prometheus.MustRegister(NetworkProgrammingLatency)
	})
}

// SinceInMicroseconds gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
