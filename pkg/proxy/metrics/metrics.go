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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const kubeProxySubsystem = "kubeproxy"

var (
	// SyncProxyRulesLatency is the latency of one round of kube-proxy syncing proxy rules.
	SyncProxyRulesLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_duration_seconds",
			Help:           "SyncProxyRules latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// SyncProxyRulesLastTimestamp is the timestamp proxy rules were last
	// successfully synced.
	SyncProxyRulesLastTimestamp = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_last_timestamp_seconds",
			Help:           "The last time proxy rules were successfully synced",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// NetworkProgrammingLatency is defined as the time it took to program the network - from the time
	// the service or pod has changed to the time the change was propagated and the proper kube-proxy
	// rules were synced. Exported for each endpoints object that were part of the rules sync.
	// See https://github.com/kubernetes/community/blob/master/sig-scalability/slos/network_programming_latency.md
	// Note that the metrics is partially based on the time exported by the endpoints controller on
	// the master machine. The measurement may be inaccurate if there is a clock drift between the
	// node and master machine.
	NetworkProgrammingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: kubeProxySubsystem,
			Name:      "network_programming_duration_seconds",
			Help:      "In Cluster Network Programming Latency in seconds",
			Buckets: merge(
				metrics.LinearBuckets(0.25, 0.25, 2), // 0.25s, 0.50s
				metrics.LinearBuckets(1, 1, 59),      // 1s, 2s, 3s, ... 59s
				metrics.LinearBuckets(60, 5, 12),     // 60s, 65s, 70s, ... 115s
				metrics.LinearBuckets(120, 30, 7),    // 2min, 2.5min, 3min, ..., 5min
			),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// EndpointChangesPending is the number of pending endpoint changes that
	// have not yet been synced to the proxy.
	EndpointChangesPending = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_endpoint_changes_pending",
			Help:           "Pending proxy rules Endpoint changes",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// EndpointChangesTotal is the number of endpoint changes that the proxy
	// has seen.
	EndpointChangesTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_endpoint_changes_total",
			Help:           "Cumulative proxy rules Endpoint changes",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// ServiceChangesPending is the number of pending service changes that
	// have not yet been synced to the proxy.
	ServiceChangesPending = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_service_changes_pending",
			Help:           "Pending proxy rules Service changes",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// ServiceChangesTotal is the number of service changes that the proxy has
	// seen.
	ServiceChangesTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_service_changes_total",
			Help:           "Cumulative proxy rules Service changes",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// IptablesRestoreFailuresTotal is the number of iptables restore failures that the proxy has
	// seen.
	IptablesRestoreFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_iptables_restore_failures_total",
			Help:           "Cumulative proxy iptables restore failures",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var registerMetricsOnce sync.Once

// RegisterMetrics registers kube-proxy metrics.
func RegisterMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(SyncProxyRulesLatency)
		legacyregistry.MustRegister(SyncProxyRulesLastTimestamp)
		legacyregistry.MustRegister(NetworkProgrammingLatency)
		legacyregistry.MustRegister(EndpointChangesPending)
		legacyregistry.MustRegister(EndpointChangesTotal)
		legacyregistry.MustRegister(ServiceChangesPending)
		legacyregistry.MustRegister(ServiceChangesTotal)
		legacyregistry.MustRegister(IptablesRestoreFailuresTotal)
	})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

func merge(slices ...[]float64) []float64 {
	result := make([]float64, 1)
	for _, s := range slices {
		result = append(result, s...)
	}
	return result
}
