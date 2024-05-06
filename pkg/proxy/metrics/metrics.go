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
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/util/nfacct"
)

const kubeProxySubsystem = "kubeproxy"

var (
	// SyncProxyRulesLatency is the latency of one round of kube-proxy syncing proxy
	// rules. (With the iptables proxy, this includes both full and partial syncs.)
	SyncProxyRulesLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_duration_seconds",
			Help:           "SyncProxyRules latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// SyncFullProxyRulesLatency is the latency of one round of full rule syncing.
	SyncFullProxyRulesLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_full_proxy_rules_duration_seconds",
			Help:           "SyncProxyRules latency in seconds for full resyncs",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// SyncPartialProxyRulesLatency is the latency of one round of partial rule syncing.
	SyncPartialProxyRulesLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_partial_proxy_rules_duration_seconds",
			Help:           "SyncProxyRules latency in seconds for partial resyncs",
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
			Buckets: metrics.MergeBuckets(
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

	// iptablesCTStateInvalidDroppedPacketsDescription describe the metrics for the number of packets dropped
	// by iptables which were marked INVALID by conntrack.
	iptablesCTStateInvalidDroppedPacketsDescription = metrics.NewDesc(
		"kubeproxy_iptables_ct_state_invalid_dropped_packets_total",
		"packets dropped by iptables to work around conntrack problems",
		nil, nil, metrics.ALPHA, "")
	IPTablesCTStateInvalidDroppedNFAcctCounter = "ct_state_invalid_dropped_pkts"

	// IPTablesRestoreFailuresTotal is the number of iptables restore failures that the proxy has
	// seen.
	IPTablesRestoreFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_iptables_restore_failures_total",
			Help:           "Cumulative proxy iptables restore failures",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// IPTablesPartialRestoreFailuresTotal is the number of iptables *partial* restore
	// failures (resulting in a fall back to a full restore) that the proxy has seen.
	IPTablesPartialRestoreFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_iptables_partial_restore_failures_total",
			Help:           "Cumulative proxy iptables partial restore failures",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// IPTablesRulesTotal is the total number of iptables rules that the iptables
	// proxy has installed.
	IPTablesRulesTotal = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_iptables_total",
			Help:           "Total number of iptables rules owned by kube-proxy",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"table"},
	)

	// IPTablesRulesLastSync is the number of iptables rules that the iptables proxy
	// updated in the last sync.
	IPTablesRulesLastSync = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_iptables_last",
			Help:           "Number of iptables rules written by kube-proxy in last sync",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"table"},
	)

	// NFTablesSyncFailuresTotal is the number of nftables sync failures that the
	// proxy has seen.
	NFTablesSyncFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_nftables_sync_failures_total",
			Help:           "Cumulative proxy nftables sync failures",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// NFTablesCleanupFailuresTotal is the number of nftables stale chain cleanup
	// failures that the proxy has seen.
	NFTablesCleanupFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_nftables_cleanup_failures_total",
			Help:           "Cumulative proxy nftables cleanup failures",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// ProxyHealthzTotal is the number of returned HTTP Status for each
	// healthz probe.
	ProxyHealthzTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "proxy_healthz_total",
			Help:           "Cumulative proxy healthz HTTP status",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)

	// ProxyLivezTotal is the number of returned HTTP Status for each
	// livez probe.
	ProxyLivezTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "proxy_livez_total",
			Help:           "Cumulative proxy livez HTTP status",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"code"},
	)

	// SyncProxyRulesLastQueuedTimestamp is the last time a proxy sync was
	// requested. If this is much larger than
	// kubeproxy_sync_proxy_rules_last_timestamp_seconds, then something is hung.
	SyncProxyRulesLastQueuedTimestamp = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_last_queued_timestamp_seconds",
			Help:           "The last time a sync of proxy rules was queued",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// SyncProxyRulesNoLocalEndpointsTotal is the total number of rules that do
	// not have an available endpoint. This can be caused by an internal
	// traffic policy with no available local workload.
	SyncProxyRulesNoLocalEndpointsTotal = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      kubeProxySubsystem,
			Name:           "sync_proxy_rules_no_local_endpoints_total",
			Help:           "Number of services with a Local traffic policy and no endpoints",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"traffic_policy"},
	)
)

var registerMetricsOnce sync.Once

// RegisterMetrics registers kube-proxy metrics.
func RegisterMetrics(mode kubeproxyconfig.ProxyMode) {
	registerMetricsOnce.Do(func() {
		// Core kube-proxy metrics for all backends
		legacyregistry.MustRegister(SyncProxyRulesLatency)
		legacyregistry.MustRegister(SyncProxyRulesLastQueuedTimestamp)
		legacyregistry.MustRegister(SyncProxyRulesLastTimestamp)
		legacyregistry.MustRegister(EndpointChangesPending)
		legacyregistry.MustRegister(EndpointChangesTotal)
		legacyregistry.MustRegister(ServiceChangesPending)
		legacyregistry.MustRegister(ServiceChangesTotal)
		legacyregistry.MustRegister(ProxyHealthzTotal)
		legacyregistry.MustRegister(ProxyLivezTotal)

		// FIXME: winkernel does not implement these
		legacyregistry.MustRegister(NetworkProgrammingLatency)
		legacyregistry.MustRegister(SyncProxyRulesNoLocalEndpointsTotal)

		switch mode {
		case kubeproxyconfig.ProxyModeIPTables:
			legacyregistry.CustomMustRegister(newCTStateInvalidPacketsCollector())
			legacyregistry.MustRegister(SyncFullProxyRulesLatency)
			legacyregistry.MustRegister(SyncPartialProxyRulesLatency)
			legacyregistry.MustRegister(IPTablesRestoreFailuresTotal)
			legacyregistry.MustRegister(IPTablesPartialRestoreFailuresTotal)
			legacyregistry.MustRegister(IPTablesRulesTotal)
			legacyregistry.MustRegister(IPTablesRulesLastSync)

		case kubeproxyconfig.ProxyModeIPVS:
			legacyregistry.MustRegister(IPTablesRestoreFailuresTotal)

		case kubeproxyconfig.ProxyModeNFTables:
			legacyregistry.MustRegister(NFTablesSyncFailuresTotal)
			legacyregistry.MustRegister(NFTablesCleanupFailuresTotal)

		case kubeproxyconfig.ProxyModeKernelspace:
			// currently no winkernel-specific metrics
		}
	})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

var _ metrics.StableCollector = &ctStateInvalidPacketsCollector{}

func newCTStateInvalidPacketsCollector() *ctStateInvalidPacketsCollector {
	client, err := nfacct.New()
	if err != nil {
		klog.ErrorS(err, "failed to initialize nfacct client")
	}
	return &ctStateInvalidPacketsCollector{client: client}
}

type ctStateInvalidPacketsCollector struct {
	metrics.BaseStableCollector
	client nfacct.Interface
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *ctStateInvalidPacketsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- iptablesCTStateInvalidDroppedPacketsDescription
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *ctStateInvalidPacketsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	if c.client != nil {
		counter, err := c.client.Get(IPTablesCTStateInvalidDroppedNFAcctCounter)
		if err != nil {
			klog.ErrorS(err, "failed to collect nfacct counter")
		} else {
			metric, err := metrics.NewConstMetric(iptablesCTStateInvalidDroppedPacketsDescription, metrics.CounterValue, float64(counter.Packets))
			if err != nil {
				klog.ErrorS(err, "failed to create constant metric")
			} else {
				ch <- metric
			}
		}
	}
}
