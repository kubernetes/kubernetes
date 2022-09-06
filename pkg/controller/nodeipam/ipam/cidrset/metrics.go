/*
Copyright 2020 The Kubernetes Authors.

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

package cidrset

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const nodeIpamSubsystem = "node_ipam_controller"

var (
	cidrSetAllocations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      nodeIpamSubsystem,
			Name:           "cidrset_cidrs_allocations_total",
			Help:           "Counter measuring total number of CIDR allocations.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"clusterCIDR"},
	)
	cidrSetReleases = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      nodeIpamSubsystem,
			Name:           "cidrset_cidrs_releases_total",
			Help:           "Counter measuring total number of CIDR releases.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"clusterCIDR"},
	)
	// This is a gauge, as in theory, a limit can increase or decrease.
	cidrSetMaxCidrs = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      nodeIpamSubsystem,
			Name:           "cirdset_max_cidrs",
			Help:           "Maximum number of CIDRs that can be allocated.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"clusterCIDR"},
	)
	cidrSetUsage = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      nodeIpamSubsystem,
			Name:           "cidrset_usage_cidrs",
			Help:           "Gauge measuring percentage of allocated CIDRs.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"clusterCIDR"},
	)
	cidrSetAllocationTriesPerRequest = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      nodeIpamSubsystem,
			Name:           "cidrset_allocation_tries_per_request",
			Help:           "Number of endpoints added on each Service sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(1, 5, 5),
		},
		[]string{"clusterCIDR"},
	)
)

var registerMetrics sync.Once

// registerCidrsetMetrics the metrics that are to be monitored.
func registerCidrsetMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(cidrSetAllocations)
		legacyregistry.MustRegister(cidrSetReleases)
		legacyregistry.MustRegister(cidrSetMaxCidrs)
		legacyregistry.MustRegister(cidrSetUsage)
		legacyregistry.MustRegister(cidrSetAllocationTriesPerRequest)
	})
}
