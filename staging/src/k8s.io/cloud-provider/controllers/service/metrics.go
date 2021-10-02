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

package service

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"sync"
)

const (
	// subSystemName is the name of this subsystem name used for prometheus metrics.
	subSystemName = "service_controller"
)

var register sync.Once

// registerMetrics registers service-controller metrics.
func registerMetrics() {
	register.Do(func() {
		legacyregistry.MustRegister(nodeSyncLatency)
		legacyregistry.MustRegister(updateLoadBalancerHostLatency)
	})
}

var (
	nodeSyncLatency = metrics.NewHistogram(&metrics.HistogramOpts{
		Name:      "nodesync_latency_seconds",
		Subsystem: subSystemName,
		Help:      "A metric measuring the latency for nodesync which updates loadbalancer hosts on cluster node updates.",
		// Buckets from 1s to 16384s
		Buckets:        metrics.ExponentialBuckets(1, 2, 15),
		StabilityLevel: metrics.ALPHA,
	})

	updateLoadBalancerHostLatency = metrics.NewHistogram(&metrics.HistogramOpts{
		Name:      "update_loadbalancer_host_latency_seconds",
		Subsystem: subSystemName,
		Help:      "A metric measuring the latency for updating each load balancer hosts.",
		// Buckets from 1s to 16384s
		Buckets:        metrics.ExponentialBuckets(1, 2, 15),
		StabilityLevel: metrics.ALPHA,
	})
)
