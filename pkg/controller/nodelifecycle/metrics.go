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

package nodelifecycle

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	nodeControllerSubsystem = "node_collector"
	zoneHealthStatisticKey  = "zone_health"
	zoneSizeKey             = "zone_size"
	zoneNoUnhealthyNodesKey = "unhealthy_nodes_in_zone"
	evictionsTotalKey       = "evictions_total"

	updateNodeHealthKey     = "update_node_health_duration_seconds"
	updateAllNodesHealthKey = "update_all_nodes_health_duration_seconds"
)

var (
	zoneHealth = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           zoneHealthStatisticKey,
			Help:           "Gauge measuring percentage of healthy nodes per zone.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"zone"},
	)
	zoneSize = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           zoneSizeKey,
			Help:           "Gauge measuring number of registered Nodes per zones.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"zone"},
	)
	unhealthyNodes = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           zoneNoUnhealthyNodesKey,
			Help:           "Gauge measuring number of not Ready Nodes per zones.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"zone"},
	)
	evictionsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           evictionsTotalKey,
			Help:           "Number of Node evictions that happened since current instance of NodeController started.",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"zone"},
	)

	updateNodeHealthDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           updateNodeHealthKey,
			Help:           "Duration in seconds for NodeController to update the health of a single node.",
			Buckets:        metrics.ExponentialBuckets(0.001, 4, 8), // 1ms -> ~15s
			StabilityLevel: metrics.ALPHA,
		},
	)
	updateAllNodesHealthDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      nodeControllerSubsystem,
			Name:           updateAllNodesHealthKey,
			Help:           "Duration in seconds for NodeController to update the health of all nodes.",
			Buckets:        metrics.ExponentialBuckets(0.01, 4, 8), // 10ms -> ~3m
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var registerMetrics sync.Once

// Register the metrics that are to be monitored.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(zoneHealth)
		legacyregistry.MustRegister(zoneSize)
		legacyregistry.MustRegister(unhealthyNodes)
		legacyregistry.MustRegister(evictionsTotal)
		legacyregistry.MustRegister(updateNodeHealthDuration)
		legacyregistry.MustRegister(updateAllNodesHealthDuration)
	})
}
