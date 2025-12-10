/*
Copyright 2023 The Kubernetes Authors.

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

package aggregated

import (
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const subsystem = "aggregator_discovery"

var (
	regenerationCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "aggregation_count_total",
			Subsystem:      subsystem,
			Help:           "Counter of number of times discovery was aggregated",
			StabilityLevel: metrics.ALPHA,
		},
	)

	PeerAggregatedCacheHitsCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "peer_aggregated_cache_hits_total",
			Subsystem:      subsystem,
			Help:           "Counter of number of times discovery was served from peer-aggregated cache",
			StabilityLevel: metrics.ALPHA,
		},
	)

	PeerAggregatedCacheMissesCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "peer_aggregated_cache_misses_total",
			Subsystem:      subsystem,
			Help:           "Counter of number of times discovery was aggregated across all API servers",
			StabilityLevel: metrics.ALPHA,
		},
	)

	NoPeerDiscoveryRequestCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "nopeer_requests_total",
			Subsystem:      subsystem,
			Help:           "Counter of number of times no-peer (non peer-aggregated) discovery was requested",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

func init() {
	legacyregistry.MustRegister(regenerationCounter)
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.UnknownVersionInteroperabilityProxy) {
		legacyregistry.MustRegister(PeerAggregatedCacheHitsCounter)
		legacyregistry.MustRegister(PeerAggregatedCacheMissesCounter)
		legacyregistry.MustRegister(NoPeerDiscoveryRequestCounter)
	}
}
