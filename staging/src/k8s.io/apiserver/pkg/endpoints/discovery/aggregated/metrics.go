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

var (
	regenerationCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "aggregator_discovery_aggregation_count_total",
			Help:           "Counter of number of times discovery was aggregated",
			StabilityLevel: metrics.ALPHA,
		},
	)

	MergedRequestCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "aggregator_discovery_merged_count_total",
			Help:           "Counter of number of times discovery was merged across all API servers",
			StabilityLevel: metrics.ALPHA,
		},
	)

	UnmergedRequestCounter = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           "aggregator_discovery_unmerged_count_total",
			Help:           "Counter of number of times unmerged discovery was requested",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

func init() {
	legacyregistry.MustRegister(regenerationCounter)
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.UnknownVersionInteroperabilityProxy) {
		legacyregistry.MustRegister(MergedRequestCounter)
		legacyregistry.MustRegister(UnmergedRequestCounter)
	}
}
