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

package node

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const nodeAuthorizerSubsystem = "node_authorizer"

var (
	graphActionsDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      nodeAuthorizerSubsystem,
			Name:           "graph_actions_duration_seconds",
			Help:           "Histogram of duration of graph actions in node authorizer.",
			StabilityLevel: metrics.ALPHA,
			// Start with 0.1ms with the last bucket being [~200ms, Inf)
			Buckets: metrics.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"operation"},
	)
)

var registerMetrics sync.Once

// RegisterMetrics registers metrics for node package.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(graphActionsDuration)
	})
}
