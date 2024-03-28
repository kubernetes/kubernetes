/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const StatefulSetControllerSubsystem = "statefulset_controller"

var StsRollingUpdateDuratationSeconds = metrics.NewHistogram(
	&metrics.HistogramOpts{
		Subsystem:      StatefulSetControllerSubsystem,
		Name:           "statefulset_rolling_update_duration_seconds",
		Help:           "Time between when a statefulset finishes rolling update, and when the corresponding statefulset started to do rolling update",
		Buckets:        metrics.ExponentialBuckets(0.001, 2, 16), // 1ms ~ 32.768s
		StabilityLevel: metrics.ALPHA,
	},
)

var registerMetrics sync.Once

// Register registers ReplicaSet controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(StsRollingUpdateDuratationSeconds)
	})
}
