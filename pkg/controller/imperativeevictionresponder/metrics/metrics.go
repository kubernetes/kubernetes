/*
Copyright The Kubernetes Authors.

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

// ImperativeEvictionResponderControllerSubsystem - subsystem name used for this controller.
const ImperativeEvictionResponderControllerSubsystem = "imperative_eviction_responder_controller"

var (
	// FailedEvictionsTotal tracks the number of failed evictions
	// where at least three unsuccessful calls have been made to pods/eviction
	FailedEvictionsTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      ImperativeEvictionResponderControllerSubsystem,
			Name:           "failed_evictions_total",
			Help:           "Total number of failed evictions where at least three unsuccessful calls have been made to pods/<name>/eviction subresource, corresponding to the Eviction .spec.target.pod.",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var registerMetrics sync.Once

// Register registers ImperativeEvictionResponder controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(FailedEvictionsTotal)
	})
}
