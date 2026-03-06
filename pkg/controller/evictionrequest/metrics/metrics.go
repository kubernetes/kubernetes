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

// EvictionRequestControllerSubsystem - subsystem name used for this controller.
const EvictionRequestControllerSubsystem = "evictionrequest_controller"

var (
	// ActiveInterceptor tracks the active interceptor per EvictionRequest.
	// Combined with the EvictionRequest creationTimestamp, this can help
	// identify stuck evictions. The interceptor label identifies which
	// interceptor is currently active.
	ActiveInterceptor = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "active_interceptor",
			Help:           "Whether the named interceptor is active for the EvictionRequest (1 = active)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target", "interceptor"},
	)

	// ProcessedInterceptor tracks whether an interceptor has been processed
	// for a given EvictionRequest. The interceptor label identifies which
	// interceptor has been processed.
	ProcessedInterceptor = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "processed_interceptor",
			Help:           "Whether the named interceptor has been processed for the EvictionRequest (1 = processed)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target", "interceptor"},
	)

	// ActiveRequester tracks active requesters per EvictionRequest.
	// The requester label identifies the requester by name (1 = active).
	ActiveRequester = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "active_requester",
			Help:           "Whether the named requester is active for the EvictionRequest (1 = active)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target", "requester"},
	)

	// PodInterceptors tracks the number of available interceptors for the
	// target pod of an EvictionRequest.
	PodInterceptors = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "target_interceptors",
			Help:           "Number of available interceptors for the target",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target"},
	)
)

var registerMetrics sync.Once

// Register registers EvictionRequest controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ActiveInterceptor)
		legacyregistry.MustRegister(ProcessedInterceptor)
		legacyregistry.MustRegister(ActiveRequester)
		legacyregistry.MustRegister(PodInterceptors)
	})
}
