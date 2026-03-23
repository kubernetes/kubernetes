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
	// ActiveResponder tracks the active responder per EvictionRequest.
	// Combined with the EvictionRequest creationTimestamp, this can help
	// identify stuck evictions. The responder label identifies which
	// responder is currently active.
	ActiveResponder = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "active_responder",
			Help:           "Whether the named responder is active for the EvictionRequest (1 = active)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target", "responder"},
	)

	// ProcessedResponder tracks whether an responder has been processed
	// for a given EvictionRequest. The responder label identifies which
	// responder has been processed.
	ProcessedResponder = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "processed_responder",
			Help:           "Whether the named responder has been processed for the EvictionRequest (1 = processed)",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target", "responder"},
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

	// PodResponders tracks the number of available responders for the
	// target pod of an EvictionRequest.
	PodResponders = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "target_responders",
			Help:           "Number of available responders for the target",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target"},
	)
)

var registerMetrics sync.Once

// Register registers EvictionRequest controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ActiveResponder)
		legacyregistry.MustRegister(ProcessedResponder)
		legacyregistry.MustRegister(ActiveRequester)
		legacyregistry.MustRegister(PodResponders)
	})
}
