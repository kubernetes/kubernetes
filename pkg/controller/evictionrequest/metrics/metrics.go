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
	// ResponderState tracks the state of each responder per EvictionRequest.
	// Combined with the EvictionRequest creationTimestamp, this can help
	// identify stuck evictions. The state label identifies which
	// responder is currently active.
	ResponderState = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "responder_state",
			Help:           "The state of each responder for the EvictionRequest.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target_name", "target_type", "responder", "state"},
	)

	// RequesterIntent tracks requesters that have registered an intent to evict a pod.
	// This intent can change over time (e.g. to a value Withdrawn).
	// The requester label identifies the requester by name.
	RequesterIntent = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "requester_intent",
			Help:           "The intent of each requester per the EvictionRequest",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target_name", "target_type", "requester", "intent"},
	)

	// TargetResponders tracks the number of available responders for the
	// target pod of an EvictionRequest.
	TargetResponders = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EvictionRequestControllerSubsystem,
			Name:           "target_responders",
			Help:           "The number of available responders for the target",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"namespace", "evictionrequest", "target_name", "target_type"},
	)
)

var registerMetrics sync.Once

// Register registers EvictionRequest controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ResponderState)
		legacyregistry.MustRegister(RequesterIntent)
		legacyregistry.MustRegister(TargetResponders)
	})
}
