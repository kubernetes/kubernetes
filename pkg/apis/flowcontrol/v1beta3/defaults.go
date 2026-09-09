/*
Copyright 2022 The Kubernetes Authors.

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

package v1beta3

import (
	"k8s.io/api/flowcontrol/v1beta3"
	"k8s.io/utils/ptr"
)

// Default settings for flow-schema
const (
	FlowSchemaDefaultMatchingPrecedence int32 = 1000
)

// Default settings for priority-level-configuration
const (
	PriorityLevelConfigurationDefaultHandSize                 int32 = 8
	PriorityLevelConfigurationDefaultQueues                   int32 = 64
	PriorityLevelConfigurationDefaultQueueLengthLimit         int32 = 50
	PriorityLevelConfigurationDefaultNominalConcurrencyShares int32 = 30
)

// SetDefaults_FlowSchema sets default values for flow schema
func SetDefaults_FlowSchemaSpec(spec *v1beta3.FlowSchemaSpec) {
	if spec.MatchingPrecedence == 0 {
		spec.MatchingPrecedence = FlowSchemaDefaultMatchingPrecedence
	}
}

// SetDefaults_PriorityLevelConfiguration sets the default values for a
// PriorityLevelConfiguration object. Since we need to inspect the presence
// of the roundtrip annotation in order to determine whether the user has
// specified a zero value for the 'NominalConcurrencyShares' field,
// the defaulting logic needs visibility to the annotations field.
func SetDefaults_PriorityLevelConfiguration(in *v1beta3.PriorityLevelConfiguration) {
	if limited := in.Spec.Limited; limited != nil {
		// for v1beta3, we apply a default value to the NominalConcurrencyShares
		// field only when:
		//   a) NominalConcurrencyShares == 0, and
		//   b) the roundtrip annotation is not set
		if _, ok := in.Annotations[v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey]; !ok && limited.NominalConcurrencyShares == 0 {
			limited.NominalConcurrencyShares = PriorityLevelConfigurationDefaultNominalConcurrencyShares
		}
	}
}

func SetDefaults_ExemptPriorityLevelConfiguration(eplc *v1beta3.ExemptPriorityLevelConfiguration) {
	if eplc.NominalConcurrencyShares == nil {
		eplc.NominalConcurrencyShares = ptr.To(int32(0))
	}
	if eplc.LendablePercent == nil {
		eplc.LendablePercent = ptr.To(int32(0))
	}
}

func SetDefaults_LimitedPriorityLevelConfiguration(in *v1beta3.LimitedPriorityLevelConfiguration) {
	if in.LendablePercent == nil {
		in.LendablePercent = ptr.To(int32(0))
	}
}

func SetDefaults_QueuingConfiguration(cfg *v1beta3.QueuingConfiguration) {
	if cfg.HandSize == 0 {
		cfg.HandSize = PriorityLevelConfigurationDefaultHandSize
	}
	if cfg.Queues == 0 {
		cfg.Queues = PriorityLevelConfigurationDefaultQueues
	}
	if cfg.QueueLengthLimit == 0 {
		cfg.QueueLengthLimit = PriorityLevelConfigurationDefaultQueueLengthLimit
	}
}
