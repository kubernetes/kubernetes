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

package ensurer

import (
	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1"
	flowcontrolapisv1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1"
)

func NewPriorityLevelConfigurationOps(client flowcontrolclient.PriorityLevelConfigurationInterface, lister flowcontrollisters.PriorityLevelConfigurationLister) ObjectOps[*flowcontrolv1.PriorityLevelConfiguration] {
	return NewObjectOps[*flowcontrolv1.PriorityLevelConfiguration](client, lister, (*flowcontrolv1.PriorityLevelConfiguration).DeepCopy,
		plcReplaceSpec, plcSpecEqualish)
}

func plcReplaceSpec(into, from *flowcontrolv1.PriorityLevelConfiguration) *flowcontrolv1.PriorityLevelConfiguration {
	copy := into.DeepCopy()
	copy.Spec = *from.Spec.DeepCopy()
	return copy
}

func plcSpecEqualish(expected, actual *flowcontrolv1.PriorityLevelConfiguration) bool {
	copiedExpected := expected.DeepCopy()
	flowcontrolapisv1.SetObjectDefaults_PriorityLevelConfiguration(copiedExpected)
	if expected.Name == flowcontrolv1.PriorityLevelConfigurationNameExempt {
		if actual.Spec.Exempt == nil {
			return false
		}
		copiedExpected.Spec.Exempt.NominalConcurrencyShares = actual.Spec.Exempt.NominalConcurrencyShares
		copiedExpected.Spec.Exempt.LendablePercent = actual.Spec.Exempt.LendablePercent
	}
	return equality.Semantic.DeepEqual(copiedExpected.Spec, actual.Spec)
}
