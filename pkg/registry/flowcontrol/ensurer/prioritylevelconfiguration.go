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
	flowcontrolv1beta3 "k8s.io/api/flowcontrol/v1beta3"
	"k8s.io/apimachinery/pkg/api/equality"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta3"
	flowcontrollisters "k8s.io/client-go/listers/flowcontrol/v1beta3"
	flowcontrolapisv1beta3 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta3"
)

func NewPriorityLevelConfigurationOps(client flowcontrolclient.PriorityLevelConfigurationInterface, lister flowcontrollisters.PriorityLevelConfigurationLister) ObjectOps[*flowcontrolv1beta3.PriorityLevelConfiguration] {
	return NewObjectOps[*flowcontrolv1beta3.PriorityLevelConfiguration](client, lister, (*flowcontrolv1beta3.PriorityLevelConfiguration).DeepCopy,
		plcReplaceSpec, plcSpecEqual)
}

func plcReplaceSpec(into, from *flowcontrolv1beta3.PriorityLevelConfiguration) *flowcontrolv1beta3.PriorityLevelConfiguration {
	copy := into.DeepCopy()
	copy.Spec = *from.Spec.DeepCopy()
	return copy
}

func plcSpecEqual(expected, actual *flowcontrolv1beta3.PriorityLevelConfiguration) bool {
	copiedExpected := expected.DeepCopy()
	flowcontrolapisv1beta3.SetObjectDefaults_PriorityLevelConfiguration(copiedExpected)
	return equality.Semantic.DeepEqual(copiedExpected.Spec, actual.Spec)
}
