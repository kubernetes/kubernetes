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

func NewFlowSchemaOps(client flowcontrolclient.FlowSchemaInterface, cache flowcontrollisters.FlowSchemaLister) ObjectOps[*flowcontrolv1.FlowSchema] {
	return NewObjectOps[*flowcontrolv1.FlowSchema](client, cache, (*flowcontrolv1.FlowSchema).DeepCopy, flowSchemaReplaceSpec, flowSchemaSpecEqual)
}

func flowSchemaReplaceSpec(into, from *flowcontrolv1.FlowSchema) *flowcontrolv1.FlowSchema {
	copy := into.DeepCopy()
	copy.Spec = *from.Spec.DeepCopy()
	return copy
}

func flowSchemaSpecEqual(expected, actual *flowcontrolv1.FlowSchema) bool {
	copiedExpectedSpec := expected.Spec.DeepCopy()
	flowcontrolapisv1.SetDefaults_FlowSchemaSpec(copiedExpectedSpec)
	return equality.Semantic.DeepEqual(copiedExpectedSpec, &actual.Spec)
}
