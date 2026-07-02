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

package validation

import (
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func validatePodCheckpointingUpdate(newPod, oldPod *core.Pod) field.ErrorList {
	if oldPod == nil || !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) {
		return nil
	}

	var checkpointing *core.PodCondition
	for i := range oldPod.Status.Conditions {
		condition := &oldPod.Status.Conditions[i]
		if condition.Type == core.PodCheckpointing && condition.Status == core.ConditionTrue {
			checkpointing = condition
			break
		}
	}
	if checkpointing == nil {
		return nil
	}

	// Scheduling gates only delay scheduling; they do not change the captured workload.
	oldSpec, newSpec := oldPod.Spec, newPod.Spec
	oldSpec.SchedulingGates, newSpec.SchedulingGates = nil, nil
	if apiequality.Semantic.DeepEqual(oldSpec, newSpec) {
		return nil
	}
	message := "pod spec cannot change while PodCheckpointing=True"
	if checkpointing.Message != "" {
		message += ": " + checkpointing.Message
	}
	return field.ErrorList{field.Forbidden(field.NewPath("spec"), message+"; scheduling gates may be removed")}
}
