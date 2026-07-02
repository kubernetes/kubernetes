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
	if checkpointing == nil || podSpecsEqualIgnoringSchedulingGates(&oldPod.Spec, &newPod.Spec) {
		return nil
	}
	message := "pod spec cannot change while PodCheckpointing=True"
	if checkpointing.Message != "" {
		message += ": " + checkpointing.Message
	}
	return field.ErrorList{field.Forbidden(field.NewPath("spec"), message+"; scheduling gates may be removed")}
}

// validatePodRestoreUpdate keeps the spec of a restoring Pod equal to the
// captured template until the restore succeeds.
func validatePodRestoreUpdate(newPod, oldPod *core.Pod) field.ErrorList {
	if oldPod == nil || !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) || oldPod.Spec.RestoreFrom == nil {
		return nil
	}
	// Declarative validation reports a changed restoreFrom; do not report it twice.
	if !apiequality.Semantic.DeepEqual(oldPod.Spec.RestoreFrom, newPod.Spec.RestoreFrom) {
		return nil
	}
	for _, condition := range oldPod.Status.Conditions {
		if condition.Type == core.PodRestored && condition.Status == core.ConditionTrue {
			return nil
		}
	}
	if podSpecsEqualIgnoringSchedulingGates(&oldPod.Spec, &newPod.Spec) {
		return nil
	}
	return field.ErrorList{field.Forbidden(field.NewPath("spec"), "pod spec cannot change until restore succeeds (PodRestored=True); scheduling gates may be removed")}
}

// podSpecsEqualIgnoringSchedulingGates compares two specs of the same Pod.
// Scheduling gates only delay scheduling; they do not change the workload.
func podSpecsEqualIgnoringSchedulingGates(oldSpec, newSpec *core.PodSpec) bool {
	a, b := *oldSpec, *newSpec
	a.SchedulingGates, b.SchedulingGates = nil, nil
	return apiequality.Semantic.DeepEqual(a, b)
}
