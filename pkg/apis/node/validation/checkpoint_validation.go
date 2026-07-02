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
	"cmp"
	"slices"

	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
)

// ValidatePodCheckpointName validates the name of a PodCheckpoint.
func ValidatePodCheckpointName(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidatePodCheckpoint validates a PodCheckpoint.
func ValidatePodCheckpoint(pc *nodeapi.PodCheckpoint) field.ErrorList {
	// Spec validation is declarative, including immutability on update.
	return apimachineryvalidation.ValidateObjectMeta(&pc.ObjectMeta, true, ValidatePodCheckpointName, field.NewPath("metadata"))
}

// ValidatePodCheckpointUpdate validates a PodCheckpoint update.
func ValidatePodCheckpointUpdate(newPC, oldPC *nodeapi.PodCheckpoint) field.ErrorList {
	return apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
}

// ValidatePodCheckpointStatusUpdate validates a status update of a PodCheckpoint.
func ValidatePodCheckpointStatusUpdate(newPC, oldPC *nodeapi.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, metav1validation.ValidateConditions(newPC.Status.Conditions, field.NewPath("status", "conditions"))...)
	if template := newPC.Status.CheckpointedPodTemplate; template != nil && oldPC.Status.CheckpointedPodTemplate == nil {
		// Validate only the first capture; unrelated status updates must retain
		// templates that were accepted under earlier validation rules.
		opts := corevalidation.GetValidationOptionsFromPodTemplate(template, nil)
		allErrs = append(allErrs, corevalidation.ValidatePodTemplateSpec(template, field.NewPath("status", "checkpointedPodTemplate"), opts)...)
	}
	if !checkpointedContainersEqual(newPC.Status.CheckpointedContainers, oldPC.Status.CheckpointedContainers) && len(newPC.Status.CheckpointedContainers) > 0 {
		allErrs = append(allErrs, validateCheckpointedContainers(newPC)...)
	}

	oldReady := meta.FindStatusCondition(oldPC.Status.Conditions, nodeapi.PodCheckpointConditionReady)
	if oldReady == nil || (oldReady.Status != metav1.ConditionTrue && oldReady.Reason != nodeapi.PodCheckpointReasonFailed && oldReady.Reason != nodeapi.PodCheckpointReasonSourcePodReplaced) {
		return allErrs
	}
	// The stored outcome is final, regardless of the caller or admission configuration.
	ready := meta.FindStatusCondition(newPC.Status.Conditions, nodeapi.PodCheckpointConditionReady)
	if ready == nil || ready.Status != oldReady.Status || ready.Reason != oldReady.Reason {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("status", "conditions"), "cannot change or remove a terminal Ready outcome"))
	}
	oldStatus, newStatus := oldPC.Status.DeepCopy(), newPC.Status.DeepCopy()
	oldStatus.Conditions, newStatus.Conditions = nil, nil
	// Container names are map-list keys; their order does not change the outcome.
	byName := func(a, b nodeapi.PodCheckpointContainerStatus) int { return cmp.Compare(a.Name, b.Name) }
	slices.SortFunc(oldStatus.CheckpointedContainers, byName)
	slices.SortFunc(newStatus.CheckpointedContainers, byName)
	if len(oldStatus.CheckpointedContainers) == 0 {
		oldStatus.CheckpointedContainers = nil
	}
	if len(newStatus.CheckpointedContainers) == 0 {
		newStatus.CheckpointedContainers = nil
	}
	if !equality.Semantic.DeepEqual(oldStatus, newStatus) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("status"), "cannot change checkpoint data after a terminal outcome"))
	}
	return allErrs
}

func checkpointedContainersEqual(a, b []nodeapi.PodCheckpointContainerStatus) bool {
	byName := func(a, b nodeapi.PodCheckpointContainerStatus) int { return cmp.Compare(a.Name, b.Name) }
	left, right := slices.Clone(a), slices.Clone(b)
	slices.SortFunc(left, byName)
	slices.SortFunc(right, byName)
	return equality.Semantic.DeepEqual(left, right)
}

func validateCheckpointedContainers(pc *nodeapi.PodCheckpoint) field.ErrorList {
	path := field.NewPath("status", "checkpointedContainers")
	template := pc.Status.CheckpointedPodTemplate
	if template == nil {
		return field.ErrorList{field.Forbidden(path, "requires a captured Pod template")}
	}

	expected := make([]string, 0, len(template.Spec.Containers)+len(template.Spec.InitContainers))
	for _, container := range template.Spec.Containers {
		expected = append(expected, container.Name)
	}
	for _, container := range template.Spec.InitContainers {
		if container.RestartPolicy != nil && *container.RestartPolicy == "Always" {
			expected = append(expected, container.Name)
		}
	}
	actual := make([]string, 0, len(pc.Status.CheckpointedContainers))
	for _, container := range pc.Status.CheckpointedContainers {
		actual = append(actual, container.Name)
	}
	slices.Sort(expected)
	slices.Sort(actual)
	if !slices.Equal(expected, actual) {
		return field.ErrorList{field.Invalid(path, actual, "must list exactly the regular and restartable init containers in checkpointedPodTemplate")}
	}
	return nil
}
