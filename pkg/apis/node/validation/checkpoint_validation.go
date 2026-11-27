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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
)

// MaxCheckpointedPodTemplateBytes bounds the protobuf encoding of
// status.checkpointedPodTemplate. Every get, list and watch of the
// PodCheckpoint carries the template, and the template cannot be replaced once
// written. The value must match the limit in the field's API documentation.
// One MiB follows the payload limits used by Secrets, ConfigMaps and
// ClusterTrustBundles while leaving room for the rest of the checkpoint object.
const MaxCheckpointedPodTemplateBytes = 1024 * 1024

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
// Field formats, set-once fields and condition list rules are covered by
// declarative validation. This function enforces the cross-field and
// transition rules that declarative tags cannot express.
func ValidatePodCheckpointStatusUpdate(newPC, oldPC *nodeapi.PodCheckpoint) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(&newPC.ObjectMeta, &oldPC.ObjectMeta, field.NewPath("metadata"))
	newStatus := newPC.Status
	if newStatus == nil {
		// The declarative NoUnset rule rejects clearing an existing status.
		return allErrs
	}
	oldStatus := oldPC.Status
	if oldStatus == nil {
		oldStatus = &nodeapi.PodCheckpointStatus{}
	}
	// conditions validation covered by DV
	allErrs = append(allErrs, metav1validation.ValidateConditions(newStatus.Conditions, field.NewPath("status", "conditions"))...)
	if template := newStatus.CheckpointedPodTemplate; template != nil && oldStatus.CheckpointedPodTemplate == nil {
		// Validate only the first capture; unrelated status updates must retain
		// templates that were accepted under earlier validation rules.
		allErrs = append(allErrs, validateCheckpointedPodTemplate(template, field.NewPath("status", "checkpointedPodTemplate"))...)
	}
	if !checkpointedContainersEqual(newStatus.CheckpointedContainers, oldStatus.CheckpointedContainers) {
		// The kubelet records the captured containers once, together with the template.
		// Declarative NoRemoveItem cannot express set-once for a list, so check it here.
		if len(oldStatus.CheckpointedContainers) > 0 {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("status", "checkpointedContainers"), "field is immutable once set"))
		} else {
			allErrs = append(allErrs, validateCheckpointedContainers(newPC)...)
		}
	}

	oldReady := meta.FindStatusCondition(oldStatus.Conditions, nodeapi.PodCheckpointConditionReady)
	terminal := oldReady != nil && (oldReady.Status == metav1.ConditionTrue ||
		oldReady.Reason == nodeapi.PodCheckpointReasonFailed ||
		oldReady.Reason == nodeapi.PodCheckpointReasonSourcePodReplaced)
	if !terminal {
		return allErrs
	}
	// The stored outcome is final, regardless of the caller or admission configuration.
	ready := meta.FindStatusCondition(newStatus.Conditions, nodeapi.PodCheckpointConditionReady)
	if ready == nil || !equality.Semantic.DeepEqual(*ready, *oldReady) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("status", "conditions"), "cannot change or remove a terminal Ready condition"))
	}
	oldStatusCopy, newStatusCopy := oldStatus.DeepCopy(), newStatus.DeepCopy()
	oldStatusCopy.Conditions, newStatusCopy.Conditions = nil, nil
	// These fields have declarative set-once constraints. The terminal check
	// only needs to reject setting them for the first time after completion.
	if oldStatusCopy.NodeName != nil {
		newStatusCopy.NodeName = oldStatusCopy.NodeName
	}
	if oldStatusCopy.SourcePodUID != nil {
		newStatusCopy.SourcePodUID = oldStatusCopy.SourcePodUID
	}
	if oldStatusCopy.CheckpointedPodTemplate != nil {
		newStatusCopy.CheckpointedPodTemplate = oldStatusCopy.CheckpointedPodTemplate
	}
	if len(oldStatusCopy.CheckpointedContainers) > 0 {
		newStatusCopy.CheckpointedContainers = oldStatusCopy.CheckpointedContainers
	} else if len(newStatusCopy.CheckpointedContainers) == 0 {
		oldStatusCopy.CheckpointedContainers, newStatusCopy.CheckpointedContainers = nil, nil
	}
	if !equality.Semantic.DeepEqual(oldStatusCopy, newStatusCopy) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("status"), "cannot change checkpoint data after a terminal outcome"))
	}
	return allErrs
}

func validateCheckpointedPodTemplate(template *core.PodTemplateSpec, path *field.Path) field.ErrorList {
	var versioned v1.PodTemplateSpec
	if err := corev1.Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(template, &versioned, nil); err != nil {
		return field.ErrorList{field.InternalError(path, err)}
	}
	if versioned.Size() > MaxCheckpointedPodTemplateBytes {
		return field.ErrorList{field.TooLong(path, "", MaxCheckpointedPodTemplateBytes)}
	}
	opts := corevalidation.GetValidationOptionsFromPodTemplate(template, nil)
	return corevalidation.ValidatePodTemplateSpec(template, path, opts)
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
	if pc.Status == nil {
		return field.ErrorList{field.Forbidden(path, "requires a captured Pod template")}
	}
	template := pc.Status.CheckpointedPodTemplate
	if template == nil {
		return field.ErrorList{field.Forbidden(path, "requires a captured Pod template")}
	}

	captured := sets.New[string]()
	for _, container := range template.Spec.Containers {
		captured.Insert(container.Name)
	}
	for _, container := range template.Spec.InitContainers {
		if container.RestartPolicy != nil && *container.RestartPolicy == core.ContainerRestartPolicyAlways {
			captured.Insert(container.Name)
		}
	}
	var allErrs field.ErrorList
	for i, container := range pc.Status.CheckpointedContainers {
		if !captured.Has(container.Name) {
			allErrs = append(allErrs, field.Invalid(path.Index(i).Child("name"), container.Name, "must match the name of a regular or restartable init container in `status.checkpointedPodTemplate`"))
		}
	}
	return allErrs
}
