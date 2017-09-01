/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/apps"
)

// ValidateStatefulSetName can be used to check whether the given StatefulSet name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateStatefulSetName(name string, prefix bool) []string {
	// TODO: Validate that there's name for the suffix inserted by the pods.
	// Currently this is just "-index". In the future we may allow a user
	// specified list of suffixes and we need  to validate the longest one.
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// Validates the given template and ensures that it is in accordance with the desired selector.
func ValidatePodTemplateSpecForStatefulSet(template *api.PodTemplateSpec, selector labels.Selector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if template == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		if !selector.Empty() {
			// Verify that the StatefulSet selector matches the labels in template.
			labels := labels.Set(template.Labels)
			if !selector.Matches(labels) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("metadata", "labels"), template.Labels, "`selector` does not match template `labels`"))
			}
		}
		// TODO: Add validation for PodSpec, currently this will check volumes, which we know will
		// fail. We should really check that the union of the given volumes and volumeClaims match
		// volume mounts in the containers.
		// allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(template, fldPath)...)
		allErrs = append(allErrs, unversionedvalidation.ValidateLabels(template.Labels, fldPath.Child("labels"))...)
		allErrs = append(allErrs, apivalidation.ValidateAnnotations(template.Annotations, fldPath.Child("annotations"))...)
		allErrs = append(allErrs, apivalidation.ValidatePodSpecificAnnotations(template.Annotations, &template.Spec, fldPath.Child("annotations"))...)
	}
	return allErrs
}

// ValidateStatefulSetSpec tests if required fields in the StatefulSet spec are set.
func ValidateStatefulSetSpec(spec *apps.StatefulSetSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	switch spec.PodManagementPolicy {
	case "":
		allErrs = append(allErrs, field.Required(fldPath.Child("podManagementPolicy"), ""))
	case apps.OrderedReadyPodManagement, apps.ParallelPodManagement:
	default:
		allErrs = append(allErrs, field.Invalid(fldPath.Child("podManagementPolicy"), spec.PodManagementPolicy, fmt.Sprintf("must be '%s' or '%s'", apps.OrderedReadyPodManagement, apps.ParallelPodManagement)))
	}

	switch spec.UpdateStrategy.Type {
	case "":
		allErrs = append(allErrs, field.Required(fldPath.Child("updateStrategy"), ""))
	case apps.OnDeleteStatefulSetStrategyType:
		if spec.UpdateStrategy.RollingUpdate != nil {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath.Child("updateStrategy").Child("rollingUpdate"),
					spec.UpdateStrategy.RollingUpdate,
					fmt.Sprintf("only allowed for updateStrategy '%s'", apps.RollingUpdateStatefulSetStrategyType)))
		}
	case apps.RollingUpdateStatefulSetStrategyType:
		if spec.UpdateStrategy.RollingUpdate != nil {
			allErrs = append(allErrs,
				apivalidation.ValidateNonnegativeField(
					int64(spec.UpdateStrategy.RollingUpdate.Partition),
					fldPath.Child("updateStrategy").Child("rollingUpdate").Child("partition"))...)
		}

	default:
		allErrs = append(allErrs,
			field.Invalid(fldPath.Child("updateStrategy"), spec.UpdateStrategy,
				fmt.Sprintf("must be '%s' or '%s'",
					apps.RollingUpdateStatefulSetStrategyType,
					apps.OnDeleteStatefulSetStrategyType)))
	}

	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(spec.Replicas), fldPath.Child("replicas"))...)
	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
		if len(spec.Selector.MatchLabels)+len(spec.Selector.MatchExpressions) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("selector"), spec.Selector, "empty selector is not valid for statefulset."))
		}
	}

	selector, err := metav1.LabelSelectorAsSelector(spec.Selector)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("selector"), spec.Selector, ""))
	} else {
		allErrs = append(allErrs, ValidatePodTemplateSpecForStatefulSet(&spec.Template, selector, fldPath.Child("template"))...)
	}

	if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"), spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
	}
	if spec.Template.Spec.ActiveDeadlineSeconds != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("spec", "activeDeadlineSeconds"), spec.Template.Spec.ActiveDeadlineSeconds, "must not be specified"))
	}

	return allErrs
}

// ValidateStatefulSet validates a StatefulSet.
func ValidateStatefulSet(statefulSet *apps.StatefulSet) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&statefulSet.ObjectMeta, true, ValidateStatefulSetName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateStatefulSetSpec(&statefulSet.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateStatefulSetUpdate tests if required fields in the StatefulSet are set.
func ValidateStatefulSetUpdate(statefulSet, oldStatefulSet *apps.StatefulSet) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&statefulSet.ObjectMeta, &oldStatefulSet.ObjectMeta, field.NewPath("metadata"))

	restoreReplicas := statefulSet.Spec.Replicas
	statefulSet.Spec.Replicas = oldStatefulSet.Spec.Replicas

	restoreTemplate := statefulSet.Spec.Template
	statefulSet.Spec.Template = oldStatefulSet.Spec.Template

	restoreStrategy := statefulSet.Spec.UpdateStrategy
	statefulSet.Spec.UpdateStrategy = oldStatefulSet.Spec.UpdateStrategy

	if !reflect.DeepEqual(statefulSet.Spec, oldStatefulSet.Spec) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "updates to statefulset spec for fields other than 'replicas', 'template', and 'updateStrategy' are forbidden."))
	}
	statefulSet.Spec.Replicas = restoreReplicas
	statefulSet.Spec.Template = restoreTemplate
	statefulSet.Spec.UpdateStrategy = restoreStrategy

	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(statefulSet.Spec.Replicas), field.NewPath("spec", "replicas"))...)
	return allErrs
}

// ValidateStatefulSetStatus validates a StatefulSetStatus.
func ValidateStatefulSetStatus(status *apps.StatefulSetStatus, fieldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Replicas), fieldPath.Child("replicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.ReadyReplicas), fieldPath.Child("readyReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.CurrentReplicas), fieldPath.Child("currentReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.UpdatedReplicas), fieldPath.Child("updatedReplicas"))...)
	if status.ObservedGeneration != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*status.ObservedGeneration), fieldPath.Child("observedGeneration"))...)
	}
	if status.CollisionCount != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*status.CollisionCount), fieldPath.Child("collisionCount"))...)
	}

	msg := "cannot be greater than status.replicas"
	if status.ReadyReplicas > status.Replicas {
		allErrs = append(allErrs, field.Invalid(fieldPath.Child("readyReplicas"), status.ReadyReplicas, msg))
	}
	if status.CurrentReplicas > status.Replicas {
		allErrs = append(allErrs, field.Invalid(fieldPath.Child("currentReplicas"), status.CurrentReplicas, msg))
	}
	if status.UpdatedReplicas > status.Replicas {
		allErrs = append(allErrs, field.Invalid(fieldPath.Child("updatedReplicas"), status.UpdatedReplicas, msg))
	}

	return allErrs
}

// ValidateStatefulSetStatusUpdate tests if required fields in the StatefulSet are set.
func ValidateStatefulSetStatusUpdate(statefulSet, oldStatefulSet *apps.StatefulSet) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateStatefulSetStatus(&statefulSet.Status, field.NewPath("status"))...)
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&statefulSet.ObjectMeta, &oldStatefulSet.ObjectMeta, field.NewPath("metadata"))...)
	// TODO: Validate status.
	return allErrs
}

// ValidateControllerRevisionName can be used to check whether the given ControllerRevision name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateControllerRevisionName = apivalidation.NameIsDNSSubdomain

// ValidateControllerRevision collects errors for the fields of state and returns those errors as an ErrorList. If the
// returned list is empty, state is valid. Validation is performed to ensure that state is a valid ObjectMeta, its name
// is valid, and that it doesn't exceed the MaxControllerRevisionSize.
func ValidateControllerRevision(revision *apps.ControllerRevision) field.ErrorList {
	errs := field.ErrorList{}

	errs = append(errs, apivalidation.ValidateObjectMeta(&revision.ObjectMeta, true, ValidateControllerRevisionName, field.NewPath("metadata"))...)
	if revision.Data == nil {
		errs = append(errs, field.Required(field.NewPath("data"), "data is mandatory"))
	}
	errs = append(errs, apivalidation.ValidateNonnegativeField(revision.Revision, field.NewPath("revision"))...)
	return errs
}

// ValidateControllerRevisionUpdate collects errors pertaining to the mutation of an ControllerRevision Object. If the
// returned ErrorList is empty the update operation is valid. Any mutation to the ControllerRevision's Data or Revision
// is considered to be invalid.
func ValidateControllerRevisionUpdate(newHistory, oldHistory *apps.ControllerRevision) field.ErrorList {
	errs := field.ErrorList{}

	errs = append(errs, apivalidation.ValidateObjectMetaUpdate(&newHistory.ObjectMeta, &oldHistory.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, ValidateControllerRevision(newHistory)...)
	errs = append(errs, apivalidation.ValidateImmutableField(newHistory.Data, oldHistory.Data, field.NewPath("data"))...)
	return errs
}
