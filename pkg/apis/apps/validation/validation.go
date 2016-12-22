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
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/apps"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/kubernetes/pkg/apis/meta/v1/validation"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/validation/field"
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

	// TODO: For now we're taking the safe route and disallowing all updates to
	// spec except for Replicas, for scaling, and Template.Spec.containers.image
	// for rolling-update. Enable others on a case by case basis.
	restoreReplicas := statefulSet.Spec.Replicas
	statefulSet.Spec.Replicas = oldStatefulSet.Spec.Replicas

	restoreContainers := statefulSet.Spec.Template.Spec.Containers
	statefulSet.Spec.Template.Spec.Containers = oldStatefulSet.Spec.Template.Spec.Containers

	if !reflect.DeepEqual(statefulSet.Spec, oldStatefulSet.Spec) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "updates to statefulset spec for fields other than 'replicas' and 'containers' are forbidden."))
	}
	statefulSet.Spec.Replicas = restoreReplicas
	statefulSet.Spec.Template.Spec.Containers = restoreContainers

	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(statefulSet.Spec.Replicas), field.NewPath("spec", "replicas"))...)
	containerErrs, _ := apivalidation.ValidateContainerUpdates(statefulSet.Spec.Template.Spec.Containers, oldStatefulSet.Spec.Template.Spec.Containers, field.NewPath("spec").Child("template").Child("containers"))
	allErrs = append(allErrs, containerErrs...)
	return allErrs
}

// ValidateStatefulSetStatusUpdate tests if required fields in the StatefulSet are set.
func ValidateStatefulSetStatusUpdate(statefulSet, oldStatefulSet *apps.StatefulSet) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&statefulSet.ObjectMeta, &oldStatefulSet.ObjectMeta, field.NewPath("metadata"))...)
	// TODO: Validate status.
	return allErrs
}
