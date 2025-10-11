/*
Copyright 2024 The Kubernetes Authors.

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
	"strconv"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/storagemigration"

	metaconditions "k8s.io/apimachinery/pkg/api/meta"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

func ValidateStorageVersionMigration(svm *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&svm.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)

	allErrs = checkAndAppendError(allErrs, field.NewPath("spec", "resource", "resource"), svm.Spec.Resource.Resource, "resource is required")

	return allErrs
}

func ValidateStorageVersionMigrationUpdate(newSVMBundle, oldSVMBundle *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := ValidateStorageVersionMigration(newSVMBundle)
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newSVMBundle.ObjectMeta, &oldSVMBundle.ObjectMeta, field.NewPath("metadata"))...)

	// prevent changes to the group, version and resource
	if newSVMBundle.Spec.Resource.Group != oldSVMBundle.Spec.Resource.Group {
		allErrs = append(allErrs, field.Invalid(field.NewPath("group"), newSVMBundle.Spec.Resource.Group, "field is immutable"))
	}
	if newSVMBundle.Spec.Resource.Resource != oldSVMBundle.Spec.Resource.Resource {
		allErrs = append(allErrs, field.Invalid(field.NewPath("resource"), newSVMBundle.Spec.Resource.Resource, "field is immutable"))
	}

	return allErrs
}

func ValidateStorageVersionMigrationStatusUpdate(newSVMBundle, oldSVMBundle *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newSVMBundle.ObjectMeta, &oldSVMBundle.ObjectMeta, field.NewPath("metadata"))

	fldPath := field.NewPath("status")

	// resource version should be a non-negative integer
	rvInt, err := convertResourceVersionToInt(newSVMBundle.Status.ResourceVersion)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newSVMBundle.Status.ResourceVersion, err.Error()))
	}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(rvInt, fldPath.Child("resourceVersion"))...)

	// TODO: after switching to metav1.Conditions in beta replace this validation with metav1.ValidateConditions
	allErrs = append(allErrs, metav1validation.ValidateConditions(newSVMBundle.Status.Conditions, fldPath.Child("conditions"))...)

	// resource version should not change once it has been set
	if len(oldSVMBundle.Status.ResourceVersion) != 0 && oldSVMBundle.Status.ResourceVersion != newSVMBundle.Status.ResourceVersion {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newSVMBundle.Status.ResourceVersion, "field is immutable"))
	}

	// at most one of success or failed may be true
	if isSuccessful(newSVMBundle) && isFailed(newSVMBundle) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), newSVMBundle.Status.Conditions, "Both success and failed conditions cannot be true at the same time"))
	}

	// running must be false when success is true or failed is true
	if isSuccessful(newSVMBundle) && isRunning(newSVMBundle) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), newSVMBundle.Status.Conditions, "Running condition cannot be true when success condition is true"))
	}
	if isFailed(newSVMBundle) && isRunning(newSVMBundle) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), newSVMBundle.Status.Conditions, "Running condition cannot be true when failed condition is true"))
	}

	// success cannot be set to false once it is true
	isOldSuccessful := isSuccessful(oldSVMBundle)
	if isOldSuccessful && !isSuccessful(newSVMBundle) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), newSVMBundle.Status.Conditions, "Success condition cannot be set to false once it is true"))
	}
	isOldFailed := isFailed(oldSVMBundle)
	if isOldFailed && !isFailed(newSVMBundle) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), newSVMBundle.Status.Conditions, "Failed condition cannot be set to false once it is true"))
	}

	return allErrs
}

func isSuccessful(svm *storagemigration.StorageVersionMigration) bool {
	successCondition := metaconditions.FindStatusCondition(svm.Status.Conditions, string(storagemigration.MigrationSucceeded))
	if successCondition != nil && successCondition.Status == metav1.ConditionTrue {
		return true
	}
	return false
}

func isFailed(svm *storagemigration.StorageVersionMigration) bool {
	failedCondition := metaconditions.FindStatusCondition(svm.Status.Conditions, string(storagemigration.MigrationFailed))
	if failedCondition != nil && failedCondition.Status == metav1.ConditionTrue {
		return true
	}
	return false
}

func isRunning(svm *storagemigration.StorageVersionMigration) bool {
	runningCondition := metaconditions.FindStatusCondition(svm.Status.Conditions, string(storagemigration.MigrationRunning))
	if runningCondition != nil && runningCondition.Status == metav1.ConditionTrue {
		return true
	}
	return false
}

func checkAndAppendError(allErrs field.ErrorList, fieldPath *field.Path, value string, message string) field.ErrorList {
	if len(value) == 0 {
		allErrs = append(allErrs, field.Required(fieldPath, message))
	}
	return allErrs
}

func convertResourceVersionToInt(rv string) (int64, error) {
	// initial value of RV is expected to be empty, which means the resource version is not set
	if len(rv) == 0 {
		return 0, nil
	}

	resourceVersion, err := strconv.ParseInt(rv, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse resource version %q: %w", rv, err)
	}

	return resourceVersion, nil
}
