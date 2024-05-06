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
	"regexp"
	"strconv"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/storagemigration"

	corev1 "k8s.io/api/core/v1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

func ValidateStorageVersionMigration(svm *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&svm.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)

	allErrs = checkAndAppendError(allErrs, field.NewPath("spec", "resource", "resource"), svm.Spec.Resource.Resource, "resource is required")
	allErrs = checkAndAppendError(allErrs, field.NewPath("spec", "resource", "version"), svm.Spec.Resource.Version, "version is required")

	return allErrs
}

func ValidateStorageVersionMigrationUpdate(newSVMBundle, oldSVMBundle *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := ValidateStorageVersionMigration(newSVMBundle)
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newSVMBundle.ObjectMeta, &oldSVMBundle.ObjectMeta, field.NewPath("metadata"))...)

	// prevent changes to the group, version and resource
	if newSVMBundle.Spec.Resource.Group != oldSVMBundle.Spec.Resource.Group {
		allErrs = append(allErrs, field.Invalid(field.NewPath("group"), newSVMBundle.Spec.Resource.Group, "field is immutable"))
	}
	if newSVMBundle.Spec.Resource.Version != oldSVMBundle.Spec.Resource.Version {
		allErrs = append(allErrs, field.Invalid(field.NewPath("version"), newSVMBundle.Spec.Resource.Version, "field is immutable"))
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
	allErrs = append(allErrs, validateConditions(newSVMBundle.Status.Conditions, fldPath.Child("conditions"))...)

	// resource version should not change once it has been set
	if len(oldSVMBundle.Status.ResourceVersion) != 0 && oldSVMBundle.Status.ResourceVersion != newSVMBundle.Status.ResourceVersion {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newSVMBundle.Status.ResourceVersion, "resourceVersion cannot be updated"))
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
	successCondition := getCondition(svm, storagemigration.MigrationSucceeded)
	if successCondition != nil && successCondition.Status == corev1.ConditionTrue {
		return true
	}
	return false
}

func isFailed(svm *storagemigration.StorageVersionMigration) bool {
	failedCondition := getCondition(svm, storagemigration.MigrationFailed)
	if failedCondition != nil && failedCondition.Status == corev1.ConditionTrue {
		return true
	}
	return false
}

func isRunning(svm *storagemigration.StorageVersionMigration) bool {
	runningCondition := getCondition(svm, storagemigration.MigrationRunning)
	if runningCondition != nil && runningCondition.Status == corev1.ConditionTrue {
		return true
	}
	return false
}

func getCondition(svm *storagemigration.StorageVersionMigration, conditionType storagemigration.MigrationConditionType) *storagemigration.MigrationCondition {
	for _, c := range svm.Status.Conditions {
		if c.Type == conditionType {
			return &c
		}
	}

	return nil
}

func validateConditions(conditions []storagemigration.MigrationCondition, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	conditionTypeToFirstIndex := map[string]int{}
	for i, condition := range conditions {
		if _, ok := conditionTypeToFirstIndex[string(condition.Type)]; ok {
			allErrs = append(allErrs, field.Duplicate(fldPath.Index(i).Child("type"), condition.Type))
		} else {
			conditionTypeToFirstIndex[string(condition.Type)] = i
		}

		allErrs = append(allErrs, validateCondition(condition, fldPath.Index(i))...)
	}

	return allErrs
}

func validateCondition(condition storagemigration.MigrationCondition, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	var validConditionStatuses = sets.NewString(string(metav1.ConditionTrue), string(metav1.ConditionFalse), string(metav1.ConditionUnknown))

	// type is set and is a valid format
	allErrs = append(allErrs, metav1validation.ValidateLabelName(string(condition.Type), fldPath.Child("type"))...)

	// status is set and is an accepted value
	if !validConditionStatuses.Has(string(condition.Status)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("status"), condition.Status, validConditionStatuses.List()))
	}

	if condition.LastUpdateTime.IsZero() {
		allErrs = append(allErrs, field.Required(fldPath.Child("lastTransitionTime"), "must be set"))
	}

	if len(condition.Reason) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("reason"), "must be set"))
	} else {
		for _, currErr := range isValidConditionReason(condition.Reason) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("reason"), condition.Reason, currErr))
		}

		const maxReasonLen int = 1 * 1024 // 1024
		if len(condition.Reason) > maxReasonLen {
			allErrs = append(allErrs, field.TooLong(fldPath.Child("reason"), condition.Reason, maxReasonLen))
		}
	}

	const maxMessageLen int = 32 * 1024 // 32768
	if len(condition.Message) > maxMessageLen {
		allErrs = append(allErrs, field.TooLong(fldPath.Child("message"), condition.Message, maxMessageLen))
	}

	return allErrs
}
func isValidConditionReason(value string) []string {
	const conditionReasonFmt string = "[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?"
	const conditionReasonErrMsg string = "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_'"
	var conditionReasonRegexp = regexp.MustCompile("^" + conditionReasonFmt + "$")

	if !conditionReasonRegexp.MatchString(value) {
		return []string{validation.RegexError(conditionReasonErrMsg, conditionReasonFmt, "my_name", "MY_NAME", "MyName", "ReasonA,ReasonB", "ReasonA:ReasonB")}
	}
	return nil
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
