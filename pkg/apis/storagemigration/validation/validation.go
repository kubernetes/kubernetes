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
	"strings"

	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/storagemigration"

	metaconditions "k8s.io/apimachinery/pkg/api/meta"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

func ValidateStorageVersionMigration(svm *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&svm.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))...)

	if len(svm.Spec.Resource.Resource) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "resource", "resource"), "resource is required to be set"))
	} else {
		// Same validations as APIService, Group must be a DNS1123 Subdomain and Resource must be DNS1035
		if errs := utilvalidation.IsDNS1035Label(svm.Spec.Resource.Resource); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "resource", "resource"), svm.Spec.Resource.Resource, strings.Join(errs, ",")))
		}
	}

	if len(svm.Spec.Resource.Group) != 0 {
		if errs := utilvalidation.IsDNS1123Subdomain(svm.Spec.Resource.Group); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "resource", "group"), svm.Spec.Resource.Group, strings.Join(errs, ",")))
		}
	}

	return allErrs
}

func ValidateStorageVersionMigrationUpdate(newSVMBundle, oldSVMBundle *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := ValidateStorageVersionMigration(newSVMBundle)
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newSVMBundle.ObjectMeta, &oldSVMBundle.ObjectMeta, field.NewPath("metadata"))...)

	// prevent changes to the spec
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(newSVMBundle.Spec, oldSVMBundle.Spec, field.NewPath("spec"))...)

	return allErrs
}

func ValidateStorageVersionMigrationStatusUpdate(newSVMBundle, oldSVMBundle *storagemigration.StorageVersionMigration) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newSVMBundle.ObjectMeta, &oldSVMBundle.ObjectMeta, field.NewPath("metadata"))

	fldPath := field.NewPath("status")

	// resource version should be a non-negative integer
	cmp, err := resourceversion.CompareResourceVersion(newSVMBundle.Status.ResourceVersion, newSVMBundle.Status.ResourceVersion)
	if err != nil || cmp != 0 {
		if err == nil {
			err = fmt.Errorf("unable to compare resource versions, %s is not equal to %s", newSVMBundle.Status.ResourceVersion, newSVMBundle.Status.ResourceVersion)
		}
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newSVMBundle.Status.ResourceVersion, err.Error()))
	}

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
