/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/resource"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// TODO: delete this global variable when we enable the validation of common
// fields by default.
var RepairMalformedUpdates bool = apimachineryvalidation.RepairMalformedUpdates

const isNegativeErrorMsg string = apimachineryvalidation.IsNegativeErrorMsg
const isInvalidQuotaResource string = `must be a standard resource for quota`
const fieldImmutableErrorMsg string = apimachineryvalidation.FieldImmutableErrorMsg
const isNotIntegerErrorMsg string = `must be an integer`
const isNotPositiveErrorMsg string = `must be greater than zero`

// BannedOwners is a black list of object that are not allowed to be owners.
var BannedOwners = apimachineryvalidation.BannedOwners

// ValidateHasLabel requires that metav1.ObjectMeta has a Label with key and expectedValue
func ValidateHasLabel(meta metav1.ObjectMeta, fldPath *field.Path, key, expectedValue string) field.ErrorList {
	allErrs := field.ErrorList{}
	actualValue, found := meta.Labels[key]
	if !found {
		allErrs = append(allErrs, field.Required(fldPath.Child("labels").Key(key),
			fmt.Sprintf("must be '%s'", expectedValue)))
		return allErrs
	}
	if actualValue != expectedValue {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("labels").Key(key), meta.Labels,
			fmt.Sprintf("must be '%s'", expectedValue)))
	}
	return allErrs
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	return apimachineryvalidation.ValidateAnnotations(annotations, fldPath)
}

func ValidateDNS1123Label(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Label(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// ValidateDNS1123Subdomain validates that a name is a proper DNS subdomain.
func ValidateDNS1123Subdomain(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Subdomain(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

// ValidateNameFunc validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names. Prefix is true
// if the name will have a value appended to it.  If the name is not valid,
// this returns a list of descriptions of individual characteristics of the
// value that were not valid.  Otherwise this returns an empty list or nil.
type ValidateNameFunc apimachineryvalidation.ValidateNameFunc

// ValidateClusterName can be used to check whether the given cluster name is valid.
var ValidateClusterName = apimachineryvalidation.ValidateClusterName

// ValidateClassName can be used to check whether the given class name is valid.
// It is defined here to avoid import cycle between pkg/apis/storage/validation
// (where it should be) and this file.
var ValidateClassName = NameIsDNSSubdomain

// ValidatePiorityClassName can be used to check whether the given priority
// class name is valid.
var ValidatePriorityClassName = NameIsDNSSubdomain

// TODO update all references to these functions to point to the apimachineryvalidation ones
// NameIsDNSSubdomain is a ValidateNameFunc for names that must be a DNS subdomain.
func NameIsDNSSubdomain(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNSSubdomain(name, prefix)
}

// NameIsDNS1035Label is a ValidateNameFunc for names that must be a DNS 952 label.
func NameIsDNS1035Label(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNS1035Label(name, prefix)
}

// Validates that given value is not negative.
func ValidateNonnegativeField(value int64, fldPath *field.Path) field.ErrorList {
	return apimachineryvalidation.ValidateNonnegativeField(value, fldPath)
}

// Validates that a Quantity is not negative
func ValidateNonnegativeQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNegativeErrorMsg))
	}
	return allErrs
}

// Validates that a Quantity is positive
func ValidatePositiveQuantityValue(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNotPositiveErrorMsg))
	}
	return allErrs
}

func ValidateImmutableField(newVal, oldVal interface{}, fldPath *field.Path) field.ErrorList {
	return apimachineryvalidation.ValidateImmutableField(newVal, oldVal, fldPath)
}

func ValidateImmutableAnnotation(newVal string, oldVal string, annotation string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if oldVal != newVal {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("annotations", annotation), newVal, fieldImmutableErrorMsg))
	}
	return allErrs
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
// It doesn't return an error for rootscoped resources with namespace, because namespace should already be cleared before.
// TODO: Remove calls to this method scattered in validations of specific resources, e.g., ValidatePodUpdate.
func ValidateObjectMeta(meta *metav1.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMeta(meta, requiresNamespace, apimachineryvalidation.ValidateNameFunc(nameFn), fldPath)
	// run additional checks for the finalizer name
	for i := range meta.Finalizers {
		allErrs = append(allErrs, validateKubeFinalizerName(string(meta.Finalizers[i]), fldPath.Child("finalizers").Index(i))...)
	}
	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(newMeta, oldMeta *metav1.ObjectMeta, fldPath *field.Path) field.ErrorList {
	allErrs := apimachineryvalidation.ValidateObjectMetaUpdate(newMeta, oldMeta, fldPath)
	// run additional checks for the finalizer name
	for i := range newMeta.Finalizers {
		allErrs = append(allErrs, validateKubeFinalizerName(string(newMeta.Finalizers[i]), fldPath.Child("finalizers").Index(i))...)
	}

	return allErrs
}

func IsDecremented(update, old *int32) bool {
	if update == nil && old != nil {
		return true
	}
	if update == nil || old == nil {
		return false
	}
	return *update < *old
}
