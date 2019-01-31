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
	"strings"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const FieldImmutableErrorMsg string = `field is immutable`

const totalAnnotationSizeLimitB int = 256 * (1 << 10) // 256 kB

// BannedOwners is a black list of object that are not allowed to be owners.
var BannedOwners = map[schema.GroupVersionKind]struct{}{
	{Group: "", Version: "v1", Kind: "Event"}: {},
}

// ValidateClusterName can be used to check whether the given cluster name is valid.
var ValidateClusterName = NameIsDNS1035Label

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	var totalSize int64
	for k, v := range annotations {
		for _, msg := range validation.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, field.Invalid(fldPath, k, msg))
		}
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(totalAnnotationSizeLimitB) {
		allErrs = append(allErrs, field.TooLong(fldPath, "", totalAnnotationSizeLimitB))
	}
	return allErrs
}

func validateOwnerReference(ownerReference metav1.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	gvk := schema.FromAPIVersionAndKind(ownerReference.APIVersion, ownerReference.Kind)
	// gvk.Group is empty for the legacy group.
	if len(gvk.Version) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("apiVersion"), ownerReference.APIVersion, "version must not be empty"))
	}
	if len(gvk.Kind) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), ownerReference.Kind, "kind must not be empty"))
	}
	if len(ownerReference.Name) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), ownerReference.Name, "name must not be empty"))
	}
	if len(ownerReference.UID) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("uid"), ownerReference.UID, "uid must not be empty"))
	}
	if _, ok := BannedOwners[gvk]; ok {
		allErrs = append(allErrs, field.Invalid(fldPath, ownerReference, fmt.Sprintf("%s is disallowed from being an owner", gvk)))
	}
	return allErrs
}

func ValidateOwnerReferences(ownerReferences []metav1.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	controllerName := ""
	for _, ref := range ownerReferences {
		allErrs = append(allErrs, validateOwnerReference(ref, fldPath)...)
		if ref.Controller != nil && *ref.Controller {
			if controllerName != "" {
				allErrs = append(allErrs, field.Invalid(fldPath, ownerReferences,
					fmt.Sprintf("Only one reference can have Controller set to true. Found \"true\" in references for %v and %v", controllerName, ref.Name)))
			} else {
				controllerName = ref.Name
			}
		}
	}
	return allErrs
}

// Validate finalizer names
func ValidateFinalizerName(stringValue string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(stringValue) {
		allErrs = append(allErrs, field.Invalid(fldPath, stringValue, msg))
	}

	return allErrs
}

func ValidateNoNewFinalizers(newFinalizers []string, oldFinalizers []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	extra := sets.NewString(newFinalizers...).Difference(sets.NewString(oldFinalizers...))
	if len(extra) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath, fmt.Sprintf("no new finalizers can be added if the object is being deleted, found new finalizers %#v", extra.List())))
	}
	return allErrs
}

func ValidateImmutableField(newVal, oldVal interface{}, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !apiequality.Semantic.DeepEqual(oldVal, newVal) {
		allErrs = append(allErrs, field.Invalid(fldPath, newVal, FieldImmutableErrorMsg))
	}
	return allErrs
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
// It doesn't return an error for rootscoped resources with namespace, because namespace should already be cleared before.
func ValidateObjectMeta(objMeta *metav1.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	metadata, err := meta.Accessor(objMeta)
	if err != nil {
		allErrs := field.ErrorList{}
		allErrs = append(allErrs, field.Invalid(fldPath, objMeta, err.Error()))
		return allErrs
	}
	return ValidateObjectMetaAccessor(metadata, requiresNamespace, nameFn, fldPath)
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
// It doesn't return an error for rootscoped resources with namespace, because namespace should already be cleared before.
func ValidateObjectMetaAccessor(meta metav1.Object, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(meta.GetGenerateName()) != 0 {
		for _, msg := range nameFn(meta.GetGenerateName(), true) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("generateName"), meta.GetGenerateName(), msg))
		}
	}
	// If the generated name validates, but the calculated value does not, it's a problem with generation, and we
	// report it here. This may confuse users, but indicates a programming bug and still must be validated.
	// If there are multiple fields out of which one is required then add an or as a separator
	if len(meta.GetName()) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "name or generateName is required"))
	} else {
		for _, msg := range nameFn(meta.GetName(), false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), meta.GetName(), msg))
		}
	}
	if requiresNamespace {
		if len(meta.GetNamespace()) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), ""))
		} else {
			for _, msg := range ValidateNamespaceName(meta.GetNamespace(), false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), meta.GetNamespace(), msg))
			}
		}
	} else {
		if len(meta.GetNamespace()) != 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("namespace"), "not allowed on this type"))
		}
	}
	if len(meta.GetClusterName()) != 0 {
		for _, msg := range ValidateClusterName(meta.GetClusterName(), false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("clusterName"), meta.GetClusterName(), msg))
		}
	}
	allErrs = append(allErrs, ValidateNonnegativeField(meta.GetGeneration(), fldPath.Child("generation"))...)
	allErrs = append(allErrs, v1validation.ValidateLabels(meta.GetLabels(), fldPath.Child("labels"))...)
	allErrs = append(allErrs, ValidateAnnotations(meta.GetAnnotations(), fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidateOwnerReferences(meta.GetOwnerReferences(), fldPath.Child("ownerReferences"))...)
	allErrs = append(allErrs, ValidateFinalizers(meta.GetFinalizers(), fldPath.Child("finalizers"))...)
	return allErrs
}

// ValidateFinalizers tests if the finalizers name are valid, and if there are conflicting finalizers.
func ValidateFinalizers(finalizers []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	hasFinalizerOrphanDependents := false
	hasFinalizerDeleteDependents := false
	for _, finalizer := range finalizers {
		allErrs = append(allErrs, ValidateFinalizerName(finalizer, fldPath)...)
		if finalizer == metav1.FinalizerOrphanDependents {
			hasFinalizerOrphanDependents = true
		}
		if finalizer == metav1.FinalizerDeleteDependents {
			hasFinalizerDeleteDependents = true
		}
	}
	if hasFinalizerDeleteDependents && hasFinalizerOrphanDependents {
		allErrs = append(allErrs, field.Invalid(fldPath, finalizers, fmt.Sprintf("finalizer %s and %s cannot be both set", metav1.FinalizerOrphanDependents, metav1.FinalizerDeleteDependents)))
	}
	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(newMeta, oldMeta *metav1.ObjectMeta, fldPath *field.Path) field.ErrorList {
	newMetadata, err := meta.Accessor(newMeta)
	if err != nil {
		allErrs := field.ErrorList{}
		allErrs = append(allErrs, field.Invalid(fldPath, newMeta, err.Error()))
		return allErrs
	}
	oldMetadata, err := meta.Accessor(oldMeta)
	if err != nil {
		allErrs := field.ErrorList{}
		allErrs = append(allErrs, field.Invalid(fldPath, oldMeta, err.Error()))
		return allErrs
	}
	return ValidateObjectMetaAccessorUpdate(newMetadata, oldMetadata, fldPath)
}

func ValidateObjectMetaAccessorUpdate(newMeta, oldMeta metav1.Object, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// Finalizers cannot be added if the object is already being deleted.
	if oldMeta.GetDeletionTimestamp() != nil {
		allErrs = append(allErrs, ValidateNoNewFinalizers(newMeta.GetFinalizers(), oldMeta.GetFinalizers(), fldPath.Child("finalizers"))...)
	}

	// Reject updates that don't specify a resource version
	if len(newMeta.GetResourceVersion()) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newMeta.GetResourceVersion(), "must be specified for an update"))
	}

	// Generation shouldn't be decremented
	if newMeta.GetGeneration() < oldMeta.GetGeneration() {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), newMeta.GetGeneration(), "must not be decremented"))
	}

	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetName(), oldMeta.GetName(), fldPath.Child("name"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetNamespace(), oldMeta.GetNamespace(), fldPath.Child("namespace"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetUID(), oldMeta.GetUID(), fldPath.Child("uid"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetCreationTimestamp(), oldMeta.GetCreationTimestamp(), fldPath.Child("creationTimestamp"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetDeletionTimestamp(), oldMeta.GetDeletionTimestamp(), fldPath.Child("deletionTimestamp"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetDeletionGracePeriodSeconds(), oldMeta.GetDeletionGracePeriodSeconds(), fldPath.Child("deletionGracePeriodSeconds"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.GetClusterName(), oldMeta.GetClusterName(), fldPath.Child("clusterName"))...)

	allErrs = append(allErrs, v1validation.ValidateLabels(newMeta.GetLabels(), fldPath.Child("labels"))...)
	allErrs = append(allErrs, ValidateAnnotations(newMeta.GetAnnotations(), fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidateOwnerReferences(newMeta.GetOwnerReferences(), fldPath.Child("ownerReferences"))...)

	return allErrs
}
