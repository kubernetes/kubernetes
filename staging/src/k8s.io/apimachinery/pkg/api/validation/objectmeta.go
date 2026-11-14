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
	"context"
	"fmt"
	"strings"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// FieldImmutableErrorMsg is a error message for field is immutable.
const FieldImmutableErrorMsg string = `field is immutable`

const TotalAnnotationSizeLimitB int = 256 * (1 << 10) // 256 kB

// BannedOwners is a black list of object that are not allowed to be owners.
var BannedOwners = map[schema.GroupVersionKind]struct{}{
	{Group: "", Version: "v1", Kind: "Event"}: {},
}

func hasUpperOrNonASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'A' && c <= 'Z') || c >= 0x80 {
			return true
		}
	}
	return false
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for k := range annotations {
		// The rule is QualifiedName except that case doesn't matter, so convert to lowercase before checking.
		checkKey := k
		if hasUpperOrNonASCII(k) {
			checkKey = strings.ToLower(k)
		}
		for _, msg := range validation.IsQualifiedName(checkKey) {
			allErrs = append(allErrs, field.Invalid(fldPath, k, msg).WithOrigin("format=k8s-label-key"))
		}
	}
	if err := ValidateAnnotationsSize(annotations); err != nil {
		allErrs = append(allErrs, field.TooLong(fldPath, "" /*unused*/, TotalAnnotationSizeLimitB))
	}
	return allErrs
}

func ValidateAnnotationsSize(annotations map[string]string) error {
	var totalSize int64
	for k, v := range annotations {
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(TotalAnnotationSizeLimitB) {
		return fmt.Errorf("annotations size %d is larger than limit %d", totalSize, TotalAnnotationSizeLimitB)
	}
	return nil
}

func validateOwnerReference(ownerReference metav1.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	gv, err := schema.ParseGroupVersion(ownerReference.APIVersion)
	// gvk.Group is empty for the legacy group.
	if len(ownerReference.APIVersion) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("apiVersion"), "must not be empty").MarkCoveredByDeclarative())
	} else if err != nil || len(gv.Version) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("apiVersion"), ownerReference.APIVersion, "must be <group>/<version> or <version>"))
	}
	if len(ownerReference.Kind) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("kind"), "must not be empty").MarkCoveredByDeclarative())
	}
	if len(ownerReference.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "must not be empty").MarkCoveredByDeclarative())
	}
	if len(ownerReference.UID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("uid"), "must not be empty").MarkCoveredByDeclarative())
	}
	gvk := gv.WithKind(ownerReference.Kind)
	if _, ok := BannedOwners[gvk]; ok {
		allErrs = append(allErrs, field.Invalid(fldPath, ownerReference, fmt.Sprintf("%s is disallowed from being an owner", gvk)))
	}
	return allErrs
}

// ValidateOwnerReferences validates that a set of owner references are correctly defined.
func ValidateOwnerReferences(ownerReferences []metav1.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	firstControllerName := ""
	for idx, ref := range ownerReferences {
		allErrs = append(allErrs, validateOwnerReference(ref, fldPath.Index(idx))...)
		if ref.Controller != nil && *ref.Controller {
			curControllerName := ref.Kind + "/" + ref.Name
			if firstControllerName != "" {
				allErrs = append(allErrs, field.Invalid(fldPath, ownerReferences,
					fmt.Sprintf("Only one reference can have Controller set to true. Found \"true\" in references for %v and %v", firstControllerName, curControllerName)))
			} else {
				firstControllerName = curControllerName
			}
		}
	}
	return allErrs
}

// ValidateFinalizerName validates finalizer names.
func ValidateFinalizerName(stringValue string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(stringValue) {
		allErrs = append(allErrs, field.Invalid(fldPath, stringValue, msg))
	}

	return allErrs
}

// ValidateNoNewFinalizers validates the new finalizers has no new finalizers compare to old finalizers.
func ValidateNoNewFinalizers(newFinalizers []string, oldFinalizers []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	extra := sets.NewString(newFinalizers...).Difference(sets.NewString(oldFinalizers...))
	if len(extra) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath, fmt.Sprintf("no new finalizers can be added if the object is being deleted, found new finalizers %#v", extra.List())))
	}
	return allErrs
}

// ValidateImmutableField validates the new value and the old value are deeply equal.
func ValidateImmutableField(newVal, oldVal interface{}, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !apiequality.Semantic.DeepEqual(oldVal, newVal) {
		allErrs = append(allErrs, field.Invalid(fldPath, newVal, FieldImmutableErrorMsg))
	}
	return allErrs
}

// ValidateObjectMetaDeclaratively validates an ObjectMeta instance declaratively and deduplicates handwritten errors.
// betaEnabled controls whether declarative validation rules at the Beta stability level are enforced.
// NOTE: This method should be used in the types for which declarative validation is not enabled yet or cannot be enabled. The types
// for which declarative validation is enabled and valdiation code is generated must use ValidateObjectMeta and ValidateObjectMetaUpdate.
func ValidateObjectMetaDeclaratively(ctx context.Context, op operation.Type, obj, oldObj *metav1.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path, betaEnabled bool) field.ErrorList {
	var errs field.ErrorList
	switch op {
	case operation.Create:
		errs = ValidateObjectMeta(obj, requiresNamespace, nameFn, fldPath)
	case operation.Update:
		errs = ValidateObjectMetaUpdate(obj, oldObj, fldPath)
	}
	dvErrs := v1validation.Validate_ObjectMeta(ctx, operation.Operation{Type: op}, fldPath, obj, oldObj)
	enforcedDVErrs := validate.FilterEnforcedDeclarativeErrors(ctx, dvErrs, betaEnabled)
	errs = errs.MarkFromImperative()
	errs = validate.FilterCoveredHandwrittenErrors(ctx, errs, enforcedDVErrs, betaEnabled)
	return append(errs, enforcedDVErrs...)
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
func ValidateObjectMeta(objMeta *metav1.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	metadata, err := meta.Accessor(objMeta)
	if err != nil {
		var allErrs field.ErrorList
		allErrs = append(allErrs, field.Invalid(fldPath, objMeta, err.Error()))
		return allErrs
	}
	return ValidateObjectMetaAccessor(metadata, requiresNamespace, nameFn, fldPath)
}

// objectMetaValidationOptions defines behavioral modifications for validating
// an ObjectMeta.
type objectMetaValidationOptions struct {
	/* nothing here yet */
}

// ObjectMetaValidationOption specifies a behavioral modifier for
// ValidateObjectMetaWithOpts and ValidateObjectMetaAccessorWithOpts.
type ObjectMetaValidationOption func(opts *objectMetaValidationOptions)

// ValidateObjectMetaWithOpts validates an object's metadata on creation. It
// expects that name generation has already been performed, so name validation
// is always executed.
//
// This is similar to ValidateObjectMeta, but uses options to buy future-safety
// and uses different signature for the name validation function.  It also does
// not directly validate the generateName field, because name generation
// should have already been performed and it is the result of that generastion
// that must conform to the nameFn.
func ValidateObjectMetaWithOpts(objMeta *metav1.ObjectMeta, isNamespaced bool, nameFn ValidateNameFuncWithErrors, fldPath *field.Path, options ...ObjectMetaValidationOption) field.ErrorList {
	metadata, err := meta.Accessor(objMeta)
	if err != nil {
		var allErrs field.ErrorList
		allErrs = append(allErrs, field.InternalError(fldPath, err))
		return allErrs
	}
	return ValidateObjectMetaAccessorWithOpts(metadata, isNamespaced, nameFn, fldPath, options...)
}

// ValidateObjectMetaAccessor validates an object's metadata on creation. It expects that name generation has already
// been performed.
func ValidateObjectMetaAccessor(meta metav1.Object, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

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

	return append(allErrs, validateObjectMetaAccessorWithOptsCommon(meta, requiresNamespace, fldPath, nil)...)
}

// ValidateObjectMetaAccessorWithOpts validates an object's metadata on
// creation. It expects that name generation has already been performed, so
// name validation is always executed.
//
// This is similar to ValidateObjectMetaAccessor, but uses options to buy
// future-safety and uses different signature for the name validation function.
// It also does not directly validate the generateName field, because name
// generation should have already been performed and it is the result of that
// generastion that must conform to the nameFn.
func ValidateObjectMetaAccessorWithOpts(meta metav1.Object, isNamespaced bool, nameFn ValidateNameFuncWithErrors, fldPath *field.Path, options ...ObjectMetaValidationOption) field.ErrorList {
	opts := objectMetaValidationOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	var allErrs field.ErrorList

	// generateName is not directly validated here. Types can have
	// different rules for name generation, and the nameFn is for validating
	// the post-generation data, not the input. In the past we assumed that
	// name generation was always "append 5 random characters", but that's not
	// NECESSARILY true. Also, the nameFn should always be considering the max
	// length of the name, and it doesn't know enough about the name generation
	// to do that. Also, given a bad generateName, the user will get errors
	// for both the generateName and name fields. We will focus validation on
	// the name field, which should give a better UX overall.
	// TODO(thockin): should we do a max-length check here? e.g. 1K or 4K?

	if len(meta.GetGenerateName()) != 0 && len(meta.GetName()) == 0 {
		allErrs = append(allErrs,
			field.InternalError(fldPath.Child("name"), fmt.Errorf("generateName was specified (%q), but no name was generated", meta.GetGenerateName())))
	}
	if len(meta.GetName()) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "name or generateName is required"))
	} else {
		allErrs = append(allErrs, nameFn(fldPath.Child("name"), meta.GetName())...)
	}

	return append(allErrs, validateObjectMetaAccessorWithOptsCommon(meta, isNamespaced, fldPath, &opts)...)
}

// validateObjectMetaAccessorWithOptsCommon is a shared function for validating
// the parts of an ObjectMeta with are handled the same in both paths..
func validateObjectMetaAccessorWithOptsCommon(meta metav1.Object, isNamespaced bool, fldPath *field.Path, _ *objectMetaValidationOptions) field.ErrorList {
	var allErrs field.ErrorList

	ns := meta.GetNamespace()
	if isNamespaced {
		if len(ns) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), ""))
		} else {
			for _, msg := range ValidateNamespaceName(ns, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), ns, msg))
			}
		}
	} else {
		if len(ns) != 0 {
			// TODO(thockin): change to "may not be specified on this type" or something
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("namespace"), "not allowed on this type"))
		}
	}

	if gen := meta.GetGeneration(); gen < 0 {
		allErrs = append(allErrs, ValidateNonnegativeField(gen, fldPath.Child("generation")).MarkCoveredByDeclarative()...)
	}
	if labels := meta.GetLabels(); len(labels) > 0 {
		allErrs = append(allErrs, v1validation.ValidateLabels(labels, fldPath.Child("labels"))...)
	}
	if annotations := meta.GetAnnotations(); len(annotations) > 0 {
		allErrs = append(allErrs, ValidateAnnotations(annotations, fldPath.Child("annotations"))...)
	}
	if ownerRefs := meta.GetOwnerReferences(); len(ownerRefs) > 0 {
		allErrs = append(allErrs, ValidateOwnerReferences(ownerRefs, fldPath.Child("ownerReferences"))...)
	}
	if finalizers := meta.GetFinalizers(); len(finalizers) > 0 {
		allErrs = append(allErrs, ValidateFinalizers(finalizers, fldPath.Child("finalizers"))...)
	}
	if managedFields := meta.GetManagedFields(); len(managedFields) > 0 {
		allErrs = append(allErrs, v1validation.ValidateManagedFields(managedFields, fldPath.Child("managedFields"), v1validation.CoveredByDeclarative)...)
	}
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

// ValidateObjectMetaUpdate validates an object's metadata when updated.
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

// ValidateObjectMetaAccessorUpdate validates an object's metadata when updated.
func ValidateObjectMetaAccessorUpdate(newMeta, oldMeta metav1.Object, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	oldDelTS := oldMeta.GetDeletionTimestamp()
	// Finalizers cannot be added if the object is already being deleted.
	if oldDelTS != nil {
		allErrs = append(allErrs, ValidateNoNewFinalizers(newMeta.GetFinalizers(), oldMeta.GetFinalizers(), fldPath.Child("finalizers"))...)
	}

	// Reject updates that don't specify a resource version
	if len(newMeta.GetResourceVersion()) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newMeta.GetResourceVersion(), "must be specified for an update"))
	}

	// Generation shouldn't be decremented
	newGen, oldGen := newMeta.GetGeneration(), oldMeta.GetGeneration()
	if newGen < 0 {
		allErrs = append(allErrs, ValidateNonnegativeField(newGen, fldPath.Child("generation")).MarkCoveredByDeclarative()...)
	}
	if oldGen != 0 && newGen == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), newGen, "field cannot be cleared once set").WithOrigin("update").MarkCoveredByDeclarative())
	} else if newGen < oldGen {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), newGen, "must not be decremented").WithOrigin("monotonic").MarkCoveredByDeclarative())
	}

	if newName, oldName := newMeta.GetName(), oldMeta.GetName(); newName != oldName {
		allErrs = append(allErrs, ValidateImmutableField(newName, oldName, fldPath.Child("name"))...)
	}
	if newNS, oldNS := newMeta.GetNamespace(), oldMeta.GetNamespace(); newNS != oldNS {
		allErrs = append(allErrs, ValidateImmutableField(newNS, oldNS, fldPath.Child("namespace")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	}
	if newUID, oldUID := newMeta.GetUID(), oldMeta.GetUID(); newUID != oldUID {
		allErrs = append(allErrs, ValidateImmutableField(newUID, oldUID, fldPath.Child("uid")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	}
	if newCT, oldCT := newMeta.GetCreationTimestamp(), oldMeta.GetCreationTimestamp(); !newCT.Time.Equal(oldCT.Time) {
		allErrs = append(allErrs, ValidateImmutableField(newCT, oldCT, fldPath.Child("creationTimestamp")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	}
	if newDelTS := newMeta.GetDeletionTimestamp(); (newDelTS != nil || oldDelTS != nil) && (newDelTS == nil || oldDelTS == nil || !newDelTS.Equal(oldDelTS)) {
		allErrs = append(allErrs, ValidateImmutableField(newDelTS, oldDelTS, fldPath.Child("deletionTimestamp")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	}
	if newDG, oldDG := newMeta.GetDeletionGracePeriodSeconds(), oldMeta.GetDeletionGracePeriodSeconds(); (newDG != nil || oldDG != nil) && (newDG == nil || oldDG == nil || *newDG != *oldDG) {
		allErrs = append(allErrs, ValidateImmutableField(newDG, oldDG, fldPath.Child("deletionGracePeriodSeconds")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	}

	if labels := newMeta.GetLabels(); len(labels) > 0 {
		allErrs = append(allErrs, v1validation.ValidateLabels(labels, fldPath.Child("labels"))...)
	}
	if annotations := newMeta.GetAnnotations(); len(annotations) > 0 {
		allErrs = append(allErrs, ValidateAnnotations(annotations, fldPath.Child("annotations"))...)
	}
	if ownerRefs := newMeta.GetOwnerReferences(); len(ownerRefs) > 0 {
		allErrs = append(allErrs, ValidateOwnerReferences(ownerRefs, fldPath.Child("ownerReferences"))...)
	}
	if managedFields := newMeta.GetManagedFields(); len(managedFields) > 0 {
		allErrs = append(allErrs, v1validation.ValidateManagedFields(managedFields, fldPath.Child("managedFields"), v1validation.CoveredByDeclarative)...)
	}

	return allErrs
}
