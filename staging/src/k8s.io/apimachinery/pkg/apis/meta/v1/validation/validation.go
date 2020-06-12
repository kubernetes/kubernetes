/*
Copyright 2015 The Kubernetes Authors.

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
	"unicode"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func ValidateLabelSelector(ps *metav1.LabelSelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if ps == nil {
		return allErrs
	}
	allErrs = append(allErrs, ValidateLabels(ps.MatchLabels, fldPath.Child("matchLabels"))...)
	for i, expr := range ps.MatchExpressions {
		allErrs = append(allErrs, ValidateLabelSelectorRequirement(expr, fldPath.Child("matchExpressions").Index(i))...)
	}
	return allErrs
}

func ValidateLabelSelectorRequirement(sr metav1.LabelSelectorRequirement, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch sr.Operator {
	case metav1.LabelSelectorOpIn, metav1.LabelSelectorOpNotIn:
		if len(sr.Values) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified when `operator` is 'In' or 'NotIn'"))
		}
	case metav1.LabelSelectorOpExists, metav1.LabelSelectorOpDoesNotExist:
		if len(sr.Values) > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("values"), "may not be specified when `operator` is 'Exists' or 'DoesNotExist'"))
		}
	default:
		allErrs = append(allErrs, field.Invalid(fldPath.Child("operator"), sr.Operator, "not a valid selector operator"))
	}
	allErrs = append(allErrs, ValidateLabelName(sr.Key, fldPath.Child("key"))...)
	return allErrs
}

// ValidateLabelName validates that the label name is correctly defined.
func ValidateLabelName(labelName string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(labelName) {
		allErrs = append(allErrs, field.Invalid(fldPath, labelName, msg))
	}
	return allErrs
}

// ValidateLabels validates that a set of labels are correctly defined.
func ValidateLabels(labels map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for k, v := range labels {
		allErrs = append(allErrs, ValidateLabelName(k, fldPath)...)
		for _, msg := range validation.IsValidLabelValue(v) {
			allErrs = append(allErrs, field.Invalid(fldPath, v, msg))
		}
	}
	return allErrs
}

func ValidateDeleteOptions(options *metav1.DeleteOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if options.OrphanDependents != nil && options.PropagationPolicy != nil {
		allErrs = append(allErrs, field.Invalid(field.NewPath("propagationPolicy"), options.PropagationPolicy, "orphanDependents and deletionPropagation cannot be both set"))
	}
	if options.PropagationPolicy != nil &&
		*options.PropagationPolicy != metav1.DeletePropagationForeground &&
		*options.PropagationPolicy != metav1.DeletePropagationBackground &&
		*options.PropagationPolicy != metav1.DeletePropagationOrphan {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("propagationPolicy"), options.PropagationPolicy, []string{string(metav1.DeletePropagationForeground), string(metav1.DeletePropagationBackground), string(metav1.DeletePropagationOrphan), "nil"}))
	}
	allErrs = append(allErrs, ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...)
	return allErrs
}

func ValidateCreateOptions(options *metav1.CreateOptions) field.ErrorList {
	return append(
		ValidateFieldManager(options.FieldManager, field.NewPath("fieldManager")),
		ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...,
	)
}

func ValidateUpdateOptions(options *metav1.UpdateOptions) field.ErrorList {
	return append(
		ValidateFieldManager(options.FieldManager, field.NewPath("fieldManager")),
		ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...,
	)
}

func ValidatePatchOptions(options *metav1.PatchOptions, patchType types.PatchType) field.ErrorList {
	allErrs := field.ErrorList{}
	if patchType != types.ApplyPatchType {
		if options.Force != nil {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("force"), "may not be specified for non-apply patch"))
		}
	} else {
		if options.FieldManager == "" {
			// This field is defaulted to "kubectl" by kubectl, but HAS TO be explicitly set by controllers.
			allErrs = append(allErrs, field.Required(field.NewPath("fieldManager"), "is required for apply patch"))
		}
	}
	allErrs = append(allErrs, ValidateFieldManager(options.FieldManager, field.NewPath("fieldManager"))...)
	allErrs = append(allErrs, ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...)
	return allErrs
}

var FieldManagerMaxLength = 128

// ValidateFieldManager valides that the fieldManager is the proper length and
// only has printable characters.
func ValidateFieldManager(fieldManager string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// the field can not be set as a `*string`, so a empty string ("") is
	// considered as not set and is defaulted by the rest of the process
	// (unless apply is used, in which case it is required).
	if len(fieldManager) > FieldManagerMaxLength {
		allErrs = append(allErrs, field.TooLong(fldPath, fieldManager, FieldManagerMaxLength))
	}
	// Verify that all characters are printable.
	for i, r := range fieldManager {
		if !unicode.IsPrint(r) {
			allErrs = append(allErrs, field.Invalid(fldPath, fieldManager, fmt.Sprintf("invalid character %#U (at position %d)", r, i)))
		}
	}

	return allErrs
}

var allowedDryRunValues = sets.NewString(metav1.DryRunAll)

// ValidateDryRun validates that a dryRun query param only contains allowed values.
func ValidateDryRun(fldPath *field.Path, dryRun []string) field.ErrorList {
	allErrs := field.ErrorList{}
	if !allowedDryRunValues.HasAll(dryRun...) {
		allErrs = append(allErrs, field.NotSupported(fldPath, dryRun, allowedDryRunValues.List()))
	}
	return allErrs
}

const UninitializedStatusUpdateErrorMsg string = `must not update status when the object is uninitialized`

// ValidateTableOptions returns any invalid flags on TableOptions.
func ValidateTableOptions(opts *metav1.TableOptions) field.ErrorList {
	var allErrs field.ErrorList
	switch opts.IncludeObject {
	case metav1.IncludeMetadata, metav1.IncludeNone, metav1.IncludeObject, "":
	default:
		allErrs = append(allErrs, field.Invalid(field.NewPath("includeObject"), opts.IncludeObject, "must be 'Metadata', 'Object', 'None', or empty"))
	}
	return allErrs
}

func ValidateManagedFields(fieldsList []metav1.ManagedFieldsEntry, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for _, fields := range fieldsList {
		switch fields.Operation {
		case metav1.ManagedFieldsOperationApply, metav1.ManagedFieldsOperationUpdate:
		default:
			allErrs = append(allErrs, field.Invalid(fldPath.Child("operation"), fields.Operation, "must be `Apply` or `Update`"))
		}
		if len(fields.FieldsType) > 0 && fields.FieldsType != "FieldsV1" {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("fieldsType"), fields.FieldsType, "must be `FieldsV1`"))
		}
	}
	return allErrs
}
