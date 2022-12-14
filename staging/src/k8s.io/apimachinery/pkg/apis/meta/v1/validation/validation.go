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
	"regexp"
	"unicode"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// LabelSelectorValidationOptions is a struct that can be passed to ValidateLabelSelector to record the validate options
type LabelSelectorValidationOptions struct {
	// Allow invalid label value in selector
	AllowInvalidLabelValueInSelector bool
}

// LabelSelectorHasInvalidLabelValue returns true if the given selector contains an invalid label value in a match expression.
// This is useful for determining whether AllowInvalidLabelValueInSelector should be set to true when validating an update
// based on existing persisted invalid values.
func LabelSelectorHasInvalidLabelValue(ps *metav1.LabelSelector) bool {
	if ps == nil {
		return false
	}
	for _, e := range ps.MatchExpressions {
		for _, v := range e.Values {
			if len(validation.IsValidLabelValue(v)) > 0 {
				return true
			}
		}
	}
	return false
}

// ValidateLabelSelector validate the LabelSelector according to the opts and returns any validation errors.
// opts.AllowInvalidLabelValueInSelector is only expected to be set to true when required for backwards compatibility with existing invalid data.
func ValidateLabelSelector(ps *metav1.LabelSelector, opts LabelSelectorValidationOptions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if ps == nil {
		return allErrs
	}
	allErrs = append(allErrs, ValidateLabels(ps.MatchLabels, fldPath.Child("matchLabels"))...)
	for i, expr := range ps.MatchExpressions {
		allErrs = append(allErrs, ValidateLabelSelectorRequirement(expr, opts, fldPath.Child("matchExpressions").Index(i))...)
	}
	return allErrs
}

// ValidateLabelSelectorRequirement validate the requirement according to the opts and returns any validation errors.
// opts.AllowInvalidLabelValueInSelector is only expected to be set to true when required for backwards compatibility with existing invalid data.
func ValidateLabelSelectorRequirement(sr metav1.LabelSelectorRequirement, opts LabelSelectorValidationOptions, fldPath *field.Path) field.ErrorList {
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
	if !opts.AllowInvalidLabelValueInSelector {
		for valueIndex, value := range sr.Values {
			for _, msg := range validation.IsValidLabelValue(value) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("values").Index(valueIndex), value, msg))
			}
		}
	}
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
	//lint:file-ignore SA1019 Keep validation for deprecated OrphanDependents option until it's being removed
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
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateFieldManager(options.FieldManager, field.NewPath("fieldManager"))...)
	allErrs = append(allErrs, ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...)
	allErrs = append(allErrs, ValidateFieldValidation(field.NewPath("fieldValidation"), options.FieldValidation)...)
	return allErrs
}

func ValidateUpdateOptions(options *metav1.UpdateOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateFieldManager(options.FieldManager, field.NewPath("fieldManager"))...)
	allErrs = append(allErrs, ValidateDryRun(field.NewPath("dryRun"), options.DryRun)...)
	allErrs = append(allErrs, ValidateFieldValidation(field.NewPath("fieldValidation"), options.FieldValidation)...)
	return allErrs
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
	allErrs = append(allErrs, ValidateFieldValidation(field.NewPath("fieldValidation"), options.FieldValidation)...)
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

var allowedFieldValidationValues = sets.NewString("", metav1.FieldValidationIgnore, metav1.FieldValidationWarn, metav1.FieldValidationStrict)

// ValidateFieldValidation validates that a fieldValidation query param only contains allowed values.
func ValidateFieldValidation(fldPath *field.Path, fieldValidation string) field.ErrorList {
	allErrs := field.ErrorList{}
	if !allowedFieldValidationValues.Has(fieldValidation) {
		allErrs = append(allErrs, field.NotSupported(fldPath, fieldValidation, allowedFieldValidationValues.List()))
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

const MaxSubresourceNameLength = 256

func ValidateManagedFields(fieldsList []metav1.ManagedFieldsEntry, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, fields := range fieldsList {
		fldPath := fldPath.Index(i)
		switch fields.Operation {
		case metav1.ManagedFieldsOperationApply, metav1.ManagedFieldsOperationUpdate:
		default:
			allErrs = append(allErrs, field.Invalid(fldPath.Child("operation"), fields.Operation, "must be `Apply` or `Update`"))
		}
		if len(fields.FieldsType) > 0 && fields.FieldsType != "FieldsV1" {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("fieldsType"), fields.FieldsType, "must be `FieldsV1`"))
		}
		allErrs = append(allErrs, ValidateFieldManager(fields.Manager, fldPath.Child("manager"))...)

		if len(fields.Subresource) > MaxSubresourceNameLength {
			allErrs = append(allErrs, field.TooLong(fldPath.Child("subresource"), fields.Subresource, MaxSubresourceNameLength))
		}
	}
	return allErrs
}

func ValidateConditions(conditions []metav1.Condition, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	conditionTypeToFirstIndex := map[string]int{}
	for i, condition := range conditions {
		if _, ok := conditionTypeToFirstIndex[condition.Type]; ok {
			allErrs = append(allErrs, field.Duplicate(fldPath.Index(i).Child("type"), condition.Type))
		} else {
			conditionTypeToFirstIndex[condition.Type] = i
		}

		allErrs = append(allErrs, ValidateCondition(condition, fldPath.Index(i))...)
	}

	return allErrs
}

// validConditionStatuses is used internally to check validity and provide a good message
var validConditionStatuses = sets.NewString(string(metav1.ConditionTrue), string(metav1.ConditionFalse), string(metav1.ConditionUnknown))

const (
	maxReasonLen  = 1 * 1024
	maxMessageLen = 32 * 1024
)

func ValidateCondition(condition metav1.Condition, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// type is set and is a valid format
	allErrs = append(allErrs, ValidateLabelName(condition.Type, fldPath.Child("type"))...)

	// status is set and is an accepted value
	if !validConditionStatuses.Has(string(condition.Status)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("status"), condition.Status, validConditionStatuses.List()))
	}

	if condition.ObservedGeneration < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("observedGeneration"), condition.ObservedGeneration, "must be greater than or equal to zero"))
	}

	if condition.LastTransitionTime.IsZero() {
		allErrs = append(allErrs, field.Required(fldPath.Child("lastTransitionTime"), "must be set"))
	}

	if len(condition.Reason) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("reason"), "must be set"))
	} else {
		for _, currErr := range isValidConditionReason(condition.Reason) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("reason"), condition.Reason, currErr))
		}
		if len(condition.Reason) > maxReasonLen {
			allErrs = append(allErrs, field.TooLong(fldPath.Child("reason"), condition.Reason, maxReasonLen))
		}
	}

	if len(condition.Message) > maxMessageLen {
		allErrs = append(allErrs, field.TooLong(fldPath.Child("message"), condition.Message, maxMessageLen))
	}

	return allErrs
}

const conditionReasonFmt string = "[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?"
const conditionReasonErrMsg string = "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_'"

var conditionReasonRegexp = regexp.MustCompile("^" + conditionReasonFmt + "$")

// isValidConditionReason tests for a string that conforms to rules for condition reasons. This checks the format, but not the length.
func isValidConditionReason(value string) []string {
	if !conditionReasonRegexp.MatchString(value) {
		return []string{validation.RegexError(conditionReasonErrMsg, conditionReasonFmt, "my_name", "MY_NAME", "MyName", "ReasonA,ReasonB", "ReasonA:ReasonB")}
	}
	return nil
}
