/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

var (
	labelValueErrorMsg    string = fmt.Sprintf(`must have at most %d characters, matching regex %s: e.g. "MyValue" or ""`, validation.LabelValueMaxLength, validation.LabelValueFmt)
	qualifiedNameErrorMsg string = fmt.Sprintf(`must be a qualified name (at most %d characters, matching regex %s), with an optional DNS subdomain prefix (at most %d characters, matching regex %s) and slash (/): e.g. "MyName" or "example.com/MyName"`, validation.QualifiedNameMaxLength, validation.QualifiedNameFmt, validation.DNS1123SubdomainMaxLength, validation.DNS1123SubdomainFmt)
)

func ValidateLabelSelector(ps *unversioned.LabelSelector, fldPath *field.Path) field.ErrorList {
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

func ValidateLabelSelectorRequirement(sr unversioned.LabelSelectorRequirement, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch sr.Operator {
	case unversioned.LabelSelectorOpIn, unversioned.LabelSelectorOpNotIn:
		if len(sr.Values) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified when `operator` is 'In' or 'NotIn'"))
		}
	case unversioned.LabelSelectorOpExists, unversioned.LabelSelectorOpDoesNotExist:
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
	if !validation.IsQualifiedName(labelName) {
		allErrs = append(allErrs, field.Invalid(fldPath, labelName, qualifiedNameErrorMsg))
	}
	return allErrs
}

// ValidateLabels validates that a set of labels are correctly defined.
func ValidateLabels(labels map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for k, v := range labels {
		allErrs = append(allErrs, ValidateLabelName(k, fldPath)...)
		if !validation.IsValidLabelValue(v) {
			allErrs = append(allErrs, field.Invalid(fldPath, v, labelValueErrorMsg))
		}
	}
	return allErrs
}
