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
	"regexp"

	policyapiv1beta1 "k8s.io/api/policy/v1beta1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	appsvalidation "k8s.io/kubernetes/pkg/apis/apps/validation"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/policy"
)

var supportedUnhealthyPodEvictionPolicies = sets.NewString(
	string(policy.IfHealthyBudget),
	string(policy.AlwaysAllow),
)

type PodDisruptionBudgetValidationOptions struct {
	AllowInvalidLabelValueInSelector bool
}

// ValidatePodDisruptionBudget validates a PodDisruptionBudget and returns an ErrorList
// with any errors.
func ValidatePodDisruptionBudget(pdb *policy.PodDisruptionBudget, opts PodDisruptionBudgetValidationOptions) field.ErrorList {
	allErrs := ValidatePodDisruptionBudgetSpec(pdb.Spec, opts, field.NewPath("spec"))
	return allErrs
}

// ValidatePodDisruptionBudgetSpec validates a PodDisruptionBudgetSpec and returns an ErrorList
// with any errors.
func ValidatePodDisruptionBudgetSpec(spec policy.PodDisruptionBudgetSpec, opts PodDisruptionBudgetValidationOptions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.MinAvailable != nil && spec.MaxUnavailable != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, spec, "minAvailable and maxUnavailable cannot be both set"))
	}

	if spec.MinAvailable != nil {
		allErrs = append(allErrs, appsvalidation.ValidatePositiveIntOrPercent(*spec.MinAvailable, fldPath.Child("minAvailable"))...)
		allErrs = append(allErrs, appsvalidation.IsNotMoreThan100Percent(*spec.MinAvailable, fldPath.Child("minAvailable"))...)
	}

	if spec.MaxUnavailable != nil {
		allErrs = append(allErrs, appsvalidation.ValidatePositiveIntOrPercent(*spec.MaxUnavailable, fldPath.Child("maxUnavailable"))...)
		allErrs = append(allErrs, appsvalidation.IsNotMoreThan100Percent(*spec.MaxUnavailable, fldPath.Child("maxUnavailable"))...)
	}

	labelSelectorValidationOptions := unversionedvalidation.LabelSelectorValidationOptions{AllowInvalidLabelValueInSelector: opts.AllowInvalidLabelValueInSelector}

	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, labelSelectorValidationOptions, fldPath.Child("selector"))...)

	if spec.UnhealthyPodEvictionPolicy != nil && !supportedUnhealthyPodEvictionPolicies.Has(string(*spec.UnhealthyPodEvictionPolicy)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("unhealthyPodEvictionPolicy"), *spec.UnhealthyPodEvictionPolicy, supportedUnhealthyPodEvictionPolicies.List()))
	}

	return allErrs
}

// ValidatePodDisruptionBudgetStatusUpdate validates a PodDisruptionBudgetStatus and returns an ErrorList
// with any errors.
func ValidatePodDisruptionBudgetStatusUpdate(status, oldStatus policy.PodDisruptionBudgetStatus, fldPath *field.Path, apiVersion schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, unversionedvalidation.ValidateConditions(status.Conditions, fldPath.Child("conditions"))...)
	// Don't run other validations for v1beta1 since we don't want to introduce
	// new validations retroactively.
	if apiVersion == policyapiv1beta1.SchemeGroupVersion {
		return allErrs
	}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.DisruptionsAllowed), fldPath.Child("disruptionsAllowed"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.CurrentHealthy), fldPath.Child("currentHealthy"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.DesiredHealthy), fldPath.Child("desiredHealthy"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.ExpectedPods), fldPath.Child("expectedPods"))...)
	return allErrs
}

const sysctlPatternSegmentFmt string = "([a-z0-9][-_a-z0-9]*)?[a-z0-9*]"

// SysctlContainSlashPatternFmt is a regex that contains a slash used for matching valid sysctl patterns.
const SysctlContainSlashPatternFmt string = "(" + apivalidation.SysctlSegmentFmt + "[\\./])*" + sysctlPatternSegmentFmt

var sysctlContainSlashPatternRegexp = regexp.MustCompile("^" + SysctlContainSlashPatternFmt + "$")

// IsValidSysctlPattern checks if name is a valid sysctl pattern.
// i.e. matches sysctlContainSlashPatternRegexp.
// More info:
//
//	https://man7.org/linux/man-pages/man8/sysctl.8.html
//	https://man7.org/linux/man-pages/man5/sysctl.d.5.html
func IsValidSysctlPattern(name string) bool {
	if len(name) > apivalidation.SysctlMaxLength {
		return false
	}
	return sysctlContainSlashPatternRegexp.MatchString(name)
}
