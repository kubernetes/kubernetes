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
	"k8s.io/apimachinery/pkg/api/validation/path"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// ValidateRBACName is exported to allow types outside of the RBAC API group to reuse this validation logic
// Minimal validation of names for roles and bindings. Identical to the validation for Openshift. See:
// * https://github.com/kubernetes/kubernetes/blob/60db507b279ce45bd16ea3db49bf181f2aeb3c3d/pkg/api/validation/name.go
// * https://github.com/openshift/origin/blob/388478c40e751c4295dcb9a44dd69e5ac65d0e3b/pkg/api/helpers.go
func ValidateRBACName(name string, prefix bool) []string {
	return path.IsValidPathSegmentName(name)
}

func ValidateRole(role *rbac.Role) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&role.ObjectMeta, true, ValidateRBACName, field.NewPath("metadata"))...)

	for i, rule := range role.Rules {
		if err := ValidatePolicyRule(rule, true, field.NewPath("rules").Index(i)); err != nil {
			allErrs = append(allErrs, err...)
		}
	}
	if len(allErrs) != 0 {
		return allErrs
	}
	return nil
}

func ValidateRoleUpdate(role *rbac.Role, oldRole *rbac.Role) field.ErrorList {
	allErrs := ValidateRole(role)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&role.ObjectMeta, &oldRole.ObjectMeta, field.NewPath("metadata"))...)

	return allErrs
}

type ClusterRoleValidationOptions struct {
	AllowInvalidLabelValueInSelector bool
}

func ValidateClusterRole(role *rbac.ClusterRole, opts ClusterRoleValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&role.ObjectMeta, false, ValidateRBACName, field.NewPath("metadata"))...)

	for i, rule := range role.Rules {
		if err := ValidatePolicyRule(rule, false, field.NewPath("rules").Index(i)); err != nil {
			allErrs = append(allErrs, err...)
		}
	}

	labelSelectorValidationOptions := unversionedvalidation.LabelSelectorValidationOptions{AllowInvalidLabelValueInSelector: opts.AllowInvalidLabelValueInSelector}

	if role.AggregationRule != nil {
		if len(role.AggregationRule.ClusterRoleSelectors) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("aggregationRule", "clusterRoleSelectors"), "at least one clusterRoleSelector required if aggregationRule is non-nil"))
		}
		for i, selector := range role.AggregationRule.ClusterRoleSelectors {
			fieldPath := field.NewPath("aggregationRule", "clusterRoleSelectors").Index(i)
			allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(&selector, labelSelectorValidationOptions, fieldPath)...)

			selector, err := metav1.LabelSelectorAsSelector(&selector)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(fieldPath, selector, "invalid label selector."))
			}
		}
	}

	if len(allErrs) != 0 {
		return allErrs
	}
	return nil
}

func ValidateClusterRoleUpdate(role *rbac.ClusterRole, oldRole *rbac.ClusterRole, opts ClusterRoleValidationOptions) field.ErrorList {
	allErrs := ValidateClusterRole(role, opts)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&role.ObjectMeta, &oldRole.ObjectMeta, field.NewPath("metadata"))...)

	return allErrs
}

// ValidatePolicyRule is exported to allow types outside of the RBAC API group to embed a rbac.PolicyRule and reuse this validation logic
func ValidatePolicyRule(rule rbac.PolicyRule, isNamespaced bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(rule.Verbs) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("verbs"), "verbs must contain at least one value"))
	}

	if len(rule.NonResourceURLs) > 0 {
		if isNamespaced {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceURLs"), rule.NonResourceURLs, "namespaced rules cannot apply to non-resource URLs"))
		}
		if len(rule.APIGroups) > 0 || len(rule.Resources) > 0 || len(rule.ResourceNames) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceURLs"), rule.NonResourceURLs, "rules cannot apply to both regular resources and non-resource URLs"))
		}
		return allErrs
	}

	if len(rule.APIGroups) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("apiGroups"), "resource rules must supply at least one api group"))
	}
	if len(rule.Resources) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("resources"), "resource rules must supply at least one resource"))
	}
	return allErrs
}

func ValidateRoleBinding(roleBinding *rbac.RoleBinding) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&roleBinding.ObjectMeta, true, ValidateRBACName, field.NewPath("metadata"))...)

	// TODO allow multiple API groups.  For now, restrict to one, but I can envision other experimental roles in other groups taking
	// advantage of the binding infrastructure
	if roleBinding.RoleRef.APIGroup != rbac.GroupName {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("roleRef", "apiGroup"), roleBinding.RoleRef.APIGroup, []string{rbac.GroupName}))
	}

	switch roleBinding.RoleRef.Kind {
	case "Role", "ClusterRole":
	default:
		allErrs = append(allErrs, field.NotSupported(field.NewPath("roleRef", "kind"), roleBinding.RoleRef.Kind, []string{"Role", "ClusterRole"}))

	}

	if len(roleBinding.RoleRef.Name) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("roleRef", "name"), "").MarkCoveredByDeclarative())
	} else {
		for _, msg := range ValidateRBACName(roleBinding.RoleRef.Name, false) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef", "name"), roleBinding.RoleRef.Name, msg))
		}
	}

	subjectsPath := field.NewPath("subjects")
	for i, subject := range roleBinding.Subjects {
		allErrs = append(allErrs, ValidateRoleBindingSubject(subject, true, subjectsPath.Index(i))...)
	}

	return allErrs
}

func ValidateRoleBindingUpdate(roleBinding *rbac.RoleBinding, oldRoleBinding *rbac.RoleBinding) field.ErrorList {
	allErrs := ValidateRoleBinding(roleBinding)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&roleBinding.ObjectMeta, &oldRoleBinding.ObjectMeta, field.NewPath("metadata"))...)

	if oldRoleBinding.RoleRef != roleBinding.RoleRef {
		allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef"), roleBinding.RoleRef, "cannot change roleRef"))
	}

	return allErrs
}

func ValidateClusterRoleBinding(roleBinding *rbac.ClusterRoleBinding) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&roleBinding.ObjectMeta, false, ValidateRBACName, field.NewPath("metadata"))...)

	// TODO allow multiple API groups.  For now, restrict to one, but I can envision other experimental roles in other groups taking
	// advantage of the binding infrastructure
	if roleBinding.RoleRef.APIGroup != rbac.GroupName {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("roleRef", "apiGroup"), roleBinding.RoleRef.APIGroup, []string{rbac.GroupName}))
	}

	switch roleBinding.RoleRef.Kind {
	case "ClusterRole":
	default:
		allErrs = append(allErrs, field.NotSupported(field.NewPath("roleRef", "kind"), roleBinding.RoleRef.Kind, []string{"ClusterRole"}))

	}

	if len(roleBinding.RoleRef.Name) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("roleRef", "name"), "").MarkCoveredByDeclarative())
	} else {
		for _, msg := range ValidateRBACName(roleBinding.RoleRef.Name, false) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef", "name"), roleBinding.RoleRef.Name, msg))
		}
	}

	subjectsPath := field.NewPath("subjects")
	for i, subject := range roleBinding.Subjects {
		allErrs = append(allErrs, ValidateRoleBindingSubject(subject, false, subjectsPath.Index(i))...)
	}

	return allErrs
}

func ValidateClusterRoleBindingUpdate(roleBinding *rbac.ClusterRoleBinding, oldRoleBinding *rbac.ClusterRoleBinding) field.ErrorList {
	allErrs := ValidateClusterRoleBinding(roleBinding)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&roleBinding.ObjectMeta, &oldRoleBinding.ObjectMeta, field.NewPath("metadata"))...)

	if oldRoleBinding.RoleRef != roleBinding.RoleRef {
		allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef"), roleBinding.RoleRef, "cannot change roleRef"))
	}

	return allErrs
}

// ValidateRoleBindingSubject is exported to allow types outside of the RBAC API group to embed a rbac.Subject and reuse this validation logic
func ValidateRoleBindingSubject(subject rbac.Subject, isNamespaced bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(subject.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "").MarkCoveredByDeclarative())
	}

	switch subject.Kind {
	case rbac.ServiceAccountKind:
		if len(subject.Name) > 0 {
			for _, msg := range validation.ValidateServiceAccountName(subject.Name, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, msg))
			}
		}
		if len(subject.APIGroup) > 0 {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{""}))
		}
		if !isNamespaced && len(subject.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), ""))
		}

	case rbac.UserKind:
		// TODO(ericchiang): What other restrictions on user name are there?
		if subject.APIGroup != rbac.GroupName {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{rbac.GroupName}))
		}

	case rbac.GroupKind:
		// TODO(ericchiang): What other restrictions on group name are there?
		if subject.APIGroup != rbac.GroupName {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{rbac.GroupName}))
		}

	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("kind"), subject.Kind, []string{rbac.ServiceAccountKind, rbac.UserKind, rbac.GroupKind}))
	}

	return allErrs
}
