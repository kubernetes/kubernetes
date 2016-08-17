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
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// Minimal validation of names for roles and bindings. Identical to the validation for Openshift. See:
// * https://github.com/kubernetes/kubernetes/blob/60db50/pkg/api/validation/name.go
// * https://github.com/openshift/origin/blob/388478/pkg/api/helpers.go
func minimalNameRequirements(name string, prefix bool) []string {
	return validation.IsValidPathSegmentName(name)
}

func ValidateRole(policy *rbac.Role) field.ErrorList {
	return validateRole(policy, true)
}

func ValidateRoleUpdate(policy *rbac.Role, oldRole *rbac.Role) field.ErrorList {
	return validateRoleUpdate(policy, oldRole, true)
}

func ValidateClusterRole(policy *rbac.ClusterRole) field.ErrorList {
	return validateRole(toRole(policy), false)
}

func ValidateClusterRoleUpdate(policy *rbac.ClusterRole, oldRole *rbac.ClusterRole) field.ErrorList {
	return validateRoleUpdate(toRole(policy), toRole(oldRole), false)
}

func validateRole(role *rbac.Role, isNamespaced bool) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&role.ObjectMeta, isNamespaced, minimalNameRequirements, field.NewPath("metadata"))...)

	for i, rule := range role.Rules {
		if err := validatePolicyRule(rule, isNamespaced, field.NewPath("rules").Index(i)); err != nil {
			allErrs = append(allErrs, err...)
		}
	}
	if len(allErrs) != 0 {
		return allErrs
	}
	return nil
}

func validatePolicyRule(rule rbac.PolicyRule, isNamespaced bool, fldPath *field.Path) field.ErrorList {
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

func validateRoleUpdate(role *rbac.Role, oldRole *rbac.Role, isNamespaced bool) field.ErrorList {
	allErrs := validateRole(role, isNamespaced)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&role.ObjectMeta, &oldRole.ObjectMeta, field.NewPath("metadata"))...)

	return allErrs
}

func ValidateRoleBinding(policy *rbac.RoleBinding) field.ErrorList {
	return validateRoleBinding(policy, true)
}

func ValidateRoleBindingUpdate(policy *rbac.RoleBinding, oldRoleBinding *rbac.RoleBinding) field.ErrorList {
	return validateRoleBindingUpdate(policy, oldRoleBinding, true)
}

func ValidateClusterRoleBinding(policy *rbac.ClusterRoleBinding) field.ErrorList {
	return validateRoleBinding(toRoleBinding(policy), false)
}

func ValidateClusterRoleBindingUpdate(policy *rbac.ClusterRoleBinding, oldRoleBinding *rbac.ClusterRoleBinding) field.ErrorList {
	return validateRoleBindingUpdate(toRoleBinding(policy), toRoleBinding(oldRoleBinding), false)
}

func validateRoleBinding(roleBinding *rbac.RoleBinding, isNamespaced bool) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&roleBinding.ObjectMeta, isNamespaced, minimalNameRequirements, field.NewPath("metadata"))...)

	// roleRef namespace is empty when referring to global policy.
	if len(roleBinding.RoleRef.Namespace) > 0 {
		for _, msg := range validation.ValidateNamespaceName(roleBinding.RoleRef.Namespace, false) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef", "namespace"), roleBinding.RoleRef.Namespace, msg))
		}
	}

	if len(roleBinding.RoleRef.Name) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("roleRef", "name"), ""))
	} else {
		for _, msg := range minimalNameRequirements(roleBinding.RoleRef.Name, false) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef", "name"), roleBinding.RoleRef.Name, msg))
		}
	}

	subjectsPath := field.NewPath("subjects")
	for i, subject := range roleBinding.Subjects {
		allErrs = append(allErrs, validateRoleBindingSubject(subject, isNamespaced, subjectsPath.Index(i))...)
	}

	return allErrs
}

func validateRoleBindingSubject(subject rbac.Subject, isNamespaced bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(subject.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}

	switch subject.Kind {
	case rbac.ServiceAccountKind:
		if len(subject.Name) > 0 {
			for _, msg := range validation.ValidateServiceAccountName(subject.Name, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, msg))
			}
		}
		if !isNamespaced && len(subject.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), ""))
		}

	case rbac.UserKind:
		// TODO(ericchiang): What other restrictions on user name are there?
		if len(subject.Name) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, "user name cannot be empty"))
		}

	case rbac.GroupKind:
		// TODO(ericchiang): What other restrictions on group name are there?
		if len(subject.Name) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, "group name cannot be empty"))
		}

	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("kind"), subject.Kind, []string{rbac.ServiceAccountKind, rbac.UserKind, rbac.GroupKind}))
	}

	return allErrs
}

func validateRoleBindingUpdate(roleBinding *rbac.RoleBinding, oldRoleBinding *rbac.RoleBinding, isNamespaced bool) field.ErrorList {
	allErrs := validateRoleBinding(roleBinding, isNamespaced)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&roleBinding.ObjectMeta, &oldRoleBinding.ObjectMeta, field.NewPath("metadata"))...)

	if oldRoleBinding.RoleRef != roleBinding.RoleRef {
		allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef"), roleBinding.RoleRef, "cannot change roleRef"))
	}

	return allErrs
}
