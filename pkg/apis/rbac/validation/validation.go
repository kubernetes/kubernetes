/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
func minimalNameRequirements(name string, prefix bool) (bool, string) {
	return validation.IsValidPathSegmentName(name)
}

func ValidateLocalRole(policy *rbac.Role) field.ErrorList {
	return ValidateRole(policy, true)
}

func ValidateLocalRoleUpdate(policy *rbac.Role, oldRole *rbac.Role) field.ErrorList {
	return ValidateRoleUpdate(policy, oldRole, true)
}

func ValidateClusterRole(policy *rbac.ClusterRole) field.ErrorList {
	return ValidateRole(toRole(policy), false)
}

func ValidateClusterRoleUpdate(policy *rbac.ClusterRole, oldRole *rbac.ClusterRole) field.ErrorList {
	return ValidateRoleUpdate(toRole(policy), toRole(oldRole), false)
}

func ValidateRole(role *rbac.Role, isNamespaced bool) field.ErrorList {
	return validateRole(role, isNamespaced, nil)
}

func validateRole(role *rbac.Role, isNamespaced bool, fldPath *field.Path) field.ErrorList {
	return validation.ValidateObjectMeta(&role.ObjectMeta, isNamespaced, minimalNameRequirements, fldPath.Child("metadata"))
}

func ValidateRoleUpdate(role *rbac.Role, oldRole *rbac.Role, isNamespaced bool) field.ErrorList {
	allErrs := ValidateRole(role, isNamespaced)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&role.ObjectMeta, &oldRole.ObjectMeta, field.NewPath("metadata"))...)

	return allErrs
}

func ValidateLocalRoleBinding(policy *rbac.RoleBinding) field.ErrorList {
	return ValidateRoleBinding(policy, true)
}

func ValidateLocalRoleBindingUpdate(policy *rbac.RoleBinding, oldRoleBinding *rbac.RoleBinding) field.ErrorList {
	return ValidateRoleBindingUpdate(policy, oldRoleBinding, true)
}

func ValidateClusterRoleBinding(policy *rbac.ClusterRoleBinding) field.ErrorList {
	return ValidateRoleBinding(toRoleBinding(policy), false)
}

func ValidateClusterRoleBindingUpdate(policy *rbac.ClusterRoleBinding, oldRoleBinding *rbac.ClusterRoleBinding) field.ErrorList {
	return ValidateRoleBindingUpdate(toRoleBinding(policy), toRoleBinding(oldRoleBinding), false)
}

func ValidateRoleBinding(roleBinding *rbac.RoleBinding, isNamespaced bool) field.ErrorList {
	return validateRoleBinding(roleBinding, isNamespaced, nil)
}

func validateRoleBinding(roleBinding *rbac.RoleBinding, isNamespaced bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateObjectMeta(&roleBinding.ObjectMeta, isNamespaced, minimalNameRequirements, fldPath.Child("metadata"))...)

	// roleRef namespace is empty when referring to global policy.
	if len(roleBinding.RoleRef.Namespace) > 0 {
		if ok, reason := validation.ValidateNamespaceName(roleBinding.RoleRef.Namespace, false); !ok {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("roleRef", "namespace"), roleBinding.RoleRef.Namespace, reason))
		}
	}

	if len(roleBinding.RoleRef.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("roleRef", "name"), ""))
	} else {
		if valid, err := minimalNameRequirements(roleBinding.RoleRef.Name, false); !valid {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("roleRef", "name"), roleBinding.RoleRef.Name, err))
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
	if len(subject.APIVersion) != 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("apiVersion"), subject.APIVersion))
	}

	switch subject.Kind {
	case rbac.ServiceAccountKind:
		if valid, reason := validation.ValidateServiceAccountName(subject.Name, false); len(subject.Name) > 0 && !valid {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, reason))
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

func ValidateRoleBindingUpdate(roleBinding *rbac.RoleBinding, oldRoleBinding *rbac.RoleBinding, isNamespaced bool) field.ErrorList {
	allErrs := ValidateRoleBinding(roleBinding, isNamespaced)
	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&roleBinding.ObjectMeta, &oldRoleBinding.ObjectMeta, field.NewPath("metadata"))...)

	if oldRoleBinding.RoleRef != roleBinding.RoleRef {
		allErrs = append(allErrs, field.Invalid(field.NewPath("roleRef"), roleBinding.RoleRef, "cannot change roleRef"))
	}

	return allErrs
}
