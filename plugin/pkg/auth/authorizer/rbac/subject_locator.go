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

// Package rbac implements the authorizer.Authorizer interface using roles base access control.
package rbac

import (
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

type RoleToRuleMapper interface {
	// GetRoleReferenceRules attempts to resolve the role reference of a RoleBinding or ClusterRoleBinding.  The passed namespace should be the namepsace
	// of the role binding, the empty string if a cluster role binding.
	GetRoleReferenceRules(roleRef rbac.RoleRef, namespace string) ([]rbac.PolicyRule, error)
}

type SubjectAccessEvaluator struct {
	superUser string

	roleBindingLister        validation.RoleBindingLister
	clusterRoleBindingLister validation.ClusterRoleBindingLister
	roleToRuleMapper         RoleToRuleMapper
}

func NewSubjectAccessEvaluator(roles validation.RoleGetter, roleBindings validation.RoleBindingLister, clusterRoles validation.ClusterRoleGetter, clusterRoleBindings validation.ClusterRoleBindingLister, superUser string) *SubjectAccessEvaluator {
	subjectLocator := &SubjectAccessEvaluator{
		superUser:                superUser,
		roleBindingLister:        roleBindings,
		clusterRoleBindingLister: clusterRoleBindings,
		roleToRuleMapper: validation.NewDefaultRuleResolver(
			roles, roleBindings, clusterRoles, clusterRoleBindings,
		),
	}
	return subjectLocator
}

func (r *SubjectAccessEvaluator) AllowedSubjects(requestAttributes authorizer.Attributes) ([]rbac.Subject, error) {
	subjects := []rbac.Subject{{Kind: rbac.GroupKind, Name: user.SystemPrivilegedGroup}}
	if len(r.superUser) > 0 {
		subjects = append(subjects, rbac.Subject{Kind: rbac.UserKind, Name: r.superUser})
	}
	errorlist := []error{}

	if clusterRoleBindings, err := r.clusterRoleBindingLister.ListClusterRoleBindings(); err != nil {
		errorlist = append(errorlist, err)

	} else {
		for _, clusterRoleBinding := range clusterRoleBindings {
			rules, err := r.roleToRuleMapper.GetRoleReferenceRules(clusterRoleBinding.RoleRef, "")
			if err != nil {
				errorlist = append(errorlist, err)
			}
			if RulesAllow(requestAttributes, rules...) {
				subjects = append(subjects, clusterRoleBinding.Subjects...)
			}
		}
	}

	if namespace := requestAttributes.GetNamespace(); len(namespace) > 0 {
		if roleBindings, err := r.roleBindingLister.ListRoleBindings(namespace); err != nil {
			errorlist = append(errorlist, err)

		} else {
			for _, roleBinding := range roleBindings {
				rules, err := r.roleToRuleMapper.GetRoleReferenceRules(roleBinding.RoleRef, namespace)
				if err != nil {
					errorlist = append(errorlist, err)
				}
				if RulesAllow(requestAttributes, rules...) {
					subjects = append(subjects, roleBinding.Subjects...)
				}
			}
		}
	}

	dedupedSubjects := []rbac.Subject{}
	for _, subject := range subjects {
		found := false
		for _, curr := range dedupedSubjects {
			if curr == subject {
				found = true
				break
			}
		}

		if !found {
			dedupedSubjects = append(dedupedSubjects, subject)
		}
	}

	return subjects, utilerrors.NewAggregate(errorlist)
}
