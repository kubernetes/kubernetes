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
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
)

type RoleToRuleMapper interface {
	// GetRoleReferenceRules attempts to resolve the role reference of a RoleBinding or ClusterRoleBinding.  The passed namespace should be the namespace
	// of the role binding, the empty string if a cluster role binding.
	GetRoleReferenceRules(ctx context.Context, roleRef rbacv1.RoleRef, namespace string) ([]rbacv1.PolicyRule, error)
}

type SubjectLocator interface {
	AllowedSubjects(ctx context.Context, attributes authorizer.Attributes) ([]rbacv1.Subject, error)
}

var _ = SubjectLocator(&SubjectAccessEvaluator{})

type SubjectAccessEvaluator struct {
	superUser string

	roleBindingLister        rbacregistryvalidation.RoleBindingLister
	clusterRoleBindingLister rbacregistryvalidation.ClusterRoleBindingLister
	roleToRuleMapper         RoleToRuleMapper
}

func NewSubjectAccessEvaluator(roles rbacregistryvalidation.RoleGetter, roleBindings rbacregistryvalidation.RoleBindingLister, clusterRoles rbacregistryvalidation.ClusterRoleGetter, clusterRoleBindings rbacregistryvalidation.ClusterRoleBindingLister, superUser string) *SubjectAccessEvaluator {
	subjectLocator := &SubjectAccessEvaluator{
		superUser:                superUser,
		roleBindingLister:        roleBindings,
		clusterRoleBindingLister: clusterRoleBindings,
		roleToRuleMapper: rbacregistryvalidation.NewDefaultRuleResolver(
			roles, roleBindings, clusterRoles, clusterRoleBindings,
		),
	}
	return subjectLocator
}

// AllowedSubjects returns the subjects that can perform an action and any errors encountered while computing the list.
// It is possible to have both subjects and errors returned if some rolebindings couldn't be resolved, but others could be.
func (r *SubjectAccessEvaluator) AllowedSubjects(ctx context.Context, requestAttributes authorizer.Attributes) ([]rbacv1.Subject, error) {
	subjects := []rbacv1.Subject{{Kind: rbacv1.GroupKind, APIGroup: rbacv1.GroupName, Name: user.SystemPrivilegedGroup}}
	if len(r.superUser) > 0 {
		subjects = append(subjects, rbacv1.Subject{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: r.superUser})
	}
	errorlist := []error{}

	if clusterRoleBindings, err := r.clusterRoleBindingLister.ListClusterRoleBindings(ctx); err != nil {
		errorlist = append(errorlist, err)

	} else {
		for _, clusterRoleBinding := range clusterRoleBindings {
			rules, err := r.roleToRuleMapper.GetRoleReferenceRules(ctx, clusterRoleBinding.RoleRef, "")
			if err != nil {
				// if we have an error, just keep track of it and keep processing.  Since rules are additive,
				// missing a reference is bad, but we can continue with other rolebindings and still have a list
				// that does not contain any invalid values
				errorlist = append(errorlist, err)
			}
			if RulesAllow(requestAttributes, rules...) {
				subjects = append(subjects, clusterRoleBinding.Subjects...)
			}
		}
	}

	if namespace := requestAttributes.GetNamespace(); len(namespace) > 0 {
		if roleBindings, err := r.roleBindingLister.ListRoleBindings(ctx, namespace); err != nil {
			errorlist = append(errorlist, err)

		} else {
			for _, roleBinding := range roleBindings {
				rules, err := r.roleToRuleMapper.GetRoleReferenceRules(ctx, roleBinding.RoleRef, namespace)
				if err != nil {
					// if we have an error, just keep track of it and keep processing.  Since rules are additive,
					// missing a reference is bad, but we can continue with other rolebindings and still have a list
					// that does not contain any invalid values
					errorlist = append(errorlist, err)
				}
				if RulesAllow(requestAttributes, rules...) {
					subjects = append(subjects, roleBinding.Subjects...)
				}
			}
		}
	}

	dedupedSubjects := []rbacv1.Subject{}
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
