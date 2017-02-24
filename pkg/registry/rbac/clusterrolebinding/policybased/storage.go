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

// Package policybased implements a standard storage for ClusterRoleBinding that prevents privilege escalation.
package policybased

import (
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacregistry "k8s.io/kubernetes/pkg/registry/rbac"
	rbacregistryvalidation "k8s.io/kubernetes/pkg/registry/rbac/validation"
)

var groupResource = rbac.Resource("clusterrolebindings")

type Storage struct {
	rest.StandardStorage

	authorizer authorizer.Authorizer

	ruleResolver rbacregistryvalidation.AuthorizationRuleResolver
}

func NewStorage(s rest.StandardStorage, authorizer authorizer.Authorizer, ruleResolver rbacregistryvalidation.AuthorizationRuleResolver) *Storage {
	return &Storage{s, authorizer, ruleResolver}
}

func (s *Storage) Create(ctx genericapirequest.Context, obj runtime.Object) (runtime.Object, error) {
	if rbacregistry.EscalationAllowed(ctx) {
		return s.StandardStorage.Create(ctx, obj)
	}

	clusterRoleBinding := obj.(*rbac.ClusterRoleBinding)
	if rbacregistry.BindingAuthorized(ctx, clusterRoleBinding.RoleRef, clusterRoleBinding.Namespace, s.authorizer) {
		return s.StandardStorage.Create(ctx, obj)
	}

	rules, err := s.ruleResolver.GetRoleReferenceRules(clusterRoleBinding.RoleRef, clusterRoleBinding.Namespace)
	if err != nil {
		return nil, err
	}
	if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
		return nil, errors.NewForbidden(groupResource, clusterRoleBinding.Name, err)
	}
	return s.StandardStorage.Create(ctx, obj)
}

func (s *Storage) Update(ctx genericapirequest.Context, name string, obj rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	if rbacregistry.EscalationAllowed(ctx) {
		return s.StandardStorage.Update(ctx, name, obj)
	}

	nonEscalatingInfo := rest.WrapUpdatedObjectInfo(obj, func(ctx genericapirequest.Context, obj runtime.Object, oldObj runtime.Object) (runtime.Object, error) {
		clusterRoleBinding := obj.(*rbac.ClusterRoleBinding)

		// if we're explicitly authorized to bind this clusterrole, return
		if rbacregistry.BindingAuthorized(ctx, clusterRoleBinding.RoleRef, clusterRoleBinding.Namespace, s.authorizer) {
			return obj, nil
		}

		// Otherwise, see if we already have all the permissions contained in the referenced clusterrole
		rules, err := s.ruleResolver.GetRoleReferenceRules(clusterRoleBinding.RoleRef, clusterRoleBinding.Namespace)
		if err != nil {
			return nil, err
		}
		if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
			return nil, errors.NewForbidden(groupResource, clusterRoleBinding.Name, err)
		}
		return obj, nil
	})

	return s.StandardStorage.Update(ctx, name, nonEscalatingInfo)
}
