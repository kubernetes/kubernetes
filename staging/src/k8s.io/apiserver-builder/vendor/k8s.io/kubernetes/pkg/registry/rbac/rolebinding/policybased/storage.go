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

// Package policybased implements a standard storage for RoleBinding that prevents privilege escalation.
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

var groupResource = rbac.Resource("rolebindings")

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

	// Get the namespace from the context (populated from the URL).
	// The namespace in the object can be empty until StandardStorage.Create()->BeforeCreate() populates it from the context.
	namespace, ok := genericapirequest.NamespaceFrom(ctx)
	if !ok {
		return nil, errors.NewBadRequest("namespace is required")
	}

	roleBinding := obj.(*rbac.RoleBinding)
	if rbacregistry.BindingAuthorized(ctx, roleBinding.RoleRef, namespace, s.authorizer) {
		return s.StandardStorage.Create(ctx, obj)
	}

	rules, err := s.ruleResolver.GetRoleReferenceRules(roleBinding.RoleRef, namespace)
	if err != nil {
		return nil, err
	}
	if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
		return nil, errors.NewForbidden(groupResource, roleBinding.Name, err)
	}
	return s.StandardStorage.Create(ctx, obj)
}

func (s *Storage) Update(ctx genericapirequest.Context, name string, obj rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	if rbacregistry.EscalationAllowed(ctx) {
		return s.StandardStorage.Update(ctx, name, obj)
	}

	nonEscalatingInfo := rest.WrapUpdatedObjectInfo(obj, func(ctx genericapirequest.Context, obj runtime.Object, oldObj runtime.Object) (runtime.Object, error) {
		// Get the namespace from the context (populated from the URL).
		// The namespace in the object can be empty until StandardStorage.Update()->BeforeUpdate() populates it from the context.
		namespace, ok := genericapirequest.NamespaceFrom(ctx)
		if !ok {
			return nil, errors.NewBadRequest("namespace is required")
		}

		roleBinding := obj.(*rbac.RoleBinding)

		// if we're explicitly authorized to bind this role, return
		if rbacregistry.BindingAuthorized(ctx, roleBinding.RoleRef, namespace, s.authorizer) {
			return obj, nil
		}

		// Otherwise, see if we already have all the permissions contained in the referenced role
		rules, err := s.ruleResolver.GetRoleReferenceRules(roleBinding.RoleRef, namespace)
		if err != nil {
			return nil, err
		}
		if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
			return nil, errors.NewForbidden(groupResource, roleBinding.Name, err)
		}
		return obj, nil
	})

	return s.StandardStorage.Update(ctx, name, nonEscalatingInfo)
}
