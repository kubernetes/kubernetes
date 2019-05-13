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
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
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

func (r *Storage) NamespaceScoped() bool {
	return true
}

func (r *Storage) StorageVersion() runtime.GroupVersioner {
	svp, ok := r.StandardStorage.(rest.StorageVersionProvider)
	if !ok {
		return nil
	}
	return svp.StorageVersion()
}

var _ rest.StorageVersionProvider = &Storage{}

func (s *Storage) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	if rbacregistry.EscalationAllowed(ctx) {
		return s.StandardStorage.Create(ctx, obj, createValidation, options)
	}

	// Get the namespace from the context (populated from the URL).
	// The namespace in the object can be empty until StandardStorage.Create()->BeforeCreate() populates it from the context.
	namespace, ok := genericapirequest.NamespaceFrom(ctx)
	if !ok {
		return nil, errors.NewBadRequest("namespace is required")
	}

	roleBinding := obj.(*rbac.RoleBinding)
	if rbacregistry.BindingAuthorized(ctx, roleBinding.RoleRef, namespace, s.authorizer) {
		return s.StandardStorage.Create(ctx, obj, createValidation, options)
	}

	v1RoleRef := rbacv1.RoleRef{}
	err := rbacv1helpers.Convert_rbac_RoleRef_To_v1_RoleRef(&roleBinding.RoleRef, &v1RoleRef, nil)
	if err != nil {
		return nil, err
	}
	rules, err := s.ruleResolver.GetRoleReferenceRules(v1RoleRef, namespace)
	if err != nil {
		return nil, err
	}
	if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
		return nil, errors.NewForbidden(groupResource, roleBinding.Name, err)
	}
	return s.StandardStorage.Create(ctx, obj, createValidation, options)
}

func (s *Storage) Update(ctx context.Context, name string, obj rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	if rbacregistry.EscalationAllowed(ctx) {
		return s.StandardStorage.Update(ctx, name, obj, createValidation, updateValidation, forceAllowCreate, options)
	}

	nonEscalatingInfo := rest.WrapUpdatedObjectInfo(obj, func(ctx context.Context, obj runtime.Object, oldObj runtime.Object) (runtime.Object, error) {
		// Get the namespace from the context (populated from the URL).
		// The namespace in the object can be empty until StandardStorage.Update()->BeforeUpdate() populates it from the context.
		namespace, ok := genericapirequest.NamespaceFrom(ctx)
		if !ok {
			return nil, errors.NewBadRequest("namespace is required")
		}

		roleBinding := obj.(*rbac.RoleBinding)

		// if we're only mutating fields needed for the GC to eventually delete this obj, return
		if rbacregistry.IsOnlyMutatingGCFields(obj, oldObj, kapihelper.Semantic) {
			return obj, nil
		}

		// if we're explicitly authorized to bind this role, return
		if rbacregistry.BindingAuthorized(ctx, roleBinding.RoleRef, namespace, s.authorizer) {
			return obj, nil
		}

		// Otherwise, see if we already have all the permissions contained in the referenced role
		v1RoleRef := rbacv1.RoleRef{}
		err := rbacv1helpers.Convert_rbac_RoleRef_To_v1_RoleRef(&roleBinding.RoleRef, &v1RoleRef, nil)
		if err != nil {
			return nil, err
		}
		rules, err := s.ruleResolver.GetRoleReferenceRules(v1RoleRef, namespace)
		if err != nil {
			return nil, err
		}
		if err := rbacregistryvalidation.ConfirmNoEscalation(ctx, s.ruleResolver, rules); err != nil {
			return nil, errors.NewForbidden(groupResource, roleBinding.Name, err)
		}
		return obj, nil
	})

	return s.StandardStorage.Update(ctx, name, nonEscalatingInfo, createValidation, updateValidation, forceAllowCreate, options)
}
