/*
Copyright 2017 The Kubernetes Authors.

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

package v1beta1

import (
	apiv1 "k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientrbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
)

// RoleBindingAdapter is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/kubectl/reconciliation/v1beta1.RoleBinding
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type RoleBindingAdapter struct {
	RoleBinding *rbacv1beta1.RoleBinding
}

// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetObject() runtime.Object {
	return o.RoleBinding
}

// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetNamespace() string {
	return o.RoleBinding.Namespace
}

// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetName() string {
	return o.RoleBinding.Name
}

// GetUID is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetUID() types.UID {
	return o.RoleBinding.UID
}

// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetLabels() map[string]string {
	return o.RoleBinding.Labels
}

// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) SetLabels(in map[string]string) {
	o.RoleBinding.Labels = in
}

// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetAnnotations() map[string]string {
	return o.RoleBinding.Annotations
}

// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) SetAnnotations(in map[string]string) {
	o.RoleBinding.Annotations = in
}

// GetRoleRef is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetRoleRef() rbacv1beta1.RoleRef {
	return o.RoleBinding.RoleRef
}

// GetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) GetSubjects() []rbacv1beta1.Subject {
	return o.RoleBinding.Subjects
}

// SetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o RoleBindingAdapter) SetSubjects(in []rbacv1beta1.Subject) {
	o.RoleBinding.Subjects = in
}

// RoleBindingClientAdapter is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RoleBindingClientAdapter struct {
	Client          clientrbacv1beta1.RoleBindingsGetter
	NamespaceClient core.NamespaceInterface
}

// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	ns := &apiv1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: in.GetNamespace()}}
	if _, err := c.NamespaceClient.Create(ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return nil, err
	}

	ret, err := c.Client.RoleBindings(in.GetNamespace()).Create(in.(RoleBindingAdapter).RoleBinding)
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(in.GetNamespace()).Update(in.(RoleBindingAdapter).RoleBinding)
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err

}

// Delete is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c RoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.RoleBindings(namespace).Delete(name, &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
