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

package v1

import (
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientrbacv1 "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// ClusterRoleBindingAdapter is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/kubectl/reconciliation/v1.RoleBinding
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleBindingAdapter struct {
	ClusterRoleBinding *rbacv1.ClusterRoleBinding
}

// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetObject() runtime.Object {
	return o.ClusterRoleBinding
}

// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetNamespace() string {
	return o.ClusterRoleBinding.Namespace
}

// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetName() string {
	return o.ClusterRoleBinding.Name
}

// GetUID is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetUID() types.UID {
	return o.ClusterRoleBinding.UID
}

// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetLabels() map[string]string {
	return o.ClusterRoleBinding.Labels
}

// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) SetLabels(in map[string]string) {
	o.ClusterRoleBinding.Labels = in
}

// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetAnnotations() map[string]string {
	return o.ClusterRoleBinding.Annotations
}

// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) SetAnnotations(in map[string]string) {
	o.ClusterRoleBinding.Annotations = in
}

// GetRoleRef is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetRoleRef() rbacv1.RoleRef {
	return o.ClusterRoleBinding.RoleRef
}

// GetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) GetSubjects() []rbacv1.Subject {
	return o.ClusterRoleBinding.Subjects
}

// SetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o ClusterRoleBindingAdapter) SetSubjects(in []rbacv1.Subject) {
	o.ClusterRoleBinding.Subjects = in
}

// ClusterRoleBindingClientAdapter is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type ClusterRoleBindingClientAdapter struct {
	Client clientrbacv1.ClusterRoleBindingInterface
}

// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Create(in.(ClusterRoleBindingAdapter).ClusterRoleBinding)
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Update(in.(ClusterRoleBindingAdapter).ClusterRoleBinding)
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err

}

// Delete is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (c ClusterRoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.Delete(name, &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
