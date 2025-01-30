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

package reconciliation

import (
	"context"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/component-helpers/auth/rbac/reconciliation.RoleBinding
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleBindingAdapter struct {
	ClusterRoleBinding *rbacv1.ClusterRoleBinding
}

func (o ClusterRoleBindingAdapter) GetObject() runtime.Object {
	return o.ClusterRoleBinding
}

func (o ClusterRoleBindingAdapter) GetNamespace() string {
	return o.ClusterRoleBinding.Namespace
}

func (o ClusterRoleBindingAdapter) GetName() string {
	return o.ClusterRoleBinding.Name
}

func (o ClusterRoleBindingAdapter) GetUID() types.UID {
	return o.ClusterRoleBinding.UID
}

func (o ClusterRoleBindingAdapter) GetLabels() map[string]string {
	return o.ClusterRoleBinding.Labels
}

func (o ClusterRoleBindingAdapter) SetLabels(in map[string]string) {
	o.ClusterRoleBinding.Labels = in
}

func (o ClusterRoleBindingAdapter) GetAnnotations() map[string]string {
	return o.ClusterRoleBinding.Annotations
}

func (o ClusterRoleBindingAdapter) SetAnnotations(in map[string]string) {
	o.ClusterRoleBinding.Annotations = in
}

func (o ClusterRoleBindingAdapter) GetRoleRef() rbacv1.RoleRef {
	return o.ClusterRoleBinding.RoleRef
}

func (o ClusterRoleBindingAdapter) GetSubjects() []rbacv1.Subject {
	return o.ClusterRoleBinding.Subjects
}

func (o ClusterRoleBindingAdapter) SetSubjects(in []rbacv1.Subject) {
	o.ClusterRoleBinding.Subjects = in
}

type ClusterRoleBindingClientAdapter struct {
	Client rbacv1client.ClusterRoleBindingInterface
}

func (c ClusterRoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

func (c ClusterRoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Create(context.TODO(), in.(ClusterRoleBindingAdapter).ClusterRoleBinding, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

func (c ClusterRoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Update(context.TODO(), in.(ClusterRoleBindingAdapter).ClusterRoleBinding, metav1.UpdateOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err

}

func (c ClusterRoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.Delete(context.TODO(), name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
