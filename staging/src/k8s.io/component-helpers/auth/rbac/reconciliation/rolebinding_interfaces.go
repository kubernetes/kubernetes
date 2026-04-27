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
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	rbacv1client "k8s.io/client-go/kubernetes/typed/rbac/v1"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/component-helpers/auth/rbac/reconciliation.RoleBinding
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type RoleBindingAdapter struct {
	RoleBinding *rbacv1.RoleBinding
}

func (o RoleBindingAdapter) GetObject() runtime.Object {
	return o.RoleBinding
}

func (o RoleBindingAdapter) GetNamespace() string {
	return o.RoleBinding.Namespace
}

func (o RoleBindingAdapter) GetName() string {
	return o.RoleBinding.Name
}

func (o RoleBindingAdapter) GetUID() types.UID {
	return o.RoleBinding.UID
}

func (o RoleBindingAdapter) GetLabels() map[string]string {
	return o.RoleBinding.Labels
}

func (o RoleBindingAdapter) SetLabels(in map[string]string) {
	o.RoleBinding.Labels = in
}

func (o RoleBindingAdapter) GetAnnotations() map[string]string {
	return o.RoleBinding.Annotations
}

func (o RoleBindingAdapter) SetAnnotations(in map[string]string) {
	o.RoleBinding.Annotations = in
}

func (o RoleBindingAdapter) GetRoleRef() rbacv1.RoleRef {
	return o.RoleBinding.RoleRef
}

func (o RoleBindingAdapter) GetSubjects() []rbacv1.Subject {
	return o.RoleBinding.Subjects
}

func (o RoleBindingAdapter) SetSubjects(in []rbacv1.Subject) {
	o.RoleBinding.Subjects = in
}

type RoleBindingClientAdapter struct {
	Client          rbacv1client.RoleBindingsGetter
	NamespaceClient corev1client.NamespaceInterface
}

func (c RoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

func (c RoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	if err := tryEnsureNamespace(c.NamespaceClient, in.GetNamespace()); err != nil {
		return nil, err
	}

	ret, err := c.Client.RoleBindings(in.GetNamespace()).Create(context.TODO(), in.(RoleBindingAdapter).RoleBinding, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

func (c RoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(in.GetNamespace()).Update(context.TODO(), in.(RoleBindingAdapter).RoleBinding, metav1.UpdateOptions{})
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err

}

func (c RoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.RoleBindings(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
