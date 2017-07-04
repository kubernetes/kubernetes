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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	core "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
)

type RoleBindingAdapter struct {
	RoleBinding *rbac.RoleBinding
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

func (o RoleBindingAdapter) GetRoleRef() rbac.RoleRef {
	return o.RoleBinding.RoleRef
}

func (o RoleBindingAdapter) GetSubjects() []rbac.Subject {
	return o.RoleBinding.Subjects
}

func (o RoleBindingAdapter) SetSubjects(in []rbac.Subject) {
	o.RoleBinding.Subjects = in
}

type RoleBindingClientAdapter struct {
	Client          internalversion.RoleBindingsGetter
	NamespaceClient core.NamespaceInterface
}

func (c RoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

func (c RoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	ns := &api.Namespace{ObjectMeta: metav1.ObjectMeta{Name: in.GetNamespace()}}
	if _, err := c.NamespaceClient.Create(ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return nil, err
	}

	ret, err := c.Client.RoleBindings(in.GetNamespace()).Create(in.(RoleBindingAdapter).RoleBinding)
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err
}

func (c RoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.RoleBindings(in.GetNamespace()).Update(in.(RoleBindingAdapter).RoleBinding)
	if err != nil {
		return nil, err
	}
	return RoleBindingAdapter{RoleBinding: ret}, err

}

func (c RoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.RoleBindings(namespace).Delete(name, &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
