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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
)

// +k8s:deepcopy-gen=true
// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/registry/rbac/reconciliation.RoleBinding
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type ClusterRoleBindingAdapter struct {
	ClusterRoleBinding *rbac.ClusterRoleBinding
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

func (o ClusterRoleBindingAdapter) GetRoleRef() rbac.RoleRef {
	return o.ClusterRoleBinding.RoleRef
}

func (o ClusterRoleBindingAdapter) GetSubjects() []rbac.Subject {
	return o.ClusterRoleBinding.Subjects
}

func (o ClusterRoleBindingAdapter) SetSubjects(in []rbac.Subject) {
	o.ClusterRoleBinding.Subjects = in
}

type ClusterRoleBindingClientAdapter struct {
	Client internalversion.ClusterRoleBindingInterface
}

func (c ClusterRoleBindingClientAdapter) Get(namespace, name string) (RoleBinding, error) {
	ret, err := c.Client.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

func (c ClusterRoleBindingClientAdapter) Create(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Create(in.(ClusterRoleBindingAdapter).ClusterRoleBinding)
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err
}

func (c ClusterRoleBindingClientAdapter) Update(in RoleBinding) (RoleBinding, error) {
	ret, err := c.Client.Update(in.(ClusterRoleBindingAdapter).ClusterRoleBinding)
	if err != nil {
		return nil, err
	}
	return ClusterRoleBindingAdapter{ClusterRoleBinding: ret}, err

}

func (c ClusterRoleBindingClientAdapter) Delete(namespace, name string, uid types.UID) error {
	return c.Client.Delete(name, &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}})
}
