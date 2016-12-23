/*
Copyright 2015 The Kubernetes Authors.

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

package clusterrolebinding

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/runtime"
)

func newClusterRoleBindingControllerFromClient(kubeClient internalclientset.Interface) *ClusterRoleBindingController {
	f := informers.NewSharedInformerFactory(nil, kubeClient, controller.NoResyncPeriodFunc())
	return NewClusterRoleBindingController(f.ClusterRoleBindings(), kubeClient)
}

func TestCreateCondition(t *testing.T) {
	name := "role"
	tests := []struct {
		Name                string
		Role                *rbac.ClusterRole
		Bindings            []*rbac.ClusterRoleBinding
		DeletedBindingNames []string
		OwnerReferences     []metav1.OwnerReference
	}{
		{
			Name: "create bindings",
			Role: &rbac.ClusterRole{
				ObjectMeta: api.ObjectMeta{
					Name: name,
					UID:  "role1",
				},
			},
			Bindings: []*rbac.ClusterRoleBinding{
				{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "rbac.authorization.k8s.io/v1alpha1",
						Kind:       "ClusterRoleBinding",
					},
					ObjectMeta: api.ObjectMeta{
						Name: "binding-1",
						UID:  "binding-1",
					},
					RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "role"},
				},
				{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "rbac.authorization.k8s.io/v1alpha1",
						Kind:       "ClusterRoleBinding",
					},
					ObjectMeta: api.ObjectMeta{
						Name: "binding-2",
						UID:  "binding-2",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "rbac.authorization.k8s.io/v1alpha1",
								Kind:       "ClusterRole",
								Name:       "role2",
								UID:        "role2",
							}},
					},
					RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "role"},
				},
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "rbac.authorization.k8s.io/v1alpha1",
					Kind:       "ClusterRole",
					Name:       name,
					UID:        "role1",
				}},
		},
		{
			Name: "create bindings to clusterrole which not exist",
			Role: &rbac.ClusterRole{
				ObjectMeta: api.ObjectMeta{
					Name: name,
					UID:  "role1",
				},
			},
			Bindings: []*rbac.ClusterRoleBinding{
				{
					TypeMeta: metav1.TypeMeta{
						APIVersion: "rbac.authorization.k8s.io/v1alpha1",
						Kind:       "ClusterRoleBinding",
					},
					ObjectMeta: api.ObjectMeta{
						Name: "binding-3",
						UID:  "binding-3",
					},
					RoleRef: rbac.RoleRef{APIGroup: rbac.GroupName, Kind: "ClusterRole", Name: "role2"},
				},
			},
			OwnerReferences: []metav1.OwnerReference{},
		},
	}

	for _, test := range tests {
		startingObjects := []runtime.Object{}
		startingObjects = append(startingObjects, test.Role)
		for _, binding := range test.Bindings {
			startingObjects = append(startingObjects, binding)
		}

		client := fake.NewSimpleClientset(startingObjects...)
		fakeClusterRoleBindingControl := controller.FakeClusterRoleBindingControl{}
		gcc := newClusterRoleBindingControllerFromClient(client)
		gcc.clusterRoleBindingControl = &fakeClusterRoleBindingControl

		for _, binding := range test.Bindings {
			fakeClusterRoleBindingControl.Patches = [][]byte{}
			key, err := clusterRoleBindingKeyFunc(binding)
			if err != nil {
				t.Errorf("%s unexpected error: %v", test.Name, err)
				continue
			}
			err = gcc.syncBinding(key)
			if err != nil {
				t.Errorf("%s unexpected error: %v", test.Name, err)
				continue
			}

			actuals := []string{}
			for _, patch := range fakeClusterRoleBindingControl.Patches {
				actuals = append(actuals, string(patch[:]))
			}

			expected := []string{}
			for _, owner := range test.OwnerReferences {
				addControllerPatch := fmt.Sprintf(
					`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
					owner.APIVersion, owner.Kind, owner.Name, owner.UID, binding.UID)
				expected = append(expected, addControllerPatch)
			}
			if !reflect.DeepEqual(expected, actuals) {
				t.Errorf("%s: expected %v, got %v", test.Name, expected, actuals)
			}
		}
	}
}
