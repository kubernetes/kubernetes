/*
Copyright 2023 The Kubernetes Authors.

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

package systemnamespaces

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// Test_Controller validates the garbage collection logic for the apiserverleasegc controller.
func Test_Controller(t *testing.T) {
	tests := []struct {
		name       string
		namespaces []string
		actions    [][]string // verb and resource
	}{
		{
			name: "no system namespaces",
			actions: [][]string{
				{"create", "namespaces"},
				{"create", "namespaces"},
				{"create", "namespaces"},
				{"create", "namespaces"},
			},
		},
		{
			name:       "no system namespaces but others",
			namespaces: []string{"foo", "bar"},
			actions: [][]string{
				{"create", "namespaces"},
				{"create", "namespaces"},
				{"create", "namespaces"},
				{"create", "namespaces"},
			},
		},
		{
			name:       "one system namespace",
			namespaces: []string{metav1.NamespaceSystem},
			actions: [][]string{
				{"create", "namespaces"},
				{"create", "namespaces"},
				{"create", "namespaces"},
			},
		},
		{
			name:       "two system namespaces",
			namespaces: []string{metav1.NamespaceSystem, metav1.NamespacePublic},
			actions: [][]string{
				{"create", "namespaces"},
				{"create", "namespaces"},
			},
		},
		{
			name:       "three namespaces",
			namespaces: []string{metav1.NamespaceSystem, metav1.NamespacePublic, v1.NamespaceNodeLease},
			actions: [][]string{
				{"create", "namespaces"},
			},
		},

		{
			name:       "the four namespaces",
			namespaces: []string{metav1.NamespaceSystem, metav1.NamespacePublic, v1.NamespaceNodeLease, v1.NamespaceDefault},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			objs := []runtime.Object{}
			for _, ns := range test.namespaces {
				objs = append(objs,
					&v1.Namespace{
						ObjectMeta: metav1.ObjectMeta{
							Name:      ns,
							Namespace: "",
						},
					},
				)
			}
			clientset := fake.NewSimpleClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(clientset, 0)
			namespaceInformer := informerFactory.Core().V1().Namespaces()
			for _, obj := range objs {
				namespaceInformer.Informer().GetIndexer().Add(obj)
			}

			systemNamespaces := []string{metav1.NamespaceSystem, metav1.NamespacePublic, v1.NamespaceNodeLease, metav1.NamespaceDefault}
			controller := NewController(systemNamespaces, clientset, namespaceInformer)

			clientset.PrependReactor("create", "namespaces", func(action k8stesting.Action) (bool, runtime.Object, error) {
				create := action.(k8stesting.CreateAction)
				namespaceInformer.Informer().GetIndexer().Add(create.GetObject())
				return true, create.GetObject(), nil
			})

			controller.sync()

			expectAction(t, clientset.Actions(), test.actions)
			namespaces, err := controller.namespaceLister.List(labels.Everything())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			got := map[string]bool{}
			for _, ns := range namespaces {
				got[ns.Name] = true
			}

			for _, ns := range systemNamespaces {
				if !got[ns] {
					t.Errorf("unexpected namespaces: %v", namespaces)
					break
				}
			}
		})
	}
}

func expectAction(t *testing.T, actions []k8stesting.Action, expected [][]string) {
	t.Helper()
	if len(actions) != len(expected) {
		t.Fatalf("Expected at least %d actions, got %d", len(expected), len(actions))
	}

	for i, action := range actions {
		verb := expected[i][0]
		if action.GetVerb() != verb {
			t.Errorf("Expected action %d verb to be %s, got %s", i, verb, action.GetVerb())
		}
		resource := expected[i][1]
		if action.GetResource().Resource != resource {
			t.Errorf("Expected action %d resource to be %s, got %s", i, resource, action.GetResource().Resource)
		}
	}
}
