/*
Copyright 2019 The Kubernetes Authors.

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

package storage

import (
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestEviction(t *testing.T) {
	testcases := []struct {
		name     string
		pdbs     []runtime.Object
		pod      *api.Pod
		eviction *policy.Eviction

		expectError   bool
		expectDeleted bool
	}{
		{
			name:          "no pdbs, unscheduled pod, nil delete options, deletes immediately",
			pdbs:          nil,
			pod:           validNewPod(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}},
			expectDeleted: true,
		},
		{
			name:          "no pdbs, scheduled pod, nil delete options, deletes gracefully",
			pdbs:          nil,
			pod:           func() *api.Pod { pod := validNewPod(); pod.Spec.NodeName = "foo"; return pod }(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}},
			expectDeleted: false, // not deleted immediately because of graceful deletion
		},
		{
			name:          "no pdbs, scheduled pod, empty delete options, deletes gracefully",
			pdbs:          nil,
			pod:           func() *api.Pod { pod := validNewPod(); pod.Spec.NodeName = "foo"; return pod }(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: &metav1.DeleteOptions{}},
			expectDeleted: false, // not deleted immediately because of graceful deletion
		},
		{
			name:          "no pdbs, scheduled pod, graceless delete options, deletes immediately",
			pdbs:          nil,
			pod:           func() *api.Pod { pod := validNewPod(); pod.Spec.NodeName = "foo"; return pod }(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
		},
		{
			name: "matching pdbs with no disruptions allowed",
			pdbs: []runtime.Object{&policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policy.PodDisruptionBudgetStatus{PodDisruptionsAllowed: 0},
			}},
			pod: func() *api.Pod {
				pod := validNewPod()
				pod.Labels = map[string]string{"a": "true"}
				pod.Spec.NodeName = "foo"
				return pod
			}(),
			eviction:    &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError: true,
		},
		{
			name: "matching pdbs with disruptions allowed",
			pdbs: []runtime.Object{&policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policy.PodDisruptionBudgetStatus{PodDisruptionsAllowed: 1},
			}},
			pod: func() *api.Pod {
				pod := validNewPod()
				pod.Labels = map[string]string{"a": "true"}
				pod.Spec.NodeName = "foo"
				return pod
			}(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
		},
		{
			name: "non-matching pdbs",
			pdbs: []runtime.Object{&policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"b": "true"}}},
				Status:     policy.PodDisruptionBudgetStatus{PodDisruptionsAllowed: 0},
			}},
			pod: func() *api.Pod {
				pod := validNewPod()
				pod.Labels = map[string]string{"a": "true"}
				pod.Spec.NodeName = "foo"
				return pod
			}(),
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
			storage, _, _, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()
			if tc.pod != nil {
				if _, err := storage.Create(testContext, tc.pod, nil, &metav1.CreateOptions{}); err != nil {
					t.Error(err)
				}
			}

			client := fake.NewSimpleClientset(tc.pdbs...)
			evictionRest := newEvictionStorage(storage.Store, client.Policy())
			_, err := evictionRest.Create(testContext, tc.eviction, nil, &metav1.CreateOptions{})
			if (err != nil) != tc.expectError {
				t.Errorf("expected error=%v, got %v", tc.expectError, err)
				return
			}
			if tc.expectError {
				return
			}

			if tc.pod != nil {
				existingPod, err := storage.Get(testContext, tc.pod.Name, &metav1.GetOptions{})
				if tc.expectDeleted {
					if !apierrors.IsNotFound(err) {
						t.Errorf("expected to be deleted, lookup returned %#v", existingPod)
					}
					return
				} else if apierrors.IsNotFound(err) {
					t.Errorf("expected graceful deletion, got %v", err)
					return
				}

				if err != nil {
					t.Errorf("%#v", err)
					return
				}

				if existingPod.(*api.Pod).DeletionTimestamp == nil {
					t.Errorf("expected gracefully deleted pod with deletionTimestamp set, got %#v", existingPod)
				}
			}
		})
	}
}
