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
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	policyv1 "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/kubernetes/fake"
	podapi "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func TestEviction(t *testing.T) {
	testcases := []struct {
		name     string
		pdbs     []runtime.Object
		policies []*policyv1.UnhealthyPodEvictionPolicyType
		eviction *policy.Eviction

		badNameInURL bool

		expectError   string
		expectDeleted bool
		podPhase      api.PodPhase
		podName       string
	}{
		{
			name: "matching pdbs with no disruptions allowed, pod running",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:    &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError: "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo needs 0 healthy pods and has 0 currently",
			podPhase:    api.PodRunning,
			podName:     "t1",
			policies:    []*policyv1.UnhealthyPodEvictionPolicyType{nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget)}, // AlwaysAllow would terminate the pod since Running pods are not guarded by this policy
		},
		{
			name: "matching pdbs with no disruptions allowed, pod pending",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t2", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:   "",
			podPhase:      api.PodPending,
			expectDeleted: true,
			podName:       "t2",
		},
		{
			name: "matching pdbs with no disruptions allowed, pod succeeded",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t3", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:   "",
			podPhase:      api.PodSucceeded,
			expectDeleted: true,
			podName:       "t3",
		},
		{
			name: "matching pdbs with no disruptions allowed, pod failed",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t4", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:   "",
			podPhase:      api.PodFailed,
			expectDeleted: true,
			podName:       "t4",
		},
		{
			name: "matching pdbs with disruptions allowed",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t5", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
			podName:       "t5",
		},
		{
			name: "non-matching pdbs",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"b": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:      &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t6", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectDeleted: true,
			podName:       "t6",
		},
		{
			name: "matching pdbs with disruptions allowed but bad name in Url",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
			badNameInURL: true,
			eviction:     &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t7", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:  "name in URL does not match name in Eviction object: BadRequest",
			podName:      "t7",
		},
		{
			name: "matching pdbs with no disruptions allowed, pod running, empty selector",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:    &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t8", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError: "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo needs 0 healthy pods and has 0 currently",
			podPhase:    api.PodRunning,
			podName:     "t8",
			policies:    []*policyv1.UnhealthyPodEvictionPolicyType{nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget)}, // AlwaysAllow would terminate the pod since Running pods are not guarded by this policy
		},
	}

	for _, unhealthyPodEvictionPolicy := range []*policyv1.UnhealthyPodEvictionPolicyType{nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget), unhealthyPolicyPtr(policyv1.AlwaysAllow)} {
		for _, tc := range testcases {
			if len(tc.policies) > 0 && !hasUnhealthyPolicy(tc.policies, unhealthyPodEvictionPolicy) {
				// unhealthyPodEvictionPolicy is not covered by this test
				continue
			}
			t.Run(fmt.Sprintf("%v with %v policy", tc.name, unhealthyPolicyStr(unhealthyPodEvictionPolicy)), func(t *testing.T) {
				// same test runs multiple times, make copy of objects to have unique ones
				evictionCopy := tc.eviction.DeepCopy()
				var pdbsCopy []runtime.Object
				for _, pdb := range tc.pdbs {
					pdbCopy := pdb.DeepCopyObject()
					pdbCopy.(*policyv1.PodDisruptionBudget).Spec.UnhealthyPodEvictionPolicy = unhealthyPodEvictionPolicy
					pdbsCopy = append(pdbsCopy, pdbCopy)
				}

				testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
				storage, _, statusStorage, server := newStorage(t)
				defer server.Terminate(t)
				defer storage.Store.DestroyFunc()

				pod := validNewPod()
				pod.Name = tc.podName
				pod.Labels = map[string]string{"a": "true"}
				pod.Spec.NodeName = "foo"
				if _, err := storage.Create(testContext, pod, nil, &metav1.CreateOptions{}); err != nil {
					t.Error(err)
				}

				if tc.podPhase != "" {
					pod.Status.Phase = tc.podPhase
					_, _, err := statusStorage.Update(testContext, pod.Name, rest.DefaultUpdatedObjectInfo(pod), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
					if err != nil {
						t.Errorf("Unexpected error: %v", err)
					}
				}

				client := fake.NewSimpleClientset(pdbsCopy...)
				evictionRest := newEvictionStorage(storage.Store, client.PolicyV1())

				name := pod.Name
				if tc.badNameInURL {
					name += "bad-name"
				}

				_, err := evictionRest.Create(testContext, name, evictionCopy, nil, &metav1.CreateOptions{})
				gotErr := errToString(err)
				if gotErr != tc.expectError {
					t.Errorf("error mismatch: expected %v, got %v; name %v", tc.expectError, gotErr, pod.Name)
					return
				}
				if tc.badNameInURL {
					if err == nil {
						t.Error("expected error here, but got nil")
						return
					}
					if err.Error() != "name in URL does not match name in Eviction object" {
						t.Errorf("got unexpected error: %v", err)
					}
				}
				if tc.expectError != "" {
					return
				}

				existingPod, err := storage.Get(testContext, pod.Name, &metav1.GetOptions{})
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
			})
		}
	}
}

func TestEvictionIgnorePDB(t *testing.T) {
	testcases := []struct {
		name     string
		pdbs     []runtime.Object
		policies []*policyv1.UnhealthyPodEvictionPolicyType
		eviction *policy.Eviction

		expectError         string
		podPhase            api.PodPhase
		podName             string
		expectedDeleteCount int
		podTerminating      bool
		prc                 *api.PodCondition
	}{
		{
			name: "pdbs No disruptions allowed, pod pending, first delete conflict, pod still pending, pod deleted successfully",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "",
			podPhase:            api.PodPending,
			podName:             "t1",
			expectedDeleteCount: 3,
		},
		// This test case is critical. If it is removed or broken we may
		// regress and allow a pod to be deleted without checking PDBs when the
		// pod should not be deleted.
		{
			name: "pdbs No disruptions allowed, pod pending, first delete conflict, pod becomes running, continueToPDBs",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t2", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo needs 0 healthy pods and has 0 currently",
			podPhase:            api.PodPending,
			podName:             "t2",
			expectedDeleteCount: 1,
			policies:            []*policyv1.UnhealthyPodEvictionPolicyType{nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget)}, // AlwaysAllow does not continueToPDBs, but straight to deletion
		},
		{
			name: "pdbs No disruptions allowed, pod pending, first delete conflict, pod becomes running, skip PDB check, conflict",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t2", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo is still being processed by the server.",
			podPhase:            api.PodPending,
			podName:             "t2",
			podTerminating:      false,
			expectedDeleteCount: 2,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionFalse,
			},
			policies: []*policyv1.UnhealthyPodEvictionPolicyType{unhealthyPolicyPtr(policyv1.AlwaysAllow)}, // nil, IfHealthyBudget continueToPDBs
		},
		{
			name: "pdbs disruptions allowed, pod pending, first delete conflict, pod becomes running, continueToPDBs",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t3", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "",
			podPhase:            api.PodPending,
			podName:             "t3",
			expectedDeleteCount: 2,
			policies:            []*policyv1.UnhealthyPodEvictionPolicyType{nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget)}, // AlwaysAllow does not continueToPDBs, but straight to deletion if pod not healthy (ready)
		},
		{
			name: "pod pending, always conflict on delete",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t4", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         `Operation cannot be fulfilled on tests "2": message: Conflict`,
			podPhase:            api.PodPending,
			podName:             "t4",
			expectedDeleteCount: EvictionsRetry.Steps,
		},
		{
			name: "pod pending, always conflict on delete, user provided ResourceVersion constraint",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t5", Namespace: "default"}, DeleteOptions: metav1.NewRVDeletionPrecondition("userProvided")},
			expectError:         `Operation cannot be fulfilled on tests "2": message: Conflict`,
			podPhase:            api.PodPending,
			podName:             "t5",
			expectedDeleteCount: 1,
		},
		{
			name: "matching pdbs with no disruptions allowed, pod terminating",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t6", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(300)},
			expectError:         "",
			podName:             "t6",
			expectedDeleteCount: 1,
			podTerminating:      true,
		},
		{
			name: "matching pdbs with no disruptions allowed, pod running, pod healthy, unhealthy pod not ours",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					// This simulates 3 pods expected, our pod healthy, unhealthy pod is not ours.
					DisruptionsAllowed: 0,
					CurrentHealthy:     2,
					DesiredHealthy:     2,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t7", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo needs 2 healthy pods and has 2 currently",
			podName:             "t7",
			expectedDeleteCount: 0,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
		{
			name: "matching pdbs with disruptions allowed, pod running, pod healthy, healthy pod ours, deletes pod by honoring the PDB",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					DisruptionsAllowed: 1,
					CurrentHealthy:     3,
					DesiredHealthy:     3,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t8", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "",
			podName:             "t8",
			expectedDeleteCount: 1,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
		{
			name: "matching pdbs with no disruptions allowed, pod running, pod unhealthy, unhealthy pod ours",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					// This simulates 3 pods expected, our pod unhealthy
					DisruptionsAllowed: 0,
					CurrentHealthy:     2,
					DesiredHealthy:     2,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t8", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "",
			podName:             "t8",
			expectedDeleteCount: 1,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionFalse,
			},
		},
		{
			name: "matching pdbs with no disruptions allowed, pod running, pod unhealthy, unhealthy pod ours, skips the PDB check and deletes",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t8", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "",
			podName:             "t8",
			expectedDeleteCount: 1,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionFalse,
			},
			policies: []*policyv1.UnhealthyPodEvictionPolicyType{unhealthyPolicyPtr(policyv1.AlwaysAllow)}, // nil, IfHealthyBudget would not skip the PDB check
		},
		{
			// This case should return the 529 retry error.
			name: "matching pdbs with no disruptions allowed, pod running, pod unhealthy, unhealthy pod ours, resource version conflict",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					// This simulates 3 pods expected, our pod unhealthy
					DisruptionsAllowed: 0,
					CurrentHealthy:     2,
					DesiredHealthy:     2,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t9", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo is still being processed by the server.",
			podName:             "t9",
			expectedDeleteCount: 1,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionFalse,
			},
		},
		{
			// This case should return the 529 retry error.
			name: "matching pdbs with no disruptions allowed, pod running, pod unhealthy, unhealthy pod ours, other error on delete",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					// This simulates 3 pods expected, our pod unhealthy
					DisruptionsAllowed: 0,
					CurrentHealthy:     2,
					DesiredHealthy:     2,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t10", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "test designed to error: BadRequest",
			podName:             "t10",
			expectedDeleteCount: 1,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionFalse,
			},
		},
		{
			name: "matching pdbs with no disruptions allowed, pod running, pod healthy, empty selector, pod not deleted by honoring the PDB",
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{}},
				Status: policyv1.PodDisruptionBudgetStatus{
					DisruptionsAllowed: 0,
					CurrentHealthy:     3,
					DesiredHealthy:     3,
				},
			}},
			eviction:            &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "t11", Namespace: "default"}, DeleteOptions: metav1.NewDeleteOptions(0)},
			expectError:         "Cannot evict pod as it would violate the pod's disruption budget.: TooManyRequests: The disruption budget foo needs 3 healthy pods and has 3 currently",
			podName:             "t11",
			expectedDeleteCount: 0,
			podTerminating:      false,
			podPhase:            api.PodRunning,
			prc: &api.PodCondition{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
	}

	for _, unhealthyPodEvictionPolicy := range []*policyv1.UnhealthyPodEvictionPolicyType{unhealthyPolicyPtr(policyv1.AlwaysAllow), nil, unhealthyPolicyPtr(policyv1.IfHealthyBudget)} {
		for _, tc := range testcases {
			if len(tc.policies) > 0 && !hasUnhealthyPolicy(tc.policies, unhealthyPodEvictionPolicy) {
				// unhealthyPodEvictionPolicy is not covered by this test
				continue
			}
			t.Run(fmt.Sprintf("%v with %v policy", tc.name, unhealthyPolicyStr(unhealthyPodEvictionPolicy)), func(t *testing.T) {
				// same test runs 3 times, make copy of objects to have unique ones
				evictionCopy := tc.eviction.DeepCopy()
				prcCopy := tc.prc.DeepCopy()
				var pdbsCopy []runtime.Object
				for _, pdb := range tc.pdbs {
					pdbCopy := pdb.DeepCopyObject()
					pdbCopy.(*policyv1.PodDisruptionBudget).Spec.UnhealthyPodEvictionPolicy = unhealthyPodEvictionPolicy
					pdbsCopy = append(pdbsCopy, pdbCopy)
				}

				testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
				ms := &mockStore{
					deleteCount: 0,
				}

				pod := validNewPod()
				pod.Name = tc.podName
				pod.Labels = map[string]string{"a": "true"}
				pod.Spec.NodeName = "foo"
				if tc.podPhase != "" {
					pod.Status.Phase = tc.podPhase
				}

				if tc.podTerminating {
					currentTime := metav1.Now()
					pod.ObjectMeta.DeletionTimestamp = &currentTime
				}

				// Setup pod condition
				if tc.prc != nil {
					if !podapi.UpdatePodCondition(&pod.Status, prcCopy) {
						t.Fatalf("Unable to update pod ready condition")
					}
				}

				client := fake.NewSimpleClientset(pdbsCopy...)
				evictionRest := newEvictionStorage(ms, client.PolicyV1())

				name := pod.Name
				ms.pod = pod

				_, err := evictionRest.Create(testContext, name, evictionCopy, nil, &metav1.CreateOptions{})
				gotErr := errToString(err)
				if gotErr != tc.expectError {
					t.Errorf("error mismatch: expected %v, got %v; name %v", tc.expectError, gotErr, pod.Name)
					return
				}

				if tc.expectedDeleteCount != ms.deleteCount {
					t.Errorf("expected delete count=%v, got %v; name %v", tc.expectedDeleteCount, ms.deleteCount, pod.Name)
				}
			})
		}
	}
}

func TestEvictionDryRun(t *testing.T) {
	testcases := []struct {
		name            string
		evictionOptions *metav1.DeleteOptions
		requestOptions  *metav1.CreateOptions
		pdbs            []runtime.Object
	}{
		{
			name:            "just request-options",
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
			evictionOptions: &metav1.DeleteOptions{},
		},
		{
			name:            "just eviction-options",
			requestOptions:  &metav1.CreateOptions{},
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
		},
		{
			name:            "both options",
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
		},
		{
			name:            "with pdbs",
			evictionOptions: &metav1.DeleteOptions{DryRun: []string{"All"}},
			requestOptions:  &metav1.CreateOptions{DryRun: []string{"All"}},
			pdbs: []runtime.Object{&policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status:     policyv1.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
			}},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
			storage, _, _, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			pod := validNewPod()
			pod.Labels = map[string]string{"a": "true"}
			pod.Spec.NodeName = "foo"
			if _, err := storage.Create(testContext, pod, nil, &metav1.CreateOptions{}); err != nil {
				t.Error(err)
			}

			client := fake.NewSimpleClientset(tc.pdbs...)
			evictionRest := newEvictionStorage(storage.Store, client.PolicyV1())
			eviction := &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}, DeleteOptions: tc.evictionOptions}
			_, err := evictionRest.Create(testContext, pod.Name, eviction, nil, tc.requestOptions)
			if err != nil {
				t.Fatalf("Failed to run eviction: %v", err)
			}
		})
	}
}

func TestEvictionPDBStatus(t *testing.T) {
	testcases := []struct {
		name                       string
		pdb                        *policyv1.PodDisruptionBudget
		expectedDisruptionsAllowed int32
		expectedReason             string
	}{
		{
			name: "pdb status is updated after eviction",
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					DisruptionsAllowed: 1,
					Conditions: []metav1.Condition{
						{
							Type:   policyv1.DisruptionAllowedCondition,
							Reason: policyv1.SufficientPodsReason,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectedDisruptionsAllowed: 0,
			expectedReason:             policyv1.InsufficientPodsReason,
		},
		{
			name: "condition reason is only updated if AllowedDisruptions becomes 0",
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       policyv1.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				Status: policyv1.PodDisruptionBudgetStatus{
					DisruptionsAllowed: 3,
					Conditions: []metav1.Condition{
						{
							Type:   policyv1.DisruptionAllowedCondition,
							Reason: policyv1.SufficientPodsReason,
							Status: metav1.ConditionTrue,
						},
					},
				},
			},
			expectedDisruptionsAllowed: 2,
			expectedReason:             policyv1.SufficientPodsReason,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
			storage, _, statusStorage, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			client := fake.NewSimpleClientset(tc.pdb)
			for _, podName := range []string{"foo-1", "foo-2"} {
				pod := validNewPod()
				pod.Labels = map[string]string{"a": "true"}
				pod.ObjectMeta.Name = podName
				pod.Spec.NodeName = "foo"
				newPod, err := storage.Create(testContext, pod, nil, &metav1.CreateOptions{})
				if err != nil {
					t.Error(err)
				}
				(newPod.(*api.Pod)).Status.Phase = api.PodRunning
				_, _, err = statusStorage.Update(testContext, pod.Name, rest.DefaultUpdatedObjectInfo(newPod),
					nil, nil, false, &metav1.UpdateOptions{})
				if err != nil {
					t.Error(err)
				}
			}

			evictionRest := newEvictionStorage(storage.Store, client.PolicyV1())
			eviction := &policy.Eviction{ObjectMeta: metav1.ObjectMeta{Name: "foo-1", Namespace: "default"}, DeleteOptions: &metav1.DeleteOptions{}}
			_, err := evictionRest.Create(testContext, "foo-1", eviction, nil, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to run eviction: %v", err)
			}

			existingPDB, err := client.PolicyV1().PodDisruptionBudgets(metav1.NamespaceDefault).Get(context.TODO(), tc.pdb.Name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("%#v", err)
				return
			}

			if want, got := tc.expectedDisruptionsAllowed, existingPDB.Status.DisruptionsAllowed; got != want {
				t.Errorf("expected DisruptionsAllowed to be %d, but got %d", want, got)
			}

			cond := apimeta.FindStatusCondition(existingPDB.Status.Conditions, policyv1.DisruptionAllowedCondition)
			if want, got := tc.expectedReason, cond.Reason; want != got {
				t.Errorf("expected Reason to be %q, but got %q", want, got)
			}
		})
	}
}

func TestAddConditionAndDelete(t *testing.T) {
	cases := []struct {
		name              string
		initialPod        bool
		makeDeleteOptions func(*api.Pod) *metav1.DeleteOptions
		expectErr         string
	}{
		{
			name:              "simple",
			initialPod:        true,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions { return &metav1.DeleteOptions{} },
		},
		{
			name:              "missing",
			initialPod:        false,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions { return &metav1.DeleteOptions{} },
			expectErr:         "not found",
		},
		{
			name:       "valid uid",
			initialPod: true,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions {
				return &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &pod.UID}}
			},
		},
		{
			name:       "invalid uid",
			initialPod: true,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions {
				badUID := pod.UID + "1"
				return &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &badUID}}
			},
			expectErr: "The object might have been deleted and then recreated",
		},
		{
			name:       "valid resourceVersion",
			initialPod: true,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions {
				return &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &pod.ResourceVersion}}
			},
		},
		{
			name:       "invalid resourceVersion",
			initialPod: true,
			makeDeleteOptions: func(pod *api.Pod) *metav1.DeleteOptions {
				badRV := pod.ResourceVersion + "1"
				return &metav1.DeleteOptions{Preconditions: &metav1.Preconditions{ResourceVersion: &badRV}}
			},
			expectErr: "The object might have been modified",
		},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)

	storage, _, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	client := fake.NewSimpleClientset()
	evictionRest := newEvictionStorage(storage.Store, client.PolicyV1())

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var deleteOptions *metav1.DeleteOptions
			if tc.initialPod {
				newPod := validNewPod()
				createdObj, err := storage.Create(testContext, newPod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					zero := int64(0)
					if _, _, err := storage.Delete(testContext, newPod.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{GracePeriodSeconds: &zero}); err != nil && !apierrors.IsNotFound(err) {
						t.Fatal(err)
					}
				})
				deleteOptions = tc.makeDeleteOptions(createdObj.(*api.Pod))
			} else {
				deleteOptions = tc.makeDeleteOptions(nil)
			}
			if deleteOptions == nil {
				deleteOptions = &metav1.DeleteOptions{}
			}

			err := addConditionAndDeletePod(evictionRest, testContext, "foo", rest.ValidateAllObjectFunc, deleteOptions)
			if err == nil {
				if tc.expectErr != "" {
					t.Fatalf("expected err containing %q, got none", tc.expectErr)
				}
				return
			}
			if tc.expectErr == "" {
				t.Fatalf("unexpected err: %v", err)
			}
			if !strings.Contains(err.Error(), tc.expectErr) {
				t.Fatalf("expected err containing %q, got %v", tc.expectErr, err)
			}
		})
	}
}

func resource(resource string) schema.GroupResource {
	return schema.GroupResource{Group: "", Resource: resource}
}

type mockStore struct {
	deleteCount int
	pod         *api.Pod
}

func (ms *mockStore) mutatorDeleteFunc(count int, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	if ms.pod.Name == "t4" {
		// Always return error for this pod
		return nil, false, apierrors.NewConflict(resource("tests"), "2", errors.New("message"))
	}
	if ms.pod.Name == "t6" || ms.pod.Name == "t8" {
		// t6: This pod has a deletionTimestamp and should not raise conflict on delete
		// t8: This pod should not have a resource conflict.
		return nil, true, nil
	}
	if ms.pod.Name == "t10" {
		return nil, false, apierrors.NewBadRequest("test designed to error")
	}
	if count == 1 {
		// This is a hack to ensure that some test pods don't change phase
		// but do change resource version
		if ms.pod.Name != "t1" && ms.pod.Name != "t5" {
			ms.pod.Status.Phase = api.PodRunning
		}
		ms.pod.ResourceVersion = "999"
		// Always return conflict on the first attempt
		return nil, false, apierrors.NewConflict(resource("tests"), "2", errors.New("message"))
	}
	// Compare enforce deletionOptions
	if options == nil || options.Preconditions == nil || options.Preconditions.ResourceVersion == nil {
		return nil, true, nil
	} else if *options.Preconditions.ResourceVersion != "1000" {
		// Here we're simulating that the pod has changed resource version again
		// pod "t4" should make it here, this validates we're getting the latest
		// resourceVersion of the pod and successfully delete on the next deletion
		// attempt after this one.
		ms.pod.ResourceVersion = "1000"
		return nil, false, apierrors.NewConflict(resource("tests"), "2", errors.New("message"))
	}
	return nil, true, nil
}

func (ms *mockStore) Delete(ctx context.Context, name string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	ms.deleteCount++
	return ms.mutatorDeleteFunc(ms.deleteCount, options)
}

func (ms *mockStore) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return nil, nil
}

func (ms *mockStore) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	return ms.pod, false, nil
}

func (ms *mockStore) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return ms.pod, nil
}

func (ms *mockStore) New() runtime.Object {
	return nil
}

func (ms *mockStore) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	return nil, nil
}

func (ms *mockStore) DeleteCollection(ctx context.Context, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	return nil, nil
}

func (ms *mockStore) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return nil, nil
}

func (ms *mockStore) NewList() runtime.Object {
	return nil
}

func (ms *mockStore) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return nil, nil
}

func (ms *mockStore) Destroy() {
}

func unhealthyPolicyPtr(unhealthyPodEvictionPolicy policyv1.UnhealthyPodEvictionPolicyType) *policyv1.UnhealthyPodEvictionPolicyType {
	return &unhealthyPodEvictionPolicy
}

func unhealthyPolicyStr(unhealthyPodEvictionPolicy *policyv1.UnhealthyPodEvictionPolicyType) string {
	if unhealthyPodEvictionPolicy == nil {
		return "nil"
	}
	return string(*unhealthyPodEvictionPolicy)
}

func hasUnhealthyPolicy(unhealthyPolicies []*policyv1.UnhealthyPodEvictionPolicyType, unhealthyPolicy *policyv1.UnhealthyPodEvictionPolicyType) bool {
	for _, p := range unhealthyPolicies {
		if reflect.DeepEqual(unhealthyPolicy, p) {
			return true
		}
	}
	return false
}

func errToString(err error) string {
	result := ""
	if err != nil {
		result = err.Error()
		if statusErr, ok := err.(*apierrors.StatusError); ok {
			result += fmt.Sprintf(": %v", statusErr.ErrStatus.Reason)
			if statusErr.ErrStatus.Details != nil {
				for _, cause := range statusErr.ErrStatus.Details.Causes {
					if !strings.HasSuffix(err.Error(), cause.Message) {
						result += fmt.Sprintf(": %v", cause.Message)
					}
				}
			}
		}
	}
	return result
}
