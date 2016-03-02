/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package namespace

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestFinalized(t *testing.T) {
	testNamespace := &api.Namespace{
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"a", "b"},
		},
	}
	if finalized(testNamespace) {
		t.Errorf("Unexpected result, namespace is not finalized")
	}
	testNamespace.Spec.Finalizers = []api.FinalizerName{}
	if !finalized(testNamespace) {
		t.Errorf("Expected object to be finalized")
	}
}

func TestFinalizeNamespaceFunc(t *testing.T) {
	mockClient := &fake.Clientset{}
	testNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "test",
			ResourceVersion: "1",
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes", "other"},
		},
	}
	finalizeNamespaceFunc(mockClient, testNamespace)
	actions := mockClient.Actions()
	if len(actions) != 1 {
		t.Errorf("Expected 1 mock client action, but got %v", len(actions))
	}
	if !actions[0].Matches("create", "namespaces") || actions[0].GetSubresource() != "finalize" {
		t.Errorf("Expected finalize-namespace action %v", actions[0])
	}
	finalizers := actions[0].(core.CreateAction).GetObject().(*api.Namespace).Spec.Finalizers
	if len(finalizers) != 1 {
		t.Errorf("There should be a single finalizer remaining")
	}
	if "other" != string(finalizers[0]) {
		t.Errorf("Unexpected finalizer value, %v", finalizers[0])
	}
}

func testSyncNamespaceThatIsTerminating(t *testing.T, versions *unversioned.APIVersions) {
	now := unversioned.Now()
	testNamespacePendingFinalize := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              "test",
			ResourceVersion:   "1",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes"},
		},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}
	testNamespaceFinalizeComplete := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:              "test",
			ResourceVersion:   "1",
			DeletionTimestamp: &now,
		},
		Spec: api.NamespaceSpec{},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}

	// TODO: Reuse the constants for all these strings from testclient
	pendingActionSet := sets.NewString(
		strings.Join([]string{"get", "namespaces", ""}, "-"),
		strings.Join([]string{"delete-collection", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"delete-collection", "resourcequotas", ""}, "-"),
		strings.Join([]string{"delete-collection", "secrets", ""}, "-"),
		strings.Join([]string{"delete-collection", "configmaps", ""}, "-"),
		strings.Join([]string{"delete-collection", "limitranges", ""}, "-"),
		strings.Join([]string{"delete-collection", "events", ""}, "-"),
		strings.Join([]string{"delete-collection", "serviceaccounts", ""}, "-"),
		strings.Join([]string{"delete-collection", "persistentvolumeclaims", ""}, "-"),
		strings.Join([]string{"create", "namespaces", "finalize"}, "-"),
	)

	if containsVersion(versions, "extensions/v1beta1") {
		pendingActionSet.Insert(
			strings.Join([]string{"delete-collection", "daemonsets", ""}, "-"),
			strings.Join([]string{"delete-collection", "deployments", ""}, "-"),
			strings.Join([]string{"delete-collection", "replicasets", ""}, "-"),
			strings.Join([]string{"delete-collection", "jobs", ""}, "-"),
			strings.Join([]string{"delete-collection", "horizontalpodautoscalers", ""}, "-"),
			strings.Join([]string{"delete-collection", "ingresses", ""}, "-"),
			strings.Join([]string{"get", "resource", ""}, "-"),
		)
	}

	scenarios := map[string]struct {
		testNamespace     *api.Namespace
		expectedActionSet sets.String
	}{
		"pending-finalize": {
			testNamespace:     testNamespacePendingFinalize,
			expectedActionSet: pendingActionSet,
		},
		"complete-finalize": {
			testNamespace: testNamespaceFinalizeComplete,
			expectedActionSet: sets.NewString(
				strings.Join([]string{"get", "namespaces", ""}, "-"),
				strings.Join([]string{"delete", "namespaces", ""}, "-"),
			),
		},
	}

	for scenario, testInput := range scenarios {
		mockClient := fake.NewSimpleClientset(testInput.testNamespace)
		if containsVersion(versions, "extensions/v1beta1") {
			resources := []unversioned.APIResource{}
			for _, resource := range []string{"daemonsets", "deployments", "replicasets", "jobs", "horizontalpodautoscalers", "ingresses"} {
				resources = append(resources, unversioned.APIResource{Name: resource})
			}
			mockClient.Resources = map[string]*unversioned.APIResourceList{
				"extensions/v1beta1": {
					GroupVersion: "extensions/v1beta1",
					APIResources: resources,
				},
			}
		}
		err := syncNamespace(mockClient, versions, testInput.testNamespace)
		if err != nil {
			t.Errorf("scenario %s - Unexpected error when synching namespace %v", scenario, err)
		}
		actionSet := sets.NewString()
		for _, action := range mockClient.Actions() {
			actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource(), action.GetSubresource()}, "-"))
		}
		if !actionSet.HasAll(testInput.expectedActionSet.List()...) {
			t.Errorf("scenario %s - Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", scenario, testInput.expectedActionSet, actionSet, testInput.expectedActionSet.Difference(actionSet))
		}
		if !testInput.expectedActionSet.HasAll(actionSet.List()...) {
			t.Errorf("scenario %s - Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", scenario, testInput.expectedActionSet, actionSet, actionSet.Difference(testInput.expectedActionSet))
		}
	}
}

func TestRetryOnConflictError(t *testing.T) {
	mockClient := &fake.Clientset{}
	numTries := 0
	retryOnce := func(kubeClient clientset.Interface, namespace *api.Namespace) (*api.Namespace, error) {
		numTries++
		if numTries <= 1 {
			return namespace, errors.NewConflict(api.Resource("namespaces"), namespace.Name, fmt.Errorf("ERROR!"))
		}
		return namespace, nil
	}
	namespace := &api.Namespace{}
	_, err := retryOnConflictError(mockClient, namespace, retryOnce)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if numTries != 2 {
		t.Errorf("Expected %v, but got %v", 2, numTries)
	}
}

func TestSyncNamespaceThatIsTerminatingNonExperimental(t *testing.T) {
	testSyncNamespaceThatIsTerminating(t, &unversioned.APIVersions{})
}

func TestSyncNamespaceThatIsTerminatingV1Beta1(t *testing.T) {
	testSyncNamespaceThatIsTerminating(t, &unversioned.APIVersions{Versions: []string{"extensions/v1beta1"}})
}

func TestSyncNamespaceThatIsActive(t *testing.T) {
	mockClient := &fake.Clientset{}
	testNamespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "test",
			ResourceVersion: "1",
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes"},
		},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceActive,
		},
	}
	err := syncNamespace(mockClient, &unversioned.APIVersions{}, testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	if len(mockClient.Actions()) != 0 {
		t.Errorf("Expected no action from controller, but got: %v", mockClient.Actions())
	}
}
