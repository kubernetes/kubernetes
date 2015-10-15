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

package namespacecontroller

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
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
	mockClient := &testclient.Fake{}
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
	finalizers := actions[0].(testclient.CreateAction).GetObject().(*api.Namespace).Spec.Finalizers
	if len(finalizers) != 1 {
		t.Errorf("There should be a single finalizer remaining")
	}
	if "other" != string(finalizers[0]) {
		t.Errorf("Unexpected finalizer value, %v", finalizers[0])
	}
}

func testSyncNamespaceThatIsTerminating(t *testing.T, versions *unversioned.APIVersions) {
	mockClient := &testclient.Fake{}
	now := unversioned.Now()
	testNamespace := &api.Namespace{
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

	if containsVersion(versions, "extensions/v1beta1") {
		resources := []unversioned.APIResource{}
		for _, resource := range []string{"daemonsets", "deployments", "jobs", "horizontalpodautoscalers", "ingress"} {
			resources = append(resources, unversioned.APIResource{Name: resource})
		}
		mockClient.Resources = map[string]*unversioned.APIResourceList{
			"extensions/v1beta1": {
				GroupVersion: "extensions/v1beta1",
				APIResources: resources,
			},
		}
	}

	err := syncNamespace(mockClient, versions, testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	// TODO: Reuse the constants for all these strings from testclient
	expectedActionSet := sets.NewString(
		strings.Join([]string{"list", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"list", "resourcequotas", ""}, "-"),
		strings.Join([]string{"list", "secrets", ""}, "-"),
		strings.Join([]string{"list", "limitranges", ""}, "-"),
		strings.Join([]string{"list", "events", ""}, "-"),
		strings.Join([]string{"list", "serviceaccounts", ""}, "-"),
		strings.Join([]string{"list", "persistentvolumeclaims", ""}, "-"),
		strings.Join([]string{"create", "namespaces", "finalize"}, "-"),
		strings.Join([]string{"delete", "namespaces", ""}, "-"),
	)

	if containsVersion(versions, "extensions/v1beta1") {
		expectedActionSet.Insert(
			strings.Join([]string{"list", "daemonsets", ""}, "-"),
			strings.Join([]string{"list", "deployments", ""}, "-"),
			strings.Join([]string{"list", "jobs", ""}, "-"),
			strings.Join([]string{"list", "horizontalpodautoscalers", ""}, "-"),
			strings.Join([]string{"list", "ingress", ""}, "-"),
			strings.Join([]string{"get", "resource", ""}, "-"),
		)
	}

	actionSet := sets.NewString()
	for _, action := range mockClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource(), action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, expectedActionSet.Difference(actionSet))
	}
	if !expectedActionSet.HasAll(actionSet.List()...) {
		t.Errorf("Expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet, actionSet, actionSet.Difference(expectedActionSet))
	}
}

func TestRetryOnConflictError(t *testing.T) {
	mockClient := &testclient.Fake{}
	numTries := 0
	retryOnce := func(kubeClient client.Interface, namespace *api.Namespace) (*api.Namespace, error) {
		numTries++
		if numTries <= 1 {
			return namespace, errors.NewConflict(namespace.Kind, namespace.Name, fmt.Errorf("ERROR!"))
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
	mockClient := &testclient.Fake{}
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

func TestRunStop(t *testing.T) {
	mockClient := &testclient.Fake{}

	nsController := NewNamespaceController(mockClient, &unversioned.APIVersions{}, 1*time.Second)

	if nsController.StopEverything != nil {
		t.Errorf("Non-running manager should not have a stop channel.  Got %v", nsController.StopEverything)
	}

	nsController.Run()

	if nsController.StopEverything == nil {
		t.Errorf("Running manager should have a stop channel.  Got nil")
	}

	nsController.Stop()

	if nsController.StopEverything != nil {
		t.Errorf("Non-running manager should not have a stop channel.  Got %v", nsController.StopEverything)
	}
}
