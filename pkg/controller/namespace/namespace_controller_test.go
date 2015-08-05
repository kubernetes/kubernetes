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
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/testclient"
	"k8s.io/kubernetes/pkg/util"
)

func TestFinalized(t *testing.T) {
	testNamespace := api.Namespace{
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

func TestFinalize(t *testing.T) {
	mockClient := &testclient.Fake{}
	testNamespace := api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:            "test",
			ResourceVersion: "1",
		},
		Spec: api.NamespaceSpec{
			Finalizers: []api.FinalizerName{"kubernetes", "other"},
		},
	}
	finalize(mockClient, testNamespace)
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

func TestSyncNamespaceThatIsTerminating(t *testing.T) {
	mockClient := &testclient.Fake{}
	now := util.Now()
	testNamespace := api.Namespace{
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
	err := syncNamespace(mockClient, testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	// TODO: Reuse the constants for all these strings from testclient
	expectedActionSet := util.NewStringSet(
		strings.Join([]string{"list", "replicationcontrollers", ""}, "-"),
		strings.Join([]string{"list", "services", ""}, "-"),
		strings.Join([]string{"list", "pods", ""}, "-"),
		strings.Join([]string{"list", "resourcequotas", ""}, "-"),
		strings.Join([]string{"list", "secrets", ""}, "-"),
		strings.Join([]string{"list", "limitranges", ""}, "-"),
		strings.Join([]string{"list", "events", ""}, "-"),
		strings.Join([]string{"create", "namespaces", "finalize"}, "-"),
		strings.Join([]string{"delete", "namespaces", ""}, "-"),
	)
	actionSet := util.NewStringSet()
	for _, action := range mockClient.Actions() {
		actionSet.Insert(strings.Join([]string{action.GetVerb(), action.GetResource(), action.GetSubresource()}, "-"))
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions: %v, but got: %v", expectedActionSet, actionSet)
	}
}

func TestSyncNamespaceThatIsActive(t *testing.T) {
	mockClient := &testclient.Fake{}
	testNamespace := api.Namespace{
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
	err := syncNamespace(mockClient, testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	if len(mockClient.Actions()) != 0 {
		t.Errorf("Expected no action from controller, but got: %v", mockClient.Actions())
	}
}

func TestRunStop(t *testing.T) {
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, api.RESTMapper)}
	nsController := NewNamespaceController(client, 1*time.Second)

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
