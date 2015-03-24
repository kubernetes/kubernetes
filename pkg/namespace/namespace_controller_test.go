/*
Copyright 2015 Google Inc. All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	mockClient := &client.Fake{}
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
	if len(mockClient.Actions) != 1 {
		t.Errorf("Expected 1 mock client action, but got %v", len(mockClient.Actions))
	}
	if mockClient.Actions[0].Action != "finalize-namespace" {
		t.Errorf("Expected finalize-namespace action %v", mockClient.Actions[0].Action)
	}
}

func TestSyncNamespaceThatIsTerminating(t *testing.T) {
	mockClient := &client.Fake{}
	nm := NamespaceManager{kubeClient: mockClient, store: cache.NewStore(cache.MetaNamespaceKeyFunc)}
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
	err := nm.syncNamespace(testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	expectedActionSet := util.NewStringSet(
		"list-services",
		"list-pods",
		"list-resourceQuotas",
		"list-controllers",
		"list-secrets",
		"list-limitRanges",
		"list-events",
		"finalize-namespace",
		"delete-namespace")
	actionSet := util.NewStringSet()
	for i := range mockClient.Actions {
		actionSet.Insert(mockClient.Actions[i].Action)
	}
	if !actionSet.HasAll(expectedActionSet.List()...) {
		t.Errorf("Expected actions: %v, but got: %v", expectedActionSet, actionSet)
	}
}

func TestSyncNamespaceThatIsActive(t *testing.T) {
	mockClient := &client.Fake{}
	nm := NamespaceManager{kubeClient: mockClient, store: cache.NewStore(cache.MetaNamespaceKeyFunc)}
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
	err := nm.syncNamespace(testNamespace)
	if err != nil {
		t.Errorf("Unexpected error when synching namespace %v", err)
	}
	actionSet := util.NewStringSet()
	for i := range mockClient.Actions {
		actionSet.Insert(mockClient.Actions[i].Action)
	}
	if len(actionSet) != 0 {
		t.Errorf("Expected no action from controller, but got: %v", actionSet)
	}
}
