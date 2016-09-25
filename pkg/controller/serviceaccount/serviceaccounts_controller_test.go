/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/util/sets"
)

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func TestServiceAccountCreation(t *testing.T) {
	ns := api.NamespaceDefault

	defaultName := "default"
	managedName := "managed"

	activeNS := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: ns},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceActive,
		},
	}
	terminatingNS := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: ns},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}
	defaultServiceAccount := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:            defaultName,
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	managedServiceAccount := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:            managedName,
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	unmanagedServiceAccount := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:            "other-unmanaged",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}

	testcases := map[string]struct {
		ExistingNamespace       *api.Namespace
		ExistingServiceAccounts []*api.ServiceAccount

		AddedNamespace        *api.Namespace
		UpdatedNamespace      *api.Namespace
		DeletedServiceAccount *api.ServiceAccount

		ExpectCreatedServiceAccounts []string
	}{
		"new active namespace missing serviceaccounts": {
			ExistingServiceAccounts:      []*api.ServiceAccount{},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: sets.NewString(defaultName, managedName).List(),
		},
		"new active namespace missing serviceaccount": {
			ExistingServiceAccounts:      []*api.ServiceAccount{managedServiceAccount},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: []string{defaultName},
		},
		"new active namespace with serviceaccounts": {
			ExistingServiceAccounts:      []*api.ServiceAccount{defaultServiceAccount, managedServiceAccount},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"new terminating namespace": {
			ExistingServiceAccounts:      []*api.ServiceAccount{},
			AddedNamespace:               terminatingNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"updated active namespace missing serviceaccounts": {
			ExistingServiceAccounts:      []*api.ServiceAccount{},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: sets.NewString(defaultName, managedName).List(),
		},
		"updated active namespace missing serviceaccount": {
			ExistingServiceAccounts:      []*api.ServiceAccount{defaultServiceAccount},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: []string{managedName},
		},
		"updated active namespace with serviceaccounts": {
			ExistingServiceAccounts:      []*api.ServiceAccount{defaultServiceAccount, managedServiceAccount},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: []string{},
		},
		"updated terminating namespace": {
			ExistingServiceAccounts:      []*api.ServiceAccount{},
			UpdatedNamespace:             terminatingNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"deleted serviceaccount without namespace": {
			DeletedServiceAccount:        defaultServiceAccount,
			ExpectCreatedServiceAccounts: []string{},
		},
		"deleted serviceaccount with active namespace": {
			ExistingNamespace:            activeNS,
			DeletedServiceAccount:        defaultServiceAccount,
			ExpectCreatedServiceAccounts: []string{defaultName},
		},
		"deleted serviceaccount with terminating namespace": {
			ExistingNamespace:            terminatingNS,
			DeletedServiceAccount:        defaultServiceAccount,
			ExpectCreatedServiceAccounts: []string{},
		},
		"deleted unmanaged serviceaccount with active namespace": {
			ExistingNamespace:            activeNS,
			DeletedServiceAccount:        unmanagedServiceAccount,
			ExpectCreatedServiceAccounts: []string{},
		},
		"deleted unmanaged serviceaccount with terminating namespace": {
			ExistingNamespace:            terminatingNS,
			DeletedServiceAccount:        unmanagedServiceAccount,
			ExpectCreatedServiceAccounts: []string{},
		},
	}

	for k, tc := range testcases {
		client := fake.NewSimpleClientset(defaultServiceAccount, managedServiceAccount)
		options := DefaultServiceAccountsControllerOptions()
		options.ServiceAccounts = []api.ServiceAccount{
			{ObjectMeta: api.ObjectMeta{Name: defaultName}},
			{ObjectMeta: api.ObjectMeta{Name: managedName}},
		}
		controller := NewServiceAccountsController(client, options)

		if tc.ExistingNamespace != nil {
			controller.namespaces.Add(tc.ExistingNamespace)
		}
		for _, s := range tc.ExistingServiceAccounts {
			controller.serviceAccounts.Add(s)
		}

		if tc.AddedNamespace != nil {
			controller.namespaces.Add(tc.AddedNamespace)
			controller.namespaceAdded(tc.AddedNamespace)
		}
		if tc.UpdatedNamespace != nil {
			controller.namespaces.Add(tc.UpdatedNamespace)
			controller.namespaceUpdated(nil, tc.UpdatedNamespace)
		}
		if tc.DeletedServiceAccount != nil {
			controller.serviceAccountDeleted(tc.DeletedServiceAccount)
		}

		actions := client.Actions()
		if len(tc.ExpectCreatedServiceAccounts) != len(actions) {
			t.Errorf("%s: Expected to create accounts %#v. Actual actions were: %#v", k, tc.ExpectCreatedServiceAccounts, actions)
			continue
		}
		for i, expectedName := range tc.ExpectCreatedServiceAccounts {
			action := actions[i]
			if !action.Matches("create", "serviceaccounts") {
				t.Errorf("%s: Unexpected action %s", k, action)
				break
			}
			createdAccount := action.(core.CreateAction).GetObject().(*api.ServiceAccount)
			if createdAccount.Name != expectedName {
				t.Errorf("%s: Expected %s to be created, got %s", k, expectedName, createdAccount.Name)
			}
		}
	}
}
