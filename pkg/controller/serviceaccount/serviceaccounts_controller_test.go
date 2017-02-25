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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
)

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func TestServiceAccountCreation(t *testing.T) {
	ns := metav1.NamespaceDefault

	defaultName := "default"
	managedName := "managed"

	activeNS := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: ns},
		Status: v1.NamespaceStatus{
			Phase: v1.NamespaceActive,
		},
	}
	terminatingNS := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: ns},
		Status: v1.NamespaceStatus{
			Phase: v1.NamespaceTerminating,
		},
	}
	defaultServiceAccount := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            defaultName,
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	managedServiceAccount := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            managedName,
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	unmanagedServiceAccount := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "other-unmanaged",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}

	testcases := map[string]struct {
		ExistingNamespace       *v1.Namespace
		ExistingServiceAccounts []*v1.ServiceAccount

		AddedNamespace        *v1.Namespace
		UpdatedNamespace      *v1.Namespace
		DeletedServiceAccount *v1.ServiceAccount

		ExpectCreatedServiceAccounts []string
	}{
		"new active namespace missing serviceaccounts": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: sets.NewString(defaultName, managedName).List(),
		},
		"new active namespace missing serviceaccount": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{managedServiceAccount},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: []string{defaultName},
		},
		"new active namespace with serviceaccounts": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{defaultServiceAccount, managedServiceAccount},
			AddedNamespace:               activeNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"new terminating namespace": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{},
			AddedNamespace:               terminatingNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"updated active namespace missing serviceaccounts": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: sets.NewString(defaultName, managedName).List(),
		},
		"updated active namespace missing serviceaccount": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{defaultServiceAccount},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: []string{managedName},
		},
		"updated active namespace with serviceaccounts": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{defaultServiceAccount, managedServiceAccount},
			UpdatedNamespace:             activeNS,
			ExpectCreatedServiceAccounts: []string{},
		},
		"updated terminating namespace": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{},
			UpdatedNamespace:             terminatingNS,
			ExpectCreatedServiceAccounts: []string{},
		},

		"deleted serviceaccount without namespace": {
			DeletedServiceAccount:        defaultServiceAccount,
			ExpectCreatedServiceAccounts: []string{},
		},
		"deleted serviceaccount with active namespace": {
			ExistingServiceAccounts:      []*v1.ServiceAccount{managedServiceAccount},
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
			ExistingServiceAccounts:      []*v1.ServiceAccount{defaultServiceAccount, managedServiceAccount},
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
		informers := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc())
		options := DefaultServiceAccountsControllerOptions()
		options.ServiceAccounts = []v1.ServiceAccount{
			{ObjectMeta: metav1.ObjectMeta{Name: defaultName}},
			{ObjectMeta: metav1.ObjectMeta{Name: managedName}},
		}
		saInformer := informers.Core().V1().ServiceAccounts()
		nsInformer := informers.Core().V1().Namespaces()
		controller := NewServiceAccountsController(
			saInformer,
			nsInformer,
			client,
			options,
		)
		controller.saListerSynced = alwaysReady
		controller.nsListerSynced = alwaysReady

		saStore := saInformer.Informer().GetStore()
		nsStore := nsInformer.Informer().GetStore()

		syncCalls := make(chan struct{})
		controller.syncHandler = func(key string) error {
			err := controller.syncNamespace(key)
			if err != nil {
				t.Logf("%s: %v", k, err)
			}

			syncCalls <- struct{}{}
			return err
		}
		stopCh := make(chan struct{})
		defer close(stopCh)
		go controller.Run(1, stopCh)

		if tc.ExistingNamespace != nil {
			nsStore.Add(tc.ExistingNamespace)
		}
		for _, s := range tc.ExistingServiceAccounts {
			saStore.Add(s)
		}

		if tc.AddedNamespace != nil {
			nsStore.Add(tc.AddedNamespace)
			controller.namespaceAdded(tc.AddedNamespace)
		}
		if tc.UpdatedNamespace != nil {
			nsStore.Add(tc.UpdatedNamespace)
			controller.namespaceUpdated(nil, tc.UpdatedNamespace)
		}
		if tc.DeletedServiceAccount != nil {
			controller.serviceAccountDeleted(tc.DeletedServiceAccount)
		}

		// wait to be called
		select {
		case <-syncCalls:
		case <-time.After(10 * time.Second):
			t.Errorf("%s: took too long", k)
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
			createdAccount := action.(core.CreateAction).GetObject().(*v1.ServiceAccount)
			if createdAccount.Name != expectedName {
				t.Errorf("%s: Expected %s to be created, got %s", k, expectedName, createdAccount.Name)
			}
		}
	}
}

var alwaysReady = func() bool { return true }
