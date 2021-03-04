/*
Copyright 2017 The Kubernetes Authors.

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

package autoregister

import (
	"fmt"
	"sync"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/fake"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1"
)

func newAutoRegisterManagedAPIService(name string) *apiregistrationv1.APIService {
	return &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{AutoRegisterManagedLabel: string("true")}},
	}
}

func newAutoRegisterManagedOnStartAPIService(name string) *apiregistrationv1.APIService {
	return &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{AutoRegisterManagedLabel: string("onstart")}},
	}
}

func newAutoRegisterManagedModifiedAPIService(name string) *apiregistrationv1.APIService {
	return &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{AutoRegisterManagedLabel: string("true")}},
		Spec: apiregistrationv1.APIServiceSpec{
			Group: "something",
		},
	}
}

func newAPIService(name string) *apiregistrationv1.APIService {
	return &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func checkForNothing(name string, client *fake.Clientset) error {
	if len(client.Actions()) > 0 {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}

	return nil
}

func checkForCreate(name string, client *fake.Clientset) error {
	if len(client.Actions()) == 0 {
		return nil
	}
	if len(client.Actions()) > 1 {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}

	action := client.Actions()[0]

	createAction, ok := action.(clienttesting.CreateAction)
	if !ok {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}
	apiService := createAction.GetObject().(*apiregistrationv1.APIService)
	if apiService.Name != name || apiService.Labels[AutoRegisterManagedLabel] != "true" {
		return fmt.Errorf("bad name or label %v", createAction)
	}

	return nil
}

func checkForCreateOnStart(name string, client *fake.Clientset) error {
	if len(client.Actions()) == 0 {
		return nil
	}
	if len(client.Actions()) > 1 {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}

	action := client.Actions()[0]

	createAction, ok := action.(clienttesting.CreateAction)
	if !ok {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}
	apiService := createAction.GetObject().(*apiregistrationv1.APIService)
	if apiService.Name != name || apiService.Labels[AutoRegisterManagedLabel] != "onstart" {
		return fmt.Errorf("bad name or label %v", createAction)
	}

	return nil
}

func checkForUpdate(name string, client *fake.Clientset) error {
	if len(client.Actions()) == 0 {
		return nil
	}
	if len(client.Actions()) > 1 {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}

	action := client.Actions()[0]
	updateAction, ok := action.(clienttesting.UpdateAction)
	if !ok {
		return fmt.Errorf("unexpected action: %v", client.Actions())
	}
	apiService := updateAction.GetObject().(*apiregistrationv1.APIService)
	if apiService.Name != name || apiService.Labels[AutoRegisterManagedLabel] != "true" || apiService.Spec.Group != "" {
		return fmt.Errorf("bad name, label, or group %v", updateAction)
	}

	return nil
}

func checkForDelete(name string, client *fake.Clientset) error {
	if len(client.Actions()) == 0 {
		return nil
	}

	for _, action := range client.Actions() {
		deleteAction, ok := action.(clienttesting.DeleteAction)
		if !ok {
			return fmt.Errorf("unexpected action: %v", client.Actions())
		}
		if deleteAction.GetName() != name {
			return fmt.Errorf("bad name %v", deleteAction)
		}
	}

	return nil
}

func TestSync(t *testing.T) {
	tests := []struct {
		name                      string
		apiServiceName            string
		addAPIServices            []*apiregistrationv1.APIService
		updateAPIServices         []*apiregistrationv1.APIService
		addSyncAPIServices        []*apiregistrationv1.APIService
		addSyncOnStartAPIServices []*apiregistrationv1.APIService
		delSyncAPIServices        []string
		alreadySynced             map[string]bool
		presentAtStart            map[string]bool
		expectedResults           func(name string, client *fake.Clientset) error
	}{
		{
			name:               "adding an API service which isn't auto-managed does nothing",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{},
			expectedResults:    checkForNothing,
		},
		{
			name:               "adding one to auto-register should create",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{},
			updateAPIServices:  []*apiregistrationv1.APIService{},
			addSyncAPIServices: []*apiregistrationv1.APIService{newAPIService("foo")},
			delSyncAPIServices: []string{},
			expectedResults:    checkForCreate,
		},
		{
			name:               "duplicate AddAPIServiceToSync don't panic",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAutoRegisterManagedAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{},
			addSyncAPIServices: []*apiregistrationv1.APIService{newAutoRegisterManagedAPIService("foo"), newAutoRegisterManagedAPIService("foo")},
			delSyncAPIServices: []string{},
			expectedResults:    checkForNothing,
		},
		{
			name:               "duplicate RemoveAPIServiceToSync don't panic",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAutoRegisterManagedAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{"foo", "foo"},
			expectedResults:    checkForDelete,
		},
		{
			name:               "removing auto-managed then RemoveAPIService should not touch APIService",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{},
			updateAPIServices:  []*apiregistrationv1.APIService{newAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{"foo"},
			expectedResults:    checkForNothing,
		},
		{
			name:               "create managed apiservice without a matching request",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{newAutoRegisterManagedAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{},
			expectedResults:    checkForDelete,
		},
		{
			name:               "modifying it should result in stomping",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{},
			updateAPIServices:  []*apiregistrationv1.APIService{newAutoRegisterManagedModifiedAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{newAutoRegisterManagedAPIService("foo")},
			delSyncAPIServices: []string{},
			expectedResults:    checkForUpdate,
		},

		{
			name:                      "adding one to auto-register on start should create",
			apiServiceName:            "foo",
			addAPIServices:            []*apiregistrationv1.APIService{},
			updateAPIServices:         []*apiregistrationv1.APIService{},
			addSyncOnStartAPIServices: []*apiregistrationv1.APIService{newAPIService("foo")},
			delSyncAPIServices:        []string{},
			expectedResults:           checkForCreateOnStart,
		},
		{
			name:                      "adding one to auto-register on start already synced should do nothing",
			apiServiceName:            "foo",
			addAPIServices:            []*apiregistrationv1.APIService{},
			updateAPIServices:         []*apiregistrationv1.APIService{},
			addSyncOnStartAPIServices: []*apiregistrationv1.APIService{newAPIService("foo")},
			delSyncAPIServices:        []string{},
			alreadySynced:             map[string]bool{"foo": true},
			expectedResults:           checkForNothing,
		},
		{
			name:               "managed onstart apiservice present at start without a matching request should delete",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{newAutoRegisterManagedOnStartAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{},
			presentAtStart:     map[string]bool{"foo": true},
			alreadySynced:      map[string]bool{},
			expectedResults:    checkForDelete,
		},
		{
			name:               "managed onstart apiservice present at start without a matching request already synced once should no-op",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{newAutoRegisterManagedOnStartAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{},
			presentAtStart:     map[string]bool{"foo": true},
			alreadySynced:      map[string]bool{"foo": true},
			expectedResults:    checkForNothing,
		},
		{
			name:               "managed onstart apiservice not present at start without a matching request should no-op",
			apiServiceName:     "foo",
			addAPIServices:     []*apiregistrationv1.APIService{newAPIService("foo")},
			updateAPIServices:  []*apiregistrationv1.APIService{newAutoRegisterManagedOnStartAPIService("foo")},
			addSyncAPIServices: []*apiregistrationv1.APIService{},
			delSyncAPIServices: []string{},
			presentAtStart:     map[string]bool{},
			alreadySynced:      map[string]bool{},
			expectedResults:    checkForNothing,
		},
		{
			name:                      "modifying onstart it should result in stomping",
			apiServiceName:            "foo",
			addAPIServices:            []*apiregistrationv1.APIService{},
			updateAPIServices:         []*apiregistrationv1.APIService{newAutoRegisterManagedModifiedAPIService("foo")},
			addSyncOnStartAPIServices: []*apiregistrationv1.APIService{newAutoRegisterManagedOnStartAPIService("foo")},
			delSyncAPIServices:        []string{},
			expectedResults:           checkForUpdate,
		},
		{
			name:                      "modifying onstart already synced should no-op",
			apiServiceName:            "foo",
			addAPIServices:            []*apiregistrationv1.APIService{},
			updateAPIServices:         []*apiregistrationv1.APIService{newAutoRegisterManagedModifiedAPIService("foo")},
			addSyncOnStartAPIServices: []*apiregistrationv1.APIService{newAutoRegisterManagedOnStartAPIService("foo")},
			delSyncAPIServices:        []string{},
			alreadySynced:             map[string]bool{"foo": true},
			expectedResults:           checkForNothing,
		},
	}

	for _, test := range tests {
		fakeClient := fake.NewSimpleClientset()
		apiServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})

		alreadySynced := map[string]bool{}
		for k, v := range test.alreadySynced {
			alreadySynced[k] = v
		}

		presentAtStart := map[string]bool{}
		for k, v := range test.presentAtStart {
			presentAtStart[k] = v
		}

		c := &autoRegisterController{
			apiServiceClient:  fakeClient.ApiregistrationV1(),
			apiServiceLister:  listers.NewAPIServiceLister(apiServiceIndexer),
			apiServicesToSync: map[string]*apiregistrationv1.APIService{},
			queue:             workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "autoregister"),

			syncedSuccessfullyLock: &sync.RWMutex{},
			syncedSuccessfully:     alreadySynced,

			apiServicesAtStart: presentAtStart,
		}

		for _, obj := range test.addAPIServices {
			apiServiceIndexer.Add(obj)
		}

		for _, obj := range test.updateAPIServices {
			apiServiceIndexer.Update(obj)
		}

		for _, obj := range test.addSyncAPIServices {
			c.AddAPIServiceToSync(obj)
		}

		for _, obj := range test.addSyncOnStartAPIServices {
			c.AddAPIServiceToSyncOnStart(obj)
		}

		for _, objName := range test.delSyncAPIServices {
			c.RemoveAPIServiceToSync(objName)
		}

		c.checkAPIService(test.apiServiceName)

		//compare the expected results
		err := test.expectedResults(test.apiServiceName, fakeClient)
		if err != nil {
			t.Errorf("%s %v", test.name, err)
		}
	}
}
