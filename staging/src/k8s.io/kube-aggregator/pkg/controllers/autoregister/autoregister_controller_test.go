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
	"reflect"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	core "k8s.io/client-go/testing"

	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kube-aggregator/pkg/client/informers/internalversion"
)

func alwaysReady() bool { return true }

func waitForNothing(startTime time.Time, client *fake.Clientset) (bool, error) {
	if len(client.Actions()) > 0 {
		return false, fmt.Errorf("unexpected action: %v", client.Actions())
	}
	if time.Now().After(startTime.Add(3 * time.Second)) {
		return true, nil
	}
	return false, nil
}

func waitForCreate(name string) func(startTime time.Time, client *fake.Clientset) (bool, error) {
	return func(startTime time.Time, client *fake.Clientset) (bool, error) {
		if len(client.Actions()) == 0 {
			return false, nil
		}
		if len(client.Actions()) > 1 {
			return false, fmt.Errorf("unexpected action: %v", client.Actions())
		}

		action := client.Actions()[0]
		createAction, ok := action.(core.CreateAction)
		if !ok {
			return false, fmt.Errorf("unexpected action: %v", client.Actions())
		}
		apiService := createAction.GetObject().(*apiregistration.APIService)
		if apiService.Name != name || apiService.Labels[AutoRegisterManagedLabel] != "true" {
			return false, fmt.Errorf("bad name or label %v", createAction)
		}

		return true, nil
	}
}

func waitForUpdate(name string) func(startTime time.Time, client *fake.Clientset) (bool, error) {
	return func(startTime time.Time, client *fake.Clientset) (bool, error) {
		if len(client.Actions()) == 0 {
			return false, nil
		}
		if len(client.Actions()) > 1 {
			return false, fmt.Errorf("unexpected action: %v", client.Actions())
		}

		action := client.Actions()[0]
		updateAction, ok := action.(core.UpdateAction)
		if !ok {
			return false, fmt.Errorf("unexpected action: %v", client.Actions())
		}
		apiService := updateAction.GetObject().(*apiregistration.APIService)
		if apiService.Name != name || apiService.Labels[AutoRegisterManagedLabel] != "true" || apiService.Spec.Group != "" {
			return false, fmt.Errorf("bad name, label, or group %v", updateAction)
		}

		return true, nil
	}
}

func waitForDelete(name string) func(startTime time.Time, client *fake.Clientset) (bool, error) {
	return func(startTime time.Time, client *fake.Clientset) (bool, error) {
		if len(client.Actions()) == 0 {
			return false, nil
		}

		// tolerate delete being called multiple times.  This happens if the delete fails on missing resource which
		// happens on an unsynced cache
		for _, action := range client.Actions() {
			deleteAction, ok := action.(core.DeleteAction)
			if !ok {
				return false, fmt.Errorf("unexpected action: %v", client.Actions())
			}
			if deleteAction.GetName() != name {
				return false, fmt.Errorf("bad name %v", deleteAction)
			}
		}

		return true, nil
	}
}

func TestCheckAPIService(t *testing.T) {
	tests := []struct {
		name string

		steps           []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface)
		expectedResults []func(startTime time.Time, client *fake.Clientset) (bool, error)
	}{
		{
			name: "do nothing",
			steps: []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface){
				// adding an API service which isn't auto-managed does nothing
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					fakeWatch.Add(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// removing an auto-sync that doesn't exist should do nothing
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.RemoveAPIServiceToSync("bar")
				},
			},
			expectedResults: []func(startTime time.Time, client *fake.Clientset) (bool, error){
				waitForNothing,
				waitForNothing,
			},
		},
		{
			name: "simple create and delete",
			steps: []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface){
				// adding one to auto-register should create
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.AddAPIServiceToSync(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// adding the same item again shouldn't do anything
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.AddAPIServiceToSync(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// removing entry should delete the API service since its managed
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.RemoveAPIServiceToSync("foo")
				},
			},
			expectedResults: []func(startTime time.Time, client *fake.Clientset) (bool, error){
				waitForCreate("foo"),
				waitForNothing,
				waitForDelete("foo"),
			},
		},
		{
			name: "create, user manage, then delete",
			steps: []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface){
				// adding one to auto-register should create
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.AddAPIServiceToSync(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// adding an API service to take ownership shouldn't cause the controller to do anything
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					fakeWatch.Modify(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// removing entry should NOT delete the API service since its user owned
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.RemoveAPIServiceToSync("foo")
				},
			},
			expectedResults: []func(startTime time.Time, client *fake.Clientset) (bool, error){
				waitForCreate("foo"),
				waitForNothing,
				waitForNothing,
			},
		},
		{
			name: "create managed apiservice without a matching request",
			steps: []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface){
				// adding an API service which isn't auto-managed does nothing
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					fakeWatch.Add(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// adding an API service which claims to be managed but isn't should be deleted
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					fakeWatch.Modify(&apiregistration.APIService{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "foo",
							Labels: map[string]string{AutoRegisterManagedLabel: "true"},
						}})
				},
			},
			expectedResults: []func(startTime time.Time, client *fake.Clientset) (bool, error){
				waitForNothing,
				waitForDelete("foo"),
			},
		},
		{
			name: "modifying it should result in stomping",
			steps: []func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface){
				// adding one to auto-register should create
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					c.AddAPIServiceToSync(&apiregistration.APIService{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
				},
				// updating a managed APIService should result in stomping it
				func(c AutoAPIServiceRegistration, fakeWatch *watch.FakeWatcher, client internalclientset.Interface) {
					fakeWatch.Modify(&apiregistration.APIService{
						ObjectMeta: metav1.ObjectMeta{
							Name:   "foo",
							Labels: map[string]string{AutoRegisterManagedLabel: "true"},
						},
						Spec: apiregistration.APIServiceSpec{
							Group: "something",
						},
					})
				},
			},
			expectedResults: []func(startTime time.Time, client *fake.Clientset) (bool, error){
				waitForCreate("foo"),
				waitForUpdate("foo"),
			},
		},
	}

NextTest:
	for _, test := range tests {
		client := fake.NewSimpleClientset()
		informerFactory := informers.NewSharedInformerFactory(client, 0)
		fakeWatch := watch.NewFake()
		client.PrependWatchReactor("apiservices", core.DefaultWatchReactor(fakeWatch, nil))

		c := NewAutoRegisterController(informerFactory.Apiregistration().InternalVersion().APIServices(), client.Apiregistration())

		stopCh := make(chan struct{})
		go informerFactory.Start(stopCh)
		go c.Run(3, stopCh)

		// wait for the initial sync to complete
		err := wait.PollImmediate(10*time.Millisecond, 10*time.Second, func() (bool, error) {
			return c.apiServiceSynced(), nil
		})
		if err != nil {
			t.Errorf("%s %v", test.name, err)
			close(stopCh)
			continue NextTest
		}

		for i, step := range test.steps {
			client.ClearActions()
			step(c, fakeWatch, client)

			startTime := time.Now()
			err := wait.PollImmediate(10*time.Millisecond, 20*time.Second, func() (bool, error) {
				return test.expectedResults[i](startTime, client)
			})
			if err != nil {
				t.Errorf("%s[%d] %v", test.name, i, err)
				close(stopCh)
				continue NextTest
			}

			// make sure that any create/update/delete is propagated to the watch
			for _, a := range client.Actions() {
				switch action := a.(type) {
				case core.CreateAction:
					fakeWatch.Add(action.GetObject())
					metadata, err := meta.Accessor(action.GetObject())
					if err != nil {
						t.Fatal(err)
					}
					err = wait.PollImmediate(10*time.Millisecond, 10*time.Second, func() (bool, error) {
						if _, err := c.apiServiceLister.Get(metadata.GetName()); err == nil {
							return true, nil
						}
						return false, nil
					})
					if err != nil {
						t.Errorf("%s[%d] %v", test.name, i, err)
						close(stopCh)
						continue NextTest
					}

				case core.DeleteAction:
					obj, err := c.apiServiceLister.Get(action.GetName())
					if apierrors.IsNotFound(err) {
						close(stopCh)
						continue NextTest
					}
					if err != nil {
						t.Fatal(err)
					}
					fakeWatch.Delete(obj)
					err = wait.PollImmediate(10*time.Millisecond, 10*time.Second, func() (bool, error) {
						if _, err := c.apiServiceLister.Get(action.GetName()); apierrors.IsNotFound(err) {
							return true, nil
						}
						return false, nil
					})
					if err != nil {
						t.Errorf("%s[%d] %v", test.name, i, err)
						close(stopCh)
						continue NextTest
					}

				case core.UpdateAction:
					fakeWatch.Modify(action.GetObject())
					metadata, err := meta.Accessor(action.GetObject())
					if err != nil {
						t.Fatal(err)
					}
					err = wait.PollImmediate(10*time.Millisecond, 10*time.Second, func() (bool, error) {
						obj, err := c.apiServiceLister.Get(metadata.GetName())
						if err != nil {
							return false, err
						}
						if reflect.DeepEqual(obj, action.GetObject()) {
							return true, nil
						}

						return false, nil
					})
					if err != nil {
						t.Errorf("%s[%d] %v", test.name, i, err)
						close(stopCh)
						continue NextTest
					}

				}
			}
		}

		close(stopCh)
	}
}
