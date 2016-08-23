/*
Copyright 2016 The Kubernetes Authors.

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

package testutil

import (
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// RegisterFakeWatch adds a new fake watcher for the specified resource in the given fake client.
// All subsequent requrest for watch on the client will result in returning this fake watcher.
func RegisterFakeWatch(resource string, client *core.Fake) *watch.FakeWatcher {
	watcher := watch.NewFake()
	client.AddWatchReactor(resource, func(action core.Action) (bool, watch.Interface, error) { return true, watcher, nil })
	return watcher
}

// RegisterFakeList registers a list response for the specified resource inside the given fake client.
// The passed value will be returned with every list call.
func RegisterFakeList(resource string, client *core.Fake, obj runtime.Object) {
	client.AddReactor("list", resource, func(action core.Action) (bool, runtime.Object, error) {
		return true, obj, nil
	})
}

// RegisterFakeCopyOnCreate register a reactor in the given fake client that passes
// all created object to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnCreate(resource string, client *core.Fake, watcher *watch.FakeWatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("create", resource, func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		obj := createAction.GetObject()
		go func() {
			watcher.Add(obj)
			objChan <- obj
		}()
		return true, obj, nil
	})
	return objChan
}

// RegisterFakeCopyOnCreate register a reactor in the given fake client that passes
// all updated object to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnUpdate(resource string, client *core.Fake, watcher *watch.FakeWatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		obj := updateAction.GetObject()
		go func() {
			watcher.Modify(obj)
			objChan <- obj
		}()
		return true, obj, nil
	})
	return objChan
}

// GetObjectFromChan tries to get an api object from the given channel
// within a reasonable time (1 min).
func GetObjectFromChan(c chan runtime.Object) runtime.Object {
	select {
	case obj := <-c:
		return obj
	case <-time.After(time.Minute):
		return nil
	}
}

func ToFederatedInformerForTestOnly(informer util.FederatedInformer) util.FederatedInformerForTestOnly {
	inter := informer.(interface{})
	return inter.(util.FederatedInformerForTestOnly)
}

// NewCluster build a new cluster object.
func NewCluster(name string, readyStatus api_v1.ConditionStatus) *federation_api.Cluster {
	return &federation_api.Cluster{
		ObjectMeta: api_v1.ObjectMeta{
			Name: name,
		},
		Status: federation_api.ClusterStatus{
			Conditions: []federation_api.ClusterCondition{
				{Type: federation_api.ClusterReady, Status: readyStatus},
			},
		},
	}
}
