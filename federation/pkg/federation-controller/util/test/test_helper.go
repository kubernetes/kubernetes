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
	"os"
	"runtime/pprof"
	"sync"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// A structure that distributes eventes to multiple watchers.
type WatcherDispatcher struct {
	sync.Mutex
	watchers    []*watch.FakeWatcher
	eventsSoFar []*watch.Event
}

func (wd *WatcherDispatcher) register(watcher *watch.FakeWatcher) {
	wd.Lock()
	defer wd.Unlock()
	wd.watchers = append(wd.watchers, watcher)
	for _, event := range wd.eventsSoFar {
		watcher.Action(event.Type, event.Object)
	}
}

// Add sends an add event.
func (wd *WatcherDispatcher) Add(obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Added, Object: obj})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Add(obj)
		}
	}
}

// Modify sends a modify event.
func (wd *WatcherDispatcher) Modify(obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Modified, Object: obj})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Modify(obj)
		}
	}
}

// Delete sends a delete event.
func (wd *WatcherDispatcher) Delete(lastValue runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Deleted, Object: lastValue})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Delete(lastValue)
		}
	}
}

// Error sends an Error event.
func (wd *WatcherDispatcher) Error(errValue runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Error, Object: errValue})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Error(errValue)
		}
	}
}

// Action sends an event of the requested type, for table-based testing.
func (wd *WatcherDispatcher) Action(action watch.EventType, obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: action, Object: obj})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Action(action, obj)
		}
	}
}

// RegisterFakeWatch adds a new fake watcher for the specified resource in the given fake client.
// All subsequent requests for a watch on the client will result in returning this fake watcher.
func RegisterFakeWatch(resource string, client *core.Fake) *WatcherDispatcher {
	dispatcher := &WatcherDispatcher{
		watchers:    make([]*watch.FakeWatcher, 0),
		eventsSoFar: make([]*watch.Event, 0),
	}

	client.AddWatchReactor(resource, func(action core.Action) (bool, watch.Interface, error) {
		watcher := watch.NewFakeWithChanSize(100)
		dispatcher.register(watcher)
		return true, watcher, nil
	})
	return dispatcher
}

// RegisterFakeList registers a list response for the specified resource inside the given fake client.
// The passed value will be returned with every list call.
func RegisterFakeList(resource string, client *core.Fake, obj runtime.Object) {
	client.AddReactor("list", resource, func(action core.Action) (bool, runtime.Object, error) {
		return true, obj, nil
	})
}

// RegisterFakeCopyOnCreate registers a reactor in the given fake client that passes
// all created objects to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnCreate(resource string, client *core.Fake, watcher *WatcherDispatcher) chan runtime.Object {
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

// RegisterFakeCopyOnCreate registers a reactor in the given fake client that passes
// all updated objects to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnUpdate(resource string, client *core.Fake, watcher *WatcherDispatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		obj := updateAction.GetObject()
		go func() {
			glog.V(4).Infof("Object updated. Writing to channel: %v", obj)
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
	case <-time.After(10 * time.Second):
		pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
		return nil
	}
}

func ToFederatedInformerForTestOnly(informer util.FederatedInformer) util.FederatedInformerForTestOnly {
	inter := informer.(interface{})
	return inter.(util.FederatedInformerForTestOnly)
}

// NewCluster builds a new cluster object.
func NewCluster(name string, readyStatus api_v1.ConditionStatus) *federation_api.Cluster {
	return &federation_api.Cluster{
		ObjectMeta: api_v1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
		Status: federation_api.ClusterStatus{
			Conditions: []federation_api.ClusterCondition{
				{Type: federation_api.ClusterReady, Status: readyStatus},
			},
		},
	}
}
