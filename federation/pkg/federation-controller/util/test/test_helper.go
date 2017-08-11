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
	"fmt"
	"os"
	"reflect"
	"runtime/pprof"
	"sync"
	"testing"
	"time"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	core "k8s.io/client-go/testing"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	finalizersutil "k8s.io/kubernetes/federation/pkg/federation-controller/util/finalizers"
	"k8s.io/kubernetes/pkg/api"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	pushTimeout = 5 * time.Second
)

// A structure that distributes events to multiple watchers.
type WatcherDispatcher struct {
	sync.Mutex
	watchers       []*watch.RaceFreeFakeWatcher
	eventsSoFar    []*watch.Event
	orderExecution chan func()
	stopChan       chan struct{}
}

func (wd *WatcherDispatcher) register(watcher *watch.RaceFreeFakeWatcher) {
	wd.Lock()
	defer wd.Unlock()
	wd.watchers = append(wd.watchers, watcher)
	for _, event := range wd.eventsSoFar {
		watcher.Action(event.Type, event.Object)
	}
}

func (wd *WatcherDispatcher) Stop() {
	wd.Lock()
	defer wd.Unlock()
	close(wd.stopChan)
	glog.Infof("Stopping WatcherDispatcher")
	for _, watcher := range wd.watchers {
		watcher.Stop()
	}
}

func copy(obj runtime.Object) runtime.Object {
	objCopy, err := api.Scheme.DeepCopy(obj)
	if err != nil {
		panic(err)
	}
	return objCopy.(runtime.Object)
}

// Add sends an add event.
func (wd *WatcherDispatcher) Add(obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Added, Object: copy(obj)})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Add(copy(obj))
		}
	}
}

// Modify sends a modify event.
func (wd *WatcherDispatcher) Modify(obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	glog.V(4).Infof("->WatcherDispatcher.Modify(%v)", obj)
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Modified, Object: copy(obj)})
	for i, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			glog.V(4).Infof("->Watcher(%d).Modify(%v)", i, obj)
			watcher.Modify(copy(obj))
		} else {
			glog.V(4).Infof("->Watcher(%d) is stopped.  Not calling Modify(%v)", i, obj)
		}
	}
}

// Delete sends a delete event.
func (wd *WatcherDispatcher) Delete(lastValue runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Deleted, Object: copy(lastValue)})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Delete(copy(lastValue))
		}
	}
}

// Error sends an Error event.
func (wd *WatcherDispatcher) Error(errValue runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: watch.Error, Object: copy(errValue)})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Error(copy(errValue))
		}
	}
}

// Action sends an event of the requested type, for table-based testing.
func (wd *WatcherDispatcher) Action(action watch.EventType, obj runtime.Object) {
	wd.Lock()
	defer wd.Unlock()
	wd.eventsSoFar = append(wd.eventsSoFar, &watch.Event{Type: action, Object: copy(obj)})
	for _, watcher := range wd.watchers {
		if !watcher.IsStopped() {
			watcher.Action(action, copy(obj))
		}
	}
}

// RegisterFakeWatch adds a new fake watcher for the specified resource in the given fake client.
// All subsequent requests for a watch on the client will result in returning this fake watcher.
func RegisterFakeWatch(resource string, client *core.Fake) *WatcherDispatcher {
	dispatcher := &WatcherDispatcher{
		watchers:       make([]*watch.RaceFreeFakeWatcher, 0),
		eventsSoFar:    make([]*watch.Event, 0),
		orderExecution: make(chan func(), 100),
		stopChan:       make(chan struct{}),
	}
	go func() {
		for {
			select {
			case fun := <-dispatcher.orderExecution:
				fun()
			case <-dispatcher.stopChan:
				return
			}
		}
	}()

	client.AddWatchReactor(resource, func(action core.Action) (bool, watch.Interface, error) {
		watcher := watch.NewRaceFreeFake()
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

// RegisterFakeClusterGet registers a get response for the cluster resource inside the given fake client.
func RegisterFakeClusterGet(client *core.Fake, obj runtime.Object) {
	clusterList, ok := obj.(*federationapi.ClusterList)
	client.AddReactor("get", "clusters", func(action core.Action) (bool, runtime.Object, error) {
		name := action.(core.GetAction).GetName()
		if ok {
			for _, cluster := range clusterList.Items {
				if cluster.Name == name {
					return true, &cluster, nil
				}
			}
		}
		return false, nil, fmt.Errorf("could not find the requested cluster: %s", name)
	})
}

// RegisterFakeOnCreate registers a reactor in the given fake client that passes
// all created objects to the given watcher.
func RegisterFakeOnCreate(resource string, client *core.Fake, watcher *WatcherDispatcher) {
	client.AddReactor("create", resource, func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		originalObj := createAction.GetObject()
		// Create a copy of the object here to prevent data races while reading the object in go routine.
		obj := copy(originalObj)
		watcher.orderExecution <- func() {
			glog.V(4).Infof("Object created: %v", obj)
			watcher.Add(obj)
		}
		return true, originalObj, nil
	})
}

// RegisterFakeCopyOnCreate registers a reactor in the given fake client that passes
// all created objects to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnCreate(resource string, client *core.Fake, watcher *WatcherDispatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("create", resource, func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		originalObj := createAction.GetObject()
		// Create a copy of the object here to prevent data races while reading the object in go routine.
		obj := copy(originalObj)
		watcher.orderExecution <- func() {
			glog.V(4).Infof("Object created. Writing to channel: %v", obj)
			watcher.Add(obj)
			objChan <- obj
		}
		return true, originalObj, nil
	})
	return objChan
}

// RegisterFakeOnUpdate registers a reactor in the given fake client that passes
// all updated objects to the given watcher.
func RegisterFakeOnUpdate(resource string, client *core.Fake, watcher *WatcherDispatcher) {
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		originalObj := updateAction.GetObject()
		glog.V(7).Infof("Updating %s: %v", resource, updateAction.GetObject())

		// Create a copy of the object here to prevent data races while reading the object in go routine.
		obj := copy(originalObj)
		operation := func() {
			glog.V(4).Infof("Object updated %v", obj)
			watcher.Modify(obj)
		}
		select {
		case watcher.orderExecution <- operation:
			break
		case <-time.After(pushTimeout):
			glog.Errorf("Fake client execution channel blocked")
			glog.Errorf("Tried to push %v", updateAction)
		}
		return true, originalObj, nil
	})
	return
}

// RegisterFakeCopyOnUpdate registers a reactor in the given fake client that passes
// all updated objects to the given watcher and also copies them to a channel for
// in-test inspection.
func RegisterFakeCopyOnUpdate(resource string, client *core.Fake, watcher *WatcherDispatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		originalObj := updateAction.GetObject()
		glog.V(7).Infof("Updating %s: %v", resource, updateAction.GetObject())

		// Create a copy of the object here to prevent data races while reading the object in go routine.
		obj := copy(originalObj)
		operation := func() {
			glog.V(4).Infof("Object updated. Writing to channel: %v", obj)
			watcher.Modify(obj)
			objChan <- obj
		}
		select {
		case watcher.orderExecution <- operation:
			break
		case <-time.After(pushTimeout):
			glog.Errorf("Fake client execution channel blocked")
			glog.Errorf("Tried to push %v", updateAction)
		}
		return true, originalObj, nil
	})
	return objChan
}

// RegisterFakeOnDelete registers a reactor in the given fake client that passes
// all deleted objects to the given watcher. Since we could get only name of the
// deleted object from DeleteAction, this register function relies on the getObject
// function passed to get the object by name and pass it watcher.
func RegisterFakeOnDelete(resource string, client *core.Fake, watcher *WatcherDispatcher, getObject func(name, namespace string) runtime.Object) {
	client.AddReactor("delete", resource, func(action core.Action) (bool, runtime.Object, error) {
		deleteAction := action.(core.DeleteAction)
		obj := getObject(deleteAction.GetName(), deleteAction.GetNamespace())
		glog.V(7).Infof("Deleting %s: %v", resource, obj)

		operation := func() {
			glog.V(4).Infof("Object deleted %v", obj)
			watcher.Delete(obj)
		}
		select {
		case watcher.orderExecution <- operation:
			break
		case <-time.After(pushTimeout):
			glog.Errorf("Fake client execution channel blocked")
			glog.Errorf("Tried to push %v", deleteAction)
		}
		return true, obj, nil
	})
	return
}

// Adds an update reactor to the given fake client.
// The reactor just returns the object passed to update action.
// This is used as a hack to workaround https://github.com/kubernetes/kubernetes/issues/40939.
// Without this, all update actions using fake client return empty objects.
func AddFakeUpdateReactor(resource string, client *core.Fake) {
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		originalObj := updateAction.GetObject()
		return true, originalObj, nil
	})
}

// GetObjectFromChan tries to get an api object from the given channel
// within a reasonable time.
func GetObjectFromChan(c chan runtime.Object) runtime.Object {
	select {
	case obj := <-c:
		return obj
	case <-time.After(wait.ForeverTestTimeout):
		pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
		return nil
	}
}

type CheckingFunction func(runtime.Object) error

// CheckObjectFromChan tries to get an object matching the given check function
// within a reasonable time.
func CheckObjectFromChan(c chan runtime.Object, checkFunction CheckingFunction) error {
	delay := 20 * time.Second
	var lastError error
	for {
		select {
		case obj := <-c:
			if lastError = checkFunction(obj); lastError == nil {
				return nil
			}
			glog.Infof("Check function failed with %v", lastError)
			delay = 5 * time.Second
		case <-time.After(delay):
			pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
			if lastError == nil {
				return fmt.Errorf("Failed to get an object from channel")
			} else {
				return lastError
			}
		}
	}
}

// CompareObjectMeta returns an error when the given objects are not equivalent.
func CompareObjectMeta(a, b metav1.ObjectMeta) error {
	if a.Namespace != b.Namespace {
		return fmt.Errorf("Different namespace expected:%s observed:%s", a.Namespace, b.Namespace)
	}
	if a.Name != b.Name {
		return fmt.Errorf("Different name expected:%s observed:%s", a.Name, b.Name)
	}
	if !reflect.DeepEqual(a.Labels, b.Labels) && (len(a.Labels) != 0 || len(b.Labels) != 0) {
		return fmt.Errorf("Labels are different expected:%v observed:%v", a.Labels, b.Labels)
	}
	if !reflect.DeepEqual(a.Annotations, b.Annotations) && (len(a.Annotations) != 0 || len(b.Annotations) != 0) {
		return fmt.Errorf("Annotations are different expected:%v observed:%v", a.Annotations, b.Annotations)
	}
	return nil
}

func ToFederatedInformerForTestOnly(informer util.FederatedInformer) util.FederatedInformerForTestOnly {
	inter := informer.(interface{})
	return inter.(util.FederatedInformerForTestOnly)
}

// NewCluster builds a new cluster object.
func NewCluster(name string, readyStatus apiv1.ConditionStatus) *federationapi.Cluster {
	return &federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
			Labels:      map[string]string{"cluster": name},
		},
		Status: federationapi.ClusterStatus{
			Conditions: []federationapi.ClusterCondition{
				{Type: federationapi.ClusterReady, Status: readyStatus},
			},
			Zones:  []string{"foozone"},
			Region: "fooregion",
		},
	}
}

// Ensure a key is in the store before returning (or timeout w/ error)
func WaitForStoreUpdate(store util.FederatedReadOnlyStore, clusterName, key string, timeout time.Duration) error {
	retryInterval := 100 * time.Millisecond
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		_, found, err := store.GetByKey(clusterName, key)
		return found, err
	})
	return err
}

// Ensure a key is in the store before returning (or timeout w/ error)
func WaitForStoreUpdateChecking(store util.FederatedReadOnlyStore, clusterName, key string, timeout time.Duration,
	checkFunction CheckingFunction) error {
	retryInterval := 500 * time.Millisecond
	var lastError error
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		item, found, err := store.GetByKey(clusterName, key)
		if err != nil || !found {
			return found, err
		}
		runtimeObj := item.(runtime.Object)
		lastError = checkFunction(runtimeObj)
		glog.V(2).Infof("Check function failed for %s %v %v", key, runtimeObj, lastError)
		return lastError == nil, nil
	})
	return err
}

func MetaAndSpecCheckingFunction(expected runtime.Object) CheckingFunction {
	return func(obj runtime.Object) error {
		if util.ObjectMetaAndSpecEquivalent(obj, expected) {
			return nil
		}
		return fmt.Errorf("Object different expected=%#v received=%#v", expected, obj)
	}
}

func AssertHasFinalizer(t *testing.T, obj runtime.Object, finalizer string) {
	hasFinalizer, err := finalizersutil.HasFinalizer(obj, finalizer)
	require.Nil(t, err)
	assert.True(t, hasFinalizer)
}

func NewInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
