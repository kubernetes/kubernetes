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

package framework

import (
	"errors"
	"math/rand"
	"strconv"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/watch"
)

func NewFakeControllerSource() *FakeControllerSource {
	return &FakeControllerSource{
		items:       map[nnu]runtime.Object{},
		broadcaster: watch.NewBroadcaster(100, watch.WaitIfChannelFull),
	}
}

// FakeControllerSource implements listing/watching for testing.
type FakeControllerSource struct {
	lock        sync.RWMutex
	items       map[nnu]runtime.Object
	changes     []watch.Event // one change per resourceVersion
	broadcaster *watch.Broadcaster
}

// namespace, name, uid to be used as a key.
type nnu struct {
	namespace, name string
	uid             types.UID
}

// Add adds an object to the set and sends an add event to watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) Add(obj runtime.Object) {
	f.Change(watch.Event{watch.Added, obj}, 1)
}

// Modify updates an object in the set and sends a modified event to watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) Modify(obj runtime.Object) {
	f.Change(watch.Event{watch.Modified, obj}, 1)
}

// Delete deletes an object from the set and sends a delete event to watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) Delete(lastValue runtime.Object) {
	f.Change(watch.Event{watch.Deleted, lastValue}, 1)
}

// AddDropWatch adds an object to the set but forgets to send an add event to
// watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) AddDropWatch(obj runtime.Object) {
	f.Change(watch.Event{watch.Added, obj}, 0)
}

// ModifyDropWatch updates an object in the set but forgets to send a modify
// event to watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) ModifyDropWatch(obj runtime.Object) {
	f.Change(watch.Event{watch.Modified, obj}, 0)
}

// DeleteDropWatch deletes an object from the set but forgets to send a delete
// event to watchers.
// obj's ResourceVersion is set.
func (f *FakeControllerSource) DeleteDropWatch(lastValue runtime.Object) {
	f.Change(watch.Event{watch.Deleted, lastValue}, 0)
}

func (f *FakeControllerSource) key(meta *api.ObjectMeta) nnu {
	return nnu{meta.Namespace, meta.Name, meta.UID}
}

// Change records the given event (setting the object's resource version) and
// sends a watch event with the specified probability.
func (f *FakeControllerSource) Change(e watch.Event, watchProbability float64) {
	f.lock.Lock()
	defer f.lock.Unlock()

	objMeta, err := api.ObjectMetaFor(e.Object)
	if err != nil {
		panic(err) // this is test code only
	}

	resourceVersion := len(f.changes)
	objMeta.ResourceVersion = strconv.Itoa(resourceVersion)
	f.changes = append(f.changes, e)
	key := f.key(objMeta)
	switch e.Type {
	case watch.Added, watch.Modified:
		f.items[key] = e.Object
	case watch.Deleted:
		delete(f.items, key)
	}

	if rand.Float64() < watchProbability {
		f.broadcaster.Action(e.Type, e.Object)
	}
}

// List returns a list object, with its resource version set.
func (f *FakeControllerSource) List() (runtime.Object, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	list := make([]runtime.Object, 0, len(f.items))
	for _, obj := range f.items {
		// Must make a copy to allow clients to modify the object.
		// Otherwise, if they make a change and write it back, they
		// will inadvertently change the our canonical copy (in
		// addition to racing with other clients).
		objCopy, err := api.Scheme.DeepCopy(obj)
		if err != nil {
			return nil, err
		}
		list = append(list, objCopy.(runtime.Object))
	}
	listObj := &api.List{}
	if err := runtime.SetList(listObj, list); err != nil {
		return nil, err
	}
	objMeta, err := api.ListMetaFor(listObj)
	if err != nil {
		return nil, err
	}
	resourceVersion := len(f.changes)
	objMeta.ResourceVersion = strconv.Itoa(resourceVersion)
	return listObj, nil
}

// Watch returns a watch, which will be pre-populated with all changes
// after resourceVersion.
func (f *FakeControllerSource) Watch(resourceVersion string) (watch.Interface, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	rc, err := strconv.Atoi(resourceVersion)
	if err != nil {
		return nil, err
	}
	rc++ // Don't re-send them a change they already have.
	if rc < len(f.changes) {
		changes := []watch.Event{}
		for _, c := range f.changes[rc:] {
			// Must make a copy to allow clients to modify the
			// object.  Otherwise, if they make a change and write
			// it back, they will inadvertently change the our
			// canonical copy (in addition to racing with other
			// clients).
			objCopy, err := api.Scheme.DeepCopy(c.Object)
			if err != nil {
				return nil, err
			}
			changes = append(changes, watch.Event{c.Type, objCopy.(runtime.Object)})
		}
		return f.broadcaster.WatchWithPrefix(changes), nil
	} else if rc > len(f.changes) {
		return nil, errors.New("resource version in the future not supported by this fake")
	}
	return f.broadcaster.Watch(), nil
}
