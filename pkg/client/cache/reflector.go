/*
Copyright 2014 Google Inc. All rights reserved.

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

package cache

import (
	"reflect"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// Reflector watches a specified resource and causes all changes to be reflected in the given store.
type Reflector struct {
	// The type of object we expect to place in the store.
	expectedType reflect.Type
	// The destination to sync up with the watch source
	store Store
	// watchFactory is called to initiate watches.
	watchFactory WatchFactory
	// period controls timing between one watch ending and
	// the beginning of the next one.
	period time.Duration
}

// WatchFactory should begin a watch at the specified version.
type WatchFactory func(resourceVersion uint64) (watch.Interface, error)

// NewReflector makes a new Reflector object which will keep the given store up to
// date with the server's contents for the given resource. Reflector promises to
// only put things in the store that have the type of expectedType.
func NewReflector(watchFactory WatchFactory, expectedType interface{}, store Store) *Reflector {
	gc := &Reflector{
		watchFactory: watchFactory,
		store:        store,
		expectedType: reflect.TypeOf(expectedType),
		period:       time.Second,
	}
	return gc
}

// Run starts a watch and handles watch events. Will restart the watch if it is closed.
// Run starts a goroutine and returns immediately.
func (gc *Reflector) Run() {
	var resourceVersion uint64
	go util.Forever(func() {
		w, err := gc.watchFactory(resourceVersion)
		if err != nil {
			glog.Errorf("failed to watch %v: %v", gc.expectedType, err)
			return
		}
		gc.watchHandler(w, &resourceVersion)
	}, gc.period)
}

// watchHandler watches w and keeps *resourceVersion up to date.
func (gc *Reflector) watchHandler(w watch.Interface, resourceVersion *uint64) {
	for {
		event, ok := <-w.ResultChan()
		if !ok {
			glog.Errorf("unexpected watch close")
			return
		}
		if e, a := gc.expectedType, reflect.TypeOf(event.Object); e != a {
			glog.Errorf("expected type %v, but watch event object had type %v", e, a)
			continue
		}
		jsonBase, err := api.FindJSONBase(event.Object)
		if err != nil {
			glog.Errorf("unable to understand watch event %#v", event)
			continue
		}
		switch event.Type {
		case watch.Added:
			gc.store.Add(jsonBase.ID(), event.Object)
		case watch.Modified:
			gc.store.Update(jsonBase.ID(), event.Object)
		case watch.Deleted:
			// TODO: Will any consumers need access to the "last known
			// state", which is passed in event.Object? If so, may need
			// to change this.
			gc.store.Delete(jsonBase.ID())
		default:
			glog.Errorf("unable to understand watch event %#v", event)
		}
		*resourceVersion = jsonBase.ResourceVersion() + 1
	}
}
