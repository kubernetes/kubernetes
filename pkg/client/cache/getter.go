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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// Store is a generic object storage interface. Getter knows how to watch a server
// and update a store. A generic store is provided, which allows Getter to be used
// as a local caching system, and an LRU store, which allows Getter to work like a
// queue of items yet to be processed.
type Store interface {
	Add(ID string, obj interface{})
	Update(ID string, obj interface{})
	Delete(ID string, obj interface{})
	List() []interface{}
	Get(ID string) (item interface{}, exists bool)
}

// Getter watches a specified resource and causes all changes to be reflected in the given store.
type Getter struct {
	kubeClient   *client.Client
	resource     string
	expectedType reflect.Type
	store        Store
}

// NewGetter makes a new Getter object which will keep the given store up to
// date with the server's contents for the given resource. Getter promises to
// only put things in the store that have the type of expectedType.
// TODO: define a query so you only locally cache a subset of items.
func NewGetter(resource string, kubeClient *client.Client, expectedType interface{}, store Store) *Getter {
	gc := &Getter{
		resource:     resource,
		kubeClient:   kubeClient,
		store:        store,
		expectedType: reflect.TypeOf(expectedType),
	}
	return gc
}

func (gc *Getter) Run() {
	go util.Forever(gc.watch, 5*time.Second)
}

func (gc *Getter) watch() {
	w, err := gc.kubeClient.Get().Path(gc.resource).Watch()
	if err != nil {
		glog.Errorf("failed to watch %v: %v", gc.resource, err)
		return
	}
	gc.watchHandler(w)
}

func (gc *Getter) watchHandler(w watch.Interface) {
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
			gc.store.Delete(jsonBase.ID(), event.Object)
		default:
			glog.Errorf("unable to understand watch event %#v", event)
		}
	}
}
