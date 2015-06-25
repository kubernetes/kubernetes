/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"io"
	"net"
	"net/url"
	"reflect"
	"sync"
	"syscall"
	"time"

	apierrs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// ListerWatcher is any object that knows how to perform an initial list and start a watch on a resource.
type ListerWatcher interface {
	// List should return a list type object; the Items field will be extracted, and the
	// ResourceVersion field will be used to start the watch in the right place.
	List() (runtime.Object, error)
	// Watch should begin a watch at the specified version.
	Watch(resourceVersion string) (watch.Interface, error)
}

// Reflector watches a specified resource and causes all changes to be reflected in the given store.
type Reflector struct {
	// The type of object we expect to place in the store.
	expectedType reflect.Type
	// The destination to sync up with the watch source
	store Store
	// listerWatcher is used to perform lists and watches.
	listerWatcher ListerWatcher
	// period controls timing between one watch ending and
	// the beginning of the next one.
	period       time.Duration
	resyncPeriod time.Duration
	// lastSyncResourceVersion is the resource version token last
	// observed when doing a sync with the underlying store
	// it is thread safe, but not synchronized with the underlying store
	lastSyncResourceVersion string
	// lastSyncResourceVersionMutex guards read/write access to lastSyncResourceVersion
	lastSyncResourceVersionMutex sync.RWMutex
}

// NewNamespaceKeyedIndexerAndReflector creates an Indexer and a Reflector
// The indexer is configured to key on namespace
func NewNamespaceKeyedIndexerAndReflector(lw ListerWatcher, expectedType interface{}, resyncPeriod time.Duration) (indexer Indexer, reflector *Reflector) {
	indexer = NewIndexer(MetaNamespaceKeyFunc, Indexers{"namespace": MetaNamespaceIndexFunc})
	reflector = NewReflector(lw, expectedType, indexer, resyncPeriod)
	return indexer, reflector
}

// NewReflector creates a new Reflector object which will keep the given store up to
// date with the server's contents for the given resource. Reflector promises to
// only put things in the store that have the type of expectedType.
// If resyncPeriod is non-zero, then lists will be executed after every resyncPeriod,
// so that you can use reflectors to periodically process everything as well as
// incrementally processing the things that change.
func NewReflector(lw ListerWatcher, expectedType interface{}, store Store, resyncPeriod time.Duration) *Reflector {
	r := &Reflector{
		listerWatcher: lw,
		store:         store,
		expectedType:  reflect.TypeOf(expectedType),
		period:        time.Second,
		resyncPeriod:  resyncPeriod,
	}
	return r
}

// Run starts a watch and handles watch events. Will restart the watch if it is closed.
// Run starts a goroutine and returns immediately.
func (r *Reflector) Run() {
	go util.Forever(func() { r.listAndWatch(util.NeverStop) }, r.period)
}

// RunUntil starts a watch and handles watch events. Will restart the watch if it is closed.
// RunUntil starts a goroutine and returns immediately. It will exit when stopCh is closed.
func (r *Reflector) RunUntil(stopCh <-chan struct{}) {
	go util.Until(func() { r.listAndWatch(stopCh) }, r.period, stopCh)
}

var (
	// nothing will ever be sent down this channel
	neverExitWatch <-chan time.Time = make(chan time.Time)

	// Used to indicate that watching stopped so that a resync could happen.
	errorResyncRequested = errors.New("resync channel fired")

	// Used to indicate that watching stopped because of a signal from the stop
	// channel passed in from a client of the reflector.
	errorStopRequested = errors.New("Stop requested")
)

// resyncChan returns a channel which will receive something when a resync is
// required, and a cleanup function.
func (r *Reflector) resyncChan() (<-chan time.Time, func() bool) {
	if r.resyncPeriod == 0 {
		return neverExitWatch, func() bool { return false }
	}
	// The cleanup function is required: imagine the scenario where watches
	// always fail so we end up listing frequently. Then, if we don't
	// manually stop the timer, we could end up with many timers active
	// concurrently.
	t := time.NewTimer(r.resyncPeriod)
	return t.C, t.Stop
}

func (r *Reflector) listAndWatch(stopCh <-chan struct{}) {
	var resourceVersion string
	resyncCh, cleanup := r.resyncChan()
	defer cleanup()

	list, err := r.listerWatcher.List()
	if err != nil {
		glog.Errorf("Failed to list %v: %v", r.expectedType, err)
		return
	}
	meta, err := meta.Accessor(list)
	if err != nil {
		glog.Errorf("Unable to understand list result %#v", list)
		return
	}
	resourceVersion = meta.ResourceVersion()
	items, err := runtime.ExtractList(list)
	if err != nil {
		glog.Errorf("Unable to understand list result %#v (%v)", list, err)
		return
	}
	if err := r.syncWith(items); err != nil {
		glog.Errorf("Unable to sync list result: %v", err)
		return
	}
	r.setLastSyncResourceVersion(resourceVersion)

	for {
		w, err := r.listerWatcher.Watch(resourceVersion)
		if err != nil {
			switch err {
			case io.EOF:
				// watch closed normally
			case io.ErrUnexpectedEOF:
				glog.V(1).Infof("Watch for %v closed with unexpected EOF: %v", r.expectedType, err)
			default:
				glog.Errorf("Failed to watch %v: %v", r.expectedType, err)
			}
			// If this is "connection refused" error, it means that most likely apiserver is not responsive.
			// It doesn't make sense to re-list all objects because most likely we will be able to restart
			// watch where we ended.
			// If that's the case wait and resend watch request.
			if urlError, ok := err.(*url.Error); ok {
				if opError, ok := urlError.Err.(*net.OpError); ok {
					if errno, ok := opError.Err.(syscall.Errno); ok && errno == syscall.ECONNREFUSED {
						time.Sleep(time.Second)
						continue
					}
				}
			}
			return
		}
		if err := r.watchHandler(w, &resourceVersion, resyncCh, stopCh); err != nil {
			if err != errorResyncRequested && err != errorStopRequested {
				glog.Errorf("watch of %v ended with: %v", r.expectedType, err)
			}
			return
		}
	}
}

// syncWith replaces the store's items with the given list.
func (r *Reflector) syncWith(items []runtime.Object) error {
	found := make([]interface{}, 0, len(items))
	for _, item := range items {
		found = append(found, item)
	}

	return r.store.Replace(found)
}

// watchHandler watches w and keeps *resourceVersion up to date.
func (r *Reflector) watchHandler(w watch.Interface, resourceVersion *string, resyncCh <-chan time.Time, stopCh <-chan struct{}) error {
	start := time.Now()
	eventCount := 0

	// Stopping the watcher should be idempotent and if we return from this function there's no way
	// we're coming back in with the same watch interface.
	defer w.Stop()

loop:
	for {
		select {
		case <-stopCh:
			return errorStopRequested
		case <-resyncCh:
			return errorResyncRequested
		case event, ok := <-w.ResultChan():
			if !ok {
				break loop
			}
			if event.Type == watch.Error {
				return apierrs.FromObject(event.Object)
			}
			if e, a := r.expectedType, reflect.TypeOf(event.Object); e != a {
				glog.Errorf("expected type %v, but watch event object had type %v", e, a)
				continue
			}
			meta, err := meta.Accessor(event.Object)
			if err != nil {
				glog.Errorf("unable to understand watch event %#v", event)
				continue
			}
			switch event.Type {
			case watch.Added:
				r.store.Add(event.Object)
			case watch.Modified:
				r.store.Update(event.Object)
			case watch.Deleted:
				// TODO: Will any consumers need access to the "last known
				// state", which is passed in event.Object? If so, may need
				// to change this.
				r.store.Delete(event.Object)
			default:
				glog.Errorf("unable to understand watch event %#v", event)
			}
			*resourceVersion = meta.ResourceVersion()
			r.setLastSyncResourceVersion(*resourceVersion)
			eventCount++
		}
	}

	watchDuration := time.Now().Sub(start)
	if watchDuration < 1*time.Second && eventCount == 0 {
		glog.V(4).Infof("Unexpected watch close - watch lasted less than a second and no items received")
		return errors.New("very short watch")
	}
	glog.V(4).Infof("Watch close - %v total %v items received", r.expectedType, eventCount)
	return nil
}

// LastSyncResourceVersion is the resource version observed when last sync with the underlying store
// The value returned is not synchronized with access to the underlying store and is not thread-safe
func (r *Reflector) LastSyncResourceVersion() string {
	r.lastSyncResourceVersionMutex.RLock()
	defer r.lastSyncResourceVersionMutex.RUnlock()
	return r.lastSyncResourceVersion
}

func (r *Reflector) setLastSyncResourceVersion(v string) {
	r.lastSyncResourceVersionMutex.Lock()
	defer r.lastSyncResourceVersionMutex.Unlock()
	r.lastSyncResourceVersion = v
}
