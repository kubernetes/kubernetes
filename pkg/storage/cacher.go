/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"net/http"
	"reflect"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// CacherConfig contains the configuration for a given Cache.
type CacherConfig struct {
	// Maximum size of the history cached in memory.
	CacheCapacity int

	// An underlying storage.Interface.
	Storage Interface

	// An underlying storage.Versioner.
	Versioner Versioner

	// The Cache will be caching objects of a given Type and assumes that they
	// are all stored under ResourcePrefix directory in the underlying database.
	Type           interface{}
	ResourcePrefix string

	// KeyFunc is used to get a key in the underyling storage for a given object.
	KeyFunc func(runtime.Object) (string, error)

	// TriggerPublisherFunc is used for optimizing amount of watchers that
	// needs to process an incoming event.
	TriggerPublisherFunc TriggerPublisherFunc

	// NewList is a function that creates new empty object storing a list of
	// objects of type Type.
	NewListFunc func() runtime.Object

	Codec runtime.Codec
}

type watchersMap map[int]*cacheWatcher

func (wm watchersMap) addWatcher(w *cacheWatcher, number int) {
	wm[number] = w
}

func (wm watchersMap) deleteWatcher(number int) {
	delete(wm, number)
}

func (wm watchersMap) terminateAll() {
	for key, watcher := range wm {
		delete(wm, key)
		watcher.stop()
	}
}

type indexedWatchers struct {
	allWatchers   watchersMap
	valueWatchers map[string]watchersMap
}

func (i *indexedWatchers) addWatcher(w *cacheWatcher, number int, value string, supported bool) {
	if supported {
		if _, ok := i.valueWatchers[value]; !ok {
			i.valueWatchers[value] = watchersMap{}
		}
		i.valueWatchers[value].addWatcher(w, number)
	} else {
		i.allWatchers.addWatcher(w, number)
	}
}

func (i *indexedWatchers) deleteWatcher(number int, value string, supported bool) {
	if supported {
		i.valueWatchers[value].deleteWatcher(number)
		if len(i.valueWatchers[value]) == 0 {
			delete(i.valueWatchers, value)
		}
	} else {
		i.allWatchers.deleteWatcher(number)
	}
}

func (i *indexedWatchers) terminateAll() {
	i.allWatchers.terminateAll()
	for index, watchers := range i.valueWatchers {
		watchers.terminateAll()
		delete(i.valueWatchers, index)
	}
}

// Cacher is responsible for serving WATCH and LIST requests for a given
// resource from its internal cache and updating its cache in the background
// based on the underlying storage contents.
// Cacher implements storage.Interface (although most of the calls are just
// delegated to the underlying storage).
type Cacher struct {
	sync.RWMutex

	// Before accessing the cacher's cache, wait for the ready to be ok.
	// This is necessary to prevent users from accessing structures that are
	// uninitialized or are being repopulated right now.
	// ready needs to be set to false when the cacher is paused or stopped.
	// ready needs to be set to true when the cacher is ready to use after
	// initialization.
	ready *ready

	// Underlying storage.Interface.
	storage Interface

	// Expected type of objects in the underlying cache.
	objectType reflect.Type

	// "sliding window" of recent changes of objects and the current state.
	watchCache *watchCache
	reflector  *cache.Reflector

	// Versioner is used to handle resource versions.
	versioner Versioner

	// keyFunc is used to get a key in the underyling storage for a given object.
	keyFunc func(runtime.Object) (string, error)

	// triggerFunc is used for optimizing amount of watchers that needs to process
	// an incoming event.
	triggerFunc TriggerPublisherFunc
	// watchers is mapping from the value of trigger function that a
	// watcher is interested into the watchers
	watcherIdx int
	watchers   indexedWatchers

	// Incoming events that should be dispatched to watchers.
	incoming chan watchCacheEvent

	// Handling graceful termination.
	stopLock sync.RWMutex
	stopped  bool
	stopCh   chan struct{}
	stopWg   sync.WaitGroup
}

// Create a new Cacher responsible from service WATCH and LIST requests from its
// internal cache and updating its cache in the background based on the given
// configuration.
func NewCacherFromConfig(config CacherConfig) *Cacher {
	watchCache := newWatchCache(config.CacheCapacity)
	listerWatcher := newCacherListerWatcher(config.Storage, config.ResourcePrefix, config.NewListFunc)

	// Give this error when it is constructed rather than when you get the
	// first watch item, because it's much easier to track down that way.
	if obj, ok := config.Type.(runtime.Object); ok {
		if err := runtime.CheckCodec(config.Codec, obj); err != nil {
			panic("storage codec doesn't seem to match given type: " + err.Error())
		}
	}

	cacher := &Cacher{
		ready:       newReady(),
		storage:     config.Storage,
		objectType:  reflect.TypeOf(config.Type),
		watchCache:  watchCache,
		reflector:   cache.NewReflector(listerWatcher, config.Type, watchCache, 0),
		versioner:   config.Versioner,
		keyFunc:     config.KeyFunc,
		triggerFunc: config.TriggerPublisherFunc,
		watcherIdx:  0,
		watchers: indexedWatchers{
			allWatchers:   make(map[int]*cacheWatcher),
			valueWatchers: make(map[string]watchersMap),
		},
		// TODO: Figure out the correct value for the buffer size.
		incoming: make(chan watchCacheEvent, 100),
		// We need to (potentially) stop both:
		// - wait.Until go-routine
		// - reflector.ListAndWatch
		// and there are no guarantees on the order that they will stop.
		// So we will be simply closing the channel, and synchronizing on the WaitGroup.
		stopCh: make(chan struct{}),
	}
	watchCache.SetOnEvent(cacher.processEvent)
	go cacher.dispatchEvents()

	stopCh := cacher.stopCh
	cacher.stopWg.Add(1)
	go func() {
		defer cacher.stopWg.Done()
		wait.Until(
			func() {
				if !cacher.isStopped() {
					cacher.startCaching(stopCh)
				}
			}, time.Second, stopCh,
		)
	}()
	return cacher
}

func (c *Cacher) startCaching(stopChannel <-chan struct{}) {
	// The 'usable' lock is always 'RLock'able when it is safe to use the cache.
	// It is safe to use the cache after a successful list until a disconnection.
	// We start with usable (write) locked. The below OnReplace function will
	// unlock it after a successful list. The below defer will then re-lock
	// it when this function exits (always due to disconnection), only if
	// we actually got a successful list. This cycle will repeat as needed.
	successfulList := false
	c.watchCache.SetOnReplace(func() {
		successfulList = true
		c.ready.set(true)
	})
	defer func() {
		if successfulList {
			c.ready.set(false)
		}
	}()

	c.terminateAllWatchers()
	// Note that since onReplace may be not called due to errors, we explicitly
	// need to retry it on errors under lock.
	// Also note that startCaching is called in a loop, so there's no need
	// to have another loop here.
	if err := c.reflector.ListAndWatch(stopChannel); err != nil {
		glog.Errorf("unexpected ListAndWatch error: %v", err)
	}
}

// Implements storage.Interface.
func (c *Cacher) Versioner() Versioner {
	return c.storage.Versioner()
}

// Implements storage.Interface.
func (c *Cacher) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	return c.storage.Create(ctx, key, obj, out, ttl)
}

// Implements storage.Interface.
func (c *Cacher) Delete(ctx context.Context, key string, out runtime.Object, preconditions *Preconditions) error {
	return c.storage.Delete(ctx, key, out, preconditions)
}

// Implements storage.Interface.
func (c *Cacher) Watch(ctx context.Context, key string, resourceVersion string, filter Filter) (watch.Interface, error) {
	watchRV, err := ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	c.ready.wait()

	// We explicitly use thread unsafe version and do locking ourself to ensure that
	// no new events will be processed in the meantime. The watchCache will be unlocked
	// on return from this function.
	// Note that we cannot do it under Cacher lock, to avoid a deadlock, since the
	// underlying watchCache is calling processEvent under its lock.
	c.watchCache.RLock()
	defer c.watchCache.RUnlock()
	initEvents, err := c.watchCache.GetAllEventsSinceThreadUnsafe(watchRV)
	if err != nil {
		// To match the uncached watch implementation, once we have passed authn/authz/admission,
		// and successfully parsed a resource version, other errors must fail with a watch event of type ERROR,
		// rather than a directly returned error.
		return newErrWatcher(err), nil
	}

	triggerValue, triggerSupported := "", false
	// TODO: Currently we assume that in a given Cacher object, any <filter> that is
	// passed here is aware of exactly the same trigger (at most one).
	// Thus, either 0 or 1 values will be returned.
	if matchValues := filter.Trigger(); len(matchValues) > 0 {
		triggerValue, triggerSupported = matchValues[0].Value, true
	}

	c.Lock()
	defer c.Unlock()
	forget := forgetWatcher(c, c.watcherIdx, triggerValue, triggerSupported)
	watcher := newCacheWatcher(watchRV, initEvents, filterFunction(key, c.keyFunc, filter), forget)

	c.watchers.addWatcher(watcher, c.watcherIdx, triggerValue, triggerSupported)
	c.watcherIdx++
	return watcher, nil
}

// Implements storage.Interface.
func (c *Cacher) WatchList(ctx context.Context, key string, resourceVersion string, filter Filter) (watch.Interface, error) {
	return c.Watch(ctx, key, resourceVersion, filter)
}

// Implements storage.Interface.
func (c *Cacher) Get(ctx context.Context, key string, objPtr runtime.Object, ignoreNotFound bool) error {
	return c.storage.Get(ctx, key, objPtr, ignoreNotFound)
}

// Implements storage.Interface.
func (c *Cacher) GetToList(ctx context.Context, key string, filter Filter, listObj runtime.Object) error {
	return c.storage.GetToList(ctx, key, filter, listObj)
}

// Implements storage.Interface.
func (c *Cacher) List(ctx context.Context, key string, resourceVersion string, filter Filter, listObj runtime.Object) error {
	if resourceVersion == "" {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility).
		return c.storage.List(ctx, key, resourceVersion, filter, listObj)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.

	listRV, err := ParseListResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	c.ready.wait()

	// List elements from cache, with at least 'listRV'.
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}
	filterFunc := filterFunction(key, c.keyFunc, filter)

	objs, readResourceVersion, err := c.watchCache.WaitUntilFreshAndList(listRV)
	if err != nil {
		return fmt.Errorf("failed to wait for fresh list: %v", err)
	}
	for _, obj := range objs {
		object, ok := obj.(runtime.Object)
		if !ok {
			return fmt.Errorf("non runtime.Object returned from storage: %v", obj)
		}
		if filterFunc.Filter(object) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(object).Elem()))
		}
	}
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, readResourceVersion); err != nil {
			return err
		}
	}
	return nil
}

// Implements storage.Interface.
func (c *Cacher) GuaranteedUpdate(ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool, preconditions *Preconditions, tryUpdate UpdateFunc) error {
	return c.storage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate)
}

func (c *Cacher) triggerValues(event *watchCacheEvent) ([]string, bool) {
	// TODO: Currently we assume that in a given Cacher object, its <c.triggerFunc>
	// is aware of exactly the same trigger (at most one). Thus calling:
	//   c.triggerFunc(<some object>)
	// can return only 0 or 1 values.
	// That means, that triggerValues itself may return up to 2 different values.
	if c.triggerFunc == nil {
		return nil, false
	}
	result := make([]string, 0, 2)
	matchValues := c.triggerFunc(event.Object)
	if len(matchValues) > 0 {
		result = append(result, matchValues[0].Value)
	}
	if event.PrevObject == nil {
		return result, len(result) > 0
	}
	prevMatchValues := c.triggerFunc(event.PrevObject)
	if len(prevMatchValues) > 0 {
		if len(result) == 0 || result[0] != prevMatchValues[0].Value {
			result = append(result, prevMatchValues[0].Value)
		}
	}
	return result, len(result) > 0
}

// TODO: Most probably splitting this method to a separate thread will visibily
// improve throughput of our watch machinery. So what we should do is to:
// - OnEvent handler simply put an element to channel
// - processEvent be another goroutine processing events from that channel
// Additionally, if we make this channel buffered, cacher will be more resistant
// to single watchers being slow - see cacheWatcher::add method.
func (c *Cacher) processEvent(event watchCacheEvent) {
	c.incoming <- event
}

func (c *Cacher) dispatchEvents() {
	for {
		select {
		case event, ok := <-c.incoming:
			if !ok {
				return
			}
			c.dispatchEvent(&event)
		case <-c.stopCh:
			return
		}
	}
}

func (c *Cacher) dispatchEvent(event *watchCacheEvent) {
	triggerValues, supported := c.triggerValues(event)

	c.Lock()
	defer c.Unlock()
	// Iterate over "allWatchers" no matter what the trigger function is.
	for _, watcher := range c.watchers.allWatchers {
		watcher.add(event)
	}
	if supported {
		// Iterate over watchers interested in the given values of the trigger.
		for _, triggerValue := range triggerValues {
			for _, watcher := range c.watchers.valueWatchers[triggerValue] {
				watcher.add(event)
			}
		}
	} else {
		// supported equal to false generally means that trigger function
		// is not defined (or not aware of any indexes). In this case,
		// watchers filters should generally also don't generate any
		// trigger values, but can cause problems in case of some
		// misconfiguration. Thus we paranoidly leave this branch.

		// Iterate over watchers interested in exact values for all values.
		for _, watchers := range c.watchers.valueWatchers {
			for _, watcher := range watchers {
				watcher.add(event)
			}
		}
	}
}

func (c *Cacher) terminateAllWatchers() {
	glog.Warningf("Terminating all watchers from cacher %v", c.objectType)
	c.Lock()
	defer c.Unlock()
	c.watchers.terminateAll()
}

func (c *Cacher) isStopped() bool {
	c.stopLock.RLock()
	defer c.stopLock.RUnlock()
	return c.stopped
}

func (c *Cacher) Stop() {
	c.stopLock.Lock()
	c.stopped = true
	c.stopLock.Unlock()
	close(c.stopCh)
	c.stopWg.Wait()
}

func forgetWatcher(c *Cacher, index int, triggerValue string, triggerSupported bool) func(bool) {
	return func(lock bool) {
		if lock {
			c.Lock()
			defer c.Unlock()
		}
		// It's possible that the watcher is already not in the structure (e.g. in case of
		// simulaneous Stop() and terminateAllWatchers(), but it doesn't break anything.
		c.watchers.deleteWatcher(index, triggerValue, triggerSupported)
	}
}

func filterFunction(key string, keyFunc func(runtime.Object) (string, error), filter Filter) Filter {
	filterFunc := func(obj runtime.Object) bool {
		objKey, err := keyFunc(obj)
		if err != nil {
			glog.Errorf("invalid object for filter: %v", obj)
			return false
		}
		if !hasPathPrefix(objKey, key) {
			return false
		}
		return filter.Filter(obj)
	}
	return NewSimpleFilter(filterFunc, filter.Trigger)
}

// Returns resource version to which the underlying cache is synced.
func (c *Cacher) LastSyncResourceVersion() (uint64, error) {
	c.ready.wait()

	resourceVersion := c.reflector.LastSyncResourceVersion()
	if resourceVersion == "" {
		return 0, nil
	}

	return strconv.ParseUint(resourceVersion, 10, 64)
}

// cacherListerWatcher opaques storage.Interface to expose cache.ListerWatcher.
type cacherListerWatcher struct {
	storage        Interface
	resourcePrefix string
	newListFunc    func() runtime.Object
}

func newCacherListerWatcher(storage Interface, resourcePrefix string, newListFunc func() runtime.Object) cache.ListerWatcher {
	return &cacherListerWatcher{
		storage:        storage,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
	}
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) List(options api.ListOptions) (runtime.Object, error) {
	list := lw.newListFunc()
	if err := lw.storage.List(context.TODO(), lw.resourcePrefix, "", Everything, list); err != nil {
		return nil, err
	}
	return list, nil
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) Watch(options api.ListOptions) (watch.Interface, error) {
	return lw.storage.WatchList(context.TODO(), lw.resourcePrefix, options.ResourceVersion, Everything)
}

// cacherWatch implements watch.Interface to return a single error
type errWatcher struct {
	result chan watch.Event
}

func newErrWatcher(err error) *errWatcher {
	// Create an error event
	errEvent := watch.Event{Type: watch.Error}
	switch err := err.(type) {
	case runtime.Object:
		errEvent.Object = err
	case *errors.StatusError:
		errEvent.Object = &err.ErrStatus
	default:
		errEvent.Object = &unversioned.Status{
			Status:  unversioned.StatusFailure,
			Message: err.Error(),
			Reason:  unversioned.StatusReasonInternalError,
			Code:    http.StatusInternalServerError,
		}
	}

	// Create a watcher with room for a single event, populate it, and close the channel
	watcher := &errWatcher{result: make(chan watch.Event, 1)}
	watcher.result <- errEvent
	close(watcher.result)

	return watcher
}

// Implements watch.Interface.
func (c *errWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *errWatcher) Stop() {
	// no-op
}

// cacherWatch implements watch.Interface
type cacheWatcher struct {
	sync.Mutex
	input   chan watchCacheEvent
	result  chan watch.Event
	filter  Filter
	stopped bool
	forget  func(bool)
}

func newCacheWatcher(resourceVersion uint64, initEvents []watchCacheEvent, filter Filter, forget func(bool)) *cacheWatcher {
	watcher := &cacheWatcher{
		input:   make(chan watchCacheEvent, 10),
		result:  make(chan watch.Event, 10),
		filter:  filter,
		stopped: false,
		forget:  forget,
	}
	go watcher.process(initEvents, resourceVersion)
	return watcher
}

// Implements watch.Interface.
func (c *cacheWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *cacheWatcher) Stop() {
	c.forget(true)
	c.stop()
}

func (c *cacheWatcher) stop() {
	c.Lock()
	defer c.Unlock()
	if !c.stopped {
		c.stopped = true
		close(c.input)
	}
}

var timerPool sync.Pool

func (c *cacheWatcher) add(event *watchCacheEvent) {
	// Try to send the event immediately, without blocking.
	select {
	case c.input <- *event:
		return
	default:
	}

	// OK, block sending, but only for up to 5 seconds.
	// cacheWatcher.add is called very often, so arrange
	// to reuse timers instead of constantly allocating.
	startTime := time.Now()
	const timeout = 5 * time.Second
	t, ok := timerPool.Get().(*time.Timer)
	if ok {
		t.Reset(timeout)
	} else {
		t = time.NewTimer(timeout)
	}
	defer timerPool.Put(t)

	select {
	case c.input <- *event:
		stopped := t.Stop()
		if !stopped {
			// Consume triggered (but not yet received) timer event
			// so that future reuse does not get a spurious timeout.
			<-t.C
		}
	case <-t.C:
		// This means that we couldn't send event to that watcher.
		// Since we don't want to block on it infinitely,
		// we simply terminate it.
		c.forget(false)
		c.stop()
	}
	glog.V(2).Infof("cacheWatcher add function blocked processing for %v", time.Since(startTime))
}

func (c *cacheWatcher) sendWatchCacheEvent(event watchCacheEvent) {
	curObjPasses := event.Type != watch.Deleted && c.filter.Filter(event.Object)
	oldObjPasses := false
	if event.PrevObject != nil {
		oldObjPasses = c.filter.Filter(event.PrevObject)
	}
	if !curObjPasses && !oldObjPasses {
		// Watcher is not interested in that object.
		return
	}

	object, err := api.Scheme.Copy(event.Object)
	if err != nil {
		glog.Errorf("unexpected copy error: %v", err)
		return
	}
	switch {
	case curObjPasses && !oldObjPasses:
		c.result <- watch.Event{Type: watch.Added, Object: object}
	case curObjPasses && oldObjPasses:
		c.result <- watch.Event{Type: watch.Modified, Object: object}
	case !curObjPasses && oldObjPasses:
		c.result <- watch.Event{Type: watch.Deleted, Object: object}
	}
}

func (c *cacheWatcher) process(initEvents []watchCacheEvent, resourceVersion uint64) {
	defer utilruntime.HandleCrash()

	for _, event := range initEvents {
		c.sendWatchCacheEvent(event)
	}
	defer close(c.result)
	defer c.Stop()
	for {
		event, ok := <-c.input
		if !ok {
			return
		}
		// only send events newer than resourceVersion
		if event.ResourceVersion > resourceVersion {
			c.sendWatchCacheEvent(event)
		}
	}
}

type ready struct {
	ok bool
	c  *sync.Cond
}

func newReady() *ready {
	return &ready{c: sync.NewCond(&sync.Mutex{})}
}

func (r *ready) wait() {
	r.c.L.Lock()
	for !r.ok {
		r.c.Wait()
	}
	r.c.L.Unlock()
}

func (r *ready) set(ok bool) {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	r.ok = ok
	r.c.Broadcast()
}
