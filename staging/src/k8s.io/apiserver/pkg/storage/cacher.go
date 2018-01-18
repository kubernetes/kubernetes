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

	"github.com/golang/glog"
	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	"k8s.io/client-go/tools/cache"
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

	// KeyFunc is used to get a key in the underlying storage for a given object.
	KeyFunc func(runtime.Object) (string, error)

	// GetAttrsFunc is used to get object labels, fields, and the uninitialized bool
	GetAttrsFunc func(runtime.Object) (label labels.Set, field fields.Set, uninitialized bool, err error)

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

func (i *indexedWatchers) terminateAll(objectType reflect.Type) {
	if len(i.allWatchers) > 0 || len(i.valueWatchers) > 0 {
		glog.Warningf("Terminating all watchers from cacher %v", objectType)
	}
	i.allWatchers.terminateAll()
	for index, watchers := range i.valueWatchers {
		watchers.terminateAll()
		delete(i.valueWatchers, index)
	}
}

type watchFilterFunc func(key string, l labels.Set, f fields.Set, uninitialized bool) bool

// Cacher is responsible for serving WATCH and LIST requests for a given
// resource from its internal cache and updating its cache in the background
// based on the underlying storage contents.
// Cacher implements storage.Interface (although most of the calls are just
// delegated to the underlying storage).
type Cacher struct {
	// HighWaterMarks for performance debugging.
	// Important: Since HighWaterMark is using sync/atomic, it has to be at the top of the struct due to a bug on 32-bit platforms
	// See: https://golang.org/pkg/sync/atomic/ for more information
	incomingHWM HighWaterMark
	// Incoming events that should be dispatched to watchers.
	incoming chan watchCacheEvent

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

	// triggerFunc is used for optimizing amount of watchers that needs to process
	// an incoming event.
	triggerFunc TriggerPublisherFunc
	// watchers is mapping from the value of trigger function that a
	// watcher is interested into the watchers
	watcherIdx int
	watchers   indexedWatchers

	// Defines a time budget that can be spend on waiting for not-ready watchers
	// while dispatching event before shutting them down.
	dispatchTimeoutBudget *timeBudget

	// Handling graceful termination.
	stopLock sync.RWMutex
	stopped  bool
	stopCh   chan struct{}
	stopWg   sync.WaitGroup
}

// Create a new Cacher responsible for servicing WATCH and LIST requests from
// its internal cache and updating its cache in the background based on the
// given configuration.
func NewCacherFromConfig(config CacherConfig) *Cacher {
	watchCache := newWatchCache(config.CacheCapacity, config.KeyFunc, config.GetAttrsFunc)
	listerWatcher := newCacherListerWatcher(config.Storage, config.ResourcePrefix, config.NewListFunc)
	reflectorName := "storage/cacher.go:" + config.ResourcePrefix

	// Give this error when it is constructed rather than when you get the
	// first watch item, because it's much easier to track down that way.
	if obj, ok := config.Type.(runtime.Object); ok {
		if err := runtime.CheckCodec(config.Codec, obj); err != nil {
			panic("storage codec doesn't seem to match given type: " + err.Error())
		}
	}

	stopCh := make(chan struct{})
	cacher := &Cacher{
		ready:       newReady(),
		storage:     config.Storage,
		objectType:  reflect.TypeOf(config.Type),
		watchCache:  watchCache,
		reflector:   cache.NewNamedReflector(reflectorName, listerWatcher, config.Type, watchCache, 0),
		versioner:   config.Versioner,
		triggerFunc: config.TriggerPublisherFunc,
		watcherIdx:  0,
		watchers: indexedWatchers{
			allWatchers:   make(map[int]*cacheWatcher),
			valueWatchers: make(map[string]watchersMap),
		},
		// TODO: Figure out the correct value for the buffer size.
		incoming:              make(chan watchCacheEvent, 100),
		dispatchTimeoutBudget: newTimeBudget(stopCh),
		// We need to (potentially) stop both:
		// - wait.Until go-routine
		// - reflector.ListAndWatch
		// and there are no guarantees on the order that they will stop.
		// So we will be simply closing the channel, and synchronizing on the WaitGroup.
		stopCh: stopCh,
	}
	watchCache.SetOnEvent(cacher.processEvent)
	go cacher.dispatchEvents()

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
func (c *Cacher) Watch(ctx context.Context, key string, resourceVersion string, pred SelectionPredicate) (watch.Interface, error) {
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
	// TODO: Currently we assume that in a given Cacher object, any <predicate> that is
	// passed here is aware of exactly the same trigger (at most one).
	// Thus, either 0 or 1 values will be returned.
	if matchValues := pred.MatcherIndex(); len(matchValues) > 0 {
		triggerValue, triggerSupported = matchValues[0].Value, true
	}

	// If there is triggerFunc defined, but triggerSupported is false,
	// we can't narrow the amount of events significantly at this point.
	//
	// That said, currently triggerFunc is defined only for Pods and Nodes,
	// and there is only constant number of watchers for which triggerSupported
	// is false (excluding those issues explicitly by users).
	// Thus, to reduce the risk of those watchers blocking all watchers of a
	// given resource in the system, we increase the sizes of buffers for them.
	chanSize := 10
	if c.triggerFunc != nil && !triggerSupported {
		// TODO: We should tune this value and ideally make it dependent on the
		// number of objects of a given type and/or their churn.
		chanSize = 1000
	}

	c.Lock()
	defer c.Unlock()
	forget := forgetWatcher(c, c.watcherIdx, triggerValue, triggerSupported)
	watcher := newCacheWatcher(watchRV, chanSize, initEvents, watchFilterFunction(key, pred), forget)

	c.watchers.addWatcher(watcher, c.watcherIdx, triggerValue, triggerSupported)
	c.watcherIdx++
	return watcher, nil
}

// Implements storage.Interface.
func (c *Cacher) WatchList(ctx context.Context, key string, resourceVersion string, pred SelectionPredicate) (watch.Interface, error) {
	return c.Watch(ctx, key, resourceVersion, pred)
}

// Implements storage.Interface.
func (c *Cacher) Get(ctx context.Context, key string, resourceVersion string, objPtr runtime.Object, ignoreNotFound bool) error {
	if resourceVersion == "" {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility).
		return c.storage.Get(ctx, key, resourceVersion, objPtr, ignoreNotFound)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	getRV, err := ParseListResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	if getRV == 0 && !c.ready.check() {
		// If Cacher is not yet initialized and we don't require any specific
		// minimal resource version, simply forward the request to storage.
		return c.storage.Get(ctx, key, resourceVersion, objPtr, ignoreNotFound)
	}

	// Do not create a trace - it's not for free and there are tons
	// of Get requests. We can add it if it will be really needed.
	c.ready.wait()

	objVal, err := conversion.EnforcePtr(objPtr)
	if err != nil {
		return err
	}

	obj, exists, readResourceVersion, err := c.watchCache.WaitUntilFreshAndGet(getRV, key, nil)
	if err != nil {
		return err
	}

	if exists {
		elem, ok := obj.(*storeElement)
		if !ok {
			return fmt.Errorf("non *storeElement returned from storage: %v", obj)
		}
		objVal.Set(reflect.ValueOf(elem.Object).Elem())
	} else {
		objVal.Set(reflect.Zero(objVal.Type()))
		if !ignoreNotFound {
			return NewKeyNotFoundError(key, int64(readResourceVersion))
		}
	}
	return nil
}

// Implements storage.Interface.
func (c *Cacher) GetToList(ctx context.Context, key string, resourceVersion string, pred SelectionPredicate, listObj runtime.Object) error {
	pagingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	if resourceVersion == "" || (pagingEnabled && (len(pred.Continue) > 0 || pred.Limit > 0)) {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility). If a continuation or limit is
		// requested, serve it from the underlying storage as well.
		return c.storage.GetToList(ctx, key, resourceVersion, pred, listObj)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	listRV, err := ParseListResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	if listRV == 0 && !c.ready.check() {
		// If Cacher is not yet initialized and we don't require any specific
		// minimal resource version, simply forward the request to storage.
		return c.storage.GetToList(ctx, key, resourceVersion, pred, listObj)
	}

	trace := utiltrace.New(fmt.Sprintf("cacher %v: List", c.objectType.String()))
	defer trace.LogIfLong(500 * time.Millisecond)

	c.ready.wait()
	trace.Step("Ready")

	// List elements with at least 'listRV' from cache.
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}
	filter := filterFunction(key, pred)

	obj, exists, readResourceVersion, err := c.watchCache.WaitUntilFreshAndGet(listRV, key, trace)
	if err != nil {
		return err
	}
	trace.Step("Got from cache")

	if exists {
		elem, ok := obj.(*storeElement)
		if !ok {
			return fmt.Errorf("non *storeElement returned from storage: %v", obj)
		}
		if filter(elem.Key, elem.Object) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(elem.Object).Elem()))
		}
	}
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, readResourceVersion, ""); err != nil {
			return err
		}
	}
	return nil
}

// Implements storage.Interface.
func (c *Cacher) List(ctx context.Context, key string, resourceVersion string, pred SelectionPredicate, listObj runtime.Object) error {
	pagingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	hasContinuation := pagingEnabled && len(pred.Continue) > 0
	hasLimit := pagingEnabled && pred.Limit > 0 && resourceVersion != "0"
	if resourceVersion == "" || hasContinuation || hasLimit {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility). If a continuation is
		// requested, serve it from the underlying storage as well.
		// Limits are only sent to storage when resourceVersion is non-zero
		// since the watch cache isn't able to perform continuations, and
		// limits are ignored when resource version is zero.
		return c.storage.List(ctx, key, resourceVersion, pred, listObj)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	listRV, err := ParseListResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	if listRV == 0 && !c.ready.check() {
		// If Cacher is not yet initialized and we don't require any specific
		// minimal resource version, simply forward the request to storage.
		return c.storage.List(ctx, key, resourceVersion, pred, listObj)
	}

	trace := utiltrace.New(fmt.Sprintf("cacher %v: List", c.objectType.String()))
	defer trace.LogIfLong(500 * time.Millisecond)

	c.ready.wait()
	trace.Step("Ready")

	// List elements with at least 'listRV' from cache.
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil || listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}
	filter := filterFunction(key, pred)

	objs, readResourceVersion, err := c.watchCache.WaitUntilFreshAndList(listRV, trace)
	if err != nil {
		return err
	}
	trace.Step(fmt.Sprintf("Listed %d items from cache", len(objs)))
	if len(objs) > listVal.Cap() && pred.Label.Empty() && pred.Field.Empty() {
		// Resize the slice appropriately, since we already know that none
		// of the elements will be filtered out.
		listVal.Set(reflect.MakeSlice(reflect.SliceOf(c.objectType.Elem()), 0, len(objs)))
		trace.Step("Resized result")
	}
	for _, obj := range objs {
		elem, ok := obj.(*storeElement)
		if !ok {
			return fmt.Errorf("non *storeElement returned from storage: %v", obj)
		}
		if filter(elem.Key, elem.Object) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(elem.Object).Elem()))
		}
	}
	trace.Step(fmt.Sprintf("Filtered %d items", listVal.Len()))
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, readResourceVersion, ""); err != nil {
			return err
		}
	}
	return nil
}

// Implements storage.Interface.
func (c *Cacher) GuaranteedUpdate(
	ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *Preconditions, tryUpdate UpdateFunc, _ ...runtime.Object) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := c.watchCache.GetByKey(key); err != nil {
		glog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return c.storage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate, currObj)
	}
	// If we couldn't get the object, fallback to no-suggestion.
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

func (c *Cacher) processEvent(event *watchCacheEvent) {
	if curLen := int64(len(c.incoming)); c.incomingHWM.Update(curLen) {
		// Monitor if this gets backed up, and how much.
		glog.V(1).Infof("cacher (%v): %v objects queued in incoming channel.", c.objectType.String(), curLen)
	}
	c.incoming <- *event
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
		watcher.add(event, c.dispatchTimeoutBudget)
	}
	if supported {
		// Iterate over watchers interested in the given values of the trigger.
		for _, triggerValue := range triggerValues {
			for _, watcher := range c.watchers.valueWatchers[triggerValue] {
				watcher.add(event, c.dispatchTimeoutBudget)
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
				watcher.add(event, c.dispatchTimeoutBudget)
			}
		}
	}
}

func (c *Cacher) terminateAllWatchers() {
	c.Lock()
	defer c.Unlock()
	c.watchers.terminateAll(c.objectType)
}

func (c *Cacher) isStopped() bool {
	c.stopLock.RLock()
	defer c.stopLock.RUnlock()
	return c.stopped
}

func (c *Cacher) Stop() {
	// avoid stopping twice (note: cachers are shared with subresources)
	if c.isStopped() {
		return
	}
	c.stopLock.Lock()
	if c.stopped {
		c.stopLock.Unlock()
		return
	}
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
		} else {
			// false is currently passed only if we are forcing watcher to close due
			// to its unresponsiveness and blocking other watchers.
			// TODO: Get this information in cleaner way.
			glog.V(1).Infof("Forcing watcher close due to unresponsiveness: %v", c.objectType.String())
		}
		// It's possible that the watcher is already not in the structure (e.g. in case of
		// simultaneous Stop() and terminateAllWatchers(), but it doesn't break anything.
		c.watchers.deleteWatcher(index, triggerValue, triggerSupported)
	}
}

func filterFunction(key string, p SelectionPredicate) func(string, runtime.Object) bool {
	filterFunc := func(objKey string, obj runtime.Object) bool {
		if !hasPathPrefix(objKey, key) {
			return false
		}
		matches, err := p.Matches(obj)
		if err != nil {
			glog.Errorf("invalid object for matching. Obj: %v. Err: %v", obj, err)
			return false
		}
		return matches
	}
	return filterFunc
}

func watchFilterFunction(key string, p SelectionPredicate) watchFilterFunc {
	filterFunc := func(objKey string, label labels.Set, field fields.Set, uninitialized bool) bool {
		if !hasPathPrefix(objKey, key) {
			return false
		}
		return p.MatchesObjectAttributes(label, field, uninitialized)
	}
	return filterFunc
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
func (lw *cacherListerWatcher) List(options metav1.ListOptions) (runtime.Object, error) {
	list := lw.newListFunc()
	if err := lw.storage.List(context.TODO(), lw.resourcePrefix, "", Everything, list); err != nil {
		return nil, err
	}
	return list, nil
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return lw.storage.WatchList(context.TODO(), lw.resourcePrefix, options.ResourceVersion, Everything)
}

// errWatcher implements watch.Interface to return a single error
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
		errEvent.Object = &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: err.Error(),
			Reason:  metav1.StatusReasonInternalError,
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

// cachWatcher implements watch.Interface
type cacheWatcher struct {
	sync.Mutex
	input   chan *watchCacheEvent
	result  chan watch.Event
	done    chan struct{}
	filter  watchFilterFunc
	stopped bool
	forget  func(bool)
}

func newCacheWatcher(resourceVersion uint64, chanSize int, initEvents []*watchCacheEvent, filter watchFilterFunc, forget func(bool)) *cacheWatcher {
	watcher := &cacheWatcher{
		input:   make(chan *watchCacheEvent, chanSize),
		result:  make(chan watch.Event, chanSize),
		done:    make(chan struct{}),
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
		close(c.done)
		close(c.input)
	}
}

var timerPool sync.Pool

func (c *cacheWatcher) add(event *watchCacheEvent, budget *timeBudget) {
	// Try to send the event immediately, without blocking.
	select {
	case c.input <- event:
		return
	default:
	}

	// OK, block sending, but only for up to <timeout>.
	// cacheWatcher.add is called very often, so arrange
	// to reuse timers instead of constantly allocating.
	startTime := time.Now()
	timeout := budget.takeAvailable()

	t, ok := timerPool.Get().(*time.Timer)
	if ok {
		t.Reset(timeout)
	} else {
		t = time.NewTimer(timeout)
	}
	defer timerPool.Put(t)

	select {
	case c.input <- event:
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

	budget.returnUnused(timeout - time.Since(startTime))
}

// NOTE: sendWatchCacheEvent is assumed to not modify <event> !!!
func (c *cacheWatcher) sendWatchCacheEvent(event *watchCacheEvent) {
	curObjPasses := event.Type != watch.Deleted && c.filter(event.Key, event.ObjLabels, event.ObjFields, event.ObjUninitialized)
	oldObjPasses := false
	if event.PrevObject != nil {
		oldObjPasses = c.filter(event.Key, event.PrevObjLabels, event.PrevObjFields, event.PrevObjUninitialized)
	}
	if !curObjPasses && !oldObjPasses {
		// Watcher is not interested in that object.
		return
	}

	var watchEvent watch.Event
	switch {
	case curObjPasses && !oldObjPasses:
		watchEvent = watch.Event{Type: watch.Added, Object: event.Object.DeepCopyObject()}
	case curObjPasses && oldObjPasses:
		watchEvent = watch.Event{Type: watch.Modified, Object: event.Object.DeepCopyObject()}
	case !curObjPasses && oldObjPasses:
		watchEvent = watch.Event{Type: watch.Deleted, Object: event.PrevObject.DeepCopyObject()}
	}

	// We need to ensure that if we put event X to the c.result, all
	// previous events were already put into it before, no matter whether
	// c.done is close or not.
	// Thus we cannot simply select from c.done and c.result and this
	// would give us non-determinism.
	// At the same time, we don't want to block infinitely on putting
	// to c.result, when c.done is already closed.

	// This ensures that with c.done already close, we at most once go
	// into the next select after this. With that, no matter which
	// statement we choose there, we will deliver only consecutive
	// events.
	select {
	case <-c.done:
		return
	default:
	}

	select {
	case c.result <- watchEvent:
	case <-c.done:
	}
}

func (c *cacheWatcher) process(initEvents []*watchCacheEvent, resourceVersion uint64) {
	defer utilruntime.HandleCrash()

	// Check how long we are processing initEvents.
	// As long as these are not processed, we are not processing
	// any incoming events, so if it takes long, we may actually
	// block all watchers for some time.
	// TODO: From the logs it seems that there happens processing
	// times even up to 1s which is very long. However, this doesn't
	// depend that much on the number of initEvents. E.g. from the
	// 2000-node Kubemark run we have logs like this, e.g.:
	// ... processing 13862 initEvents took 66.808689ms
	// ... processing 14040 initEvents took 993.532539ms
	// We should understand what is blocking us in those cases (e.g.
	// is it lack of CPU, network, or sth else) and potentially
	// consider increase size of result buffer in those cases.
	const initProcessThreshold = 500 * time.Millisecond
	startTime := time.Now()
	for _, event := range initEvents {
		c.sendWatchCacheEvent(event)
	}
	processingTime := time.Since(startTime)
	if processingTime > initProcessThreshold {
		objType := "<null>"
		if len(initEvents) > 0 {
			objType = reflect.TypeOf(initEvents[0].Object).String()
		}
		glog.V(2).Infof("processing %d initEvents of %s took %v", len(initEvents), objType, processingTime)
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

// TODO: Make check() function more sophisticated, in particular
// allow it to behave as "waitWithTimeout".
func (r *ready) check() bool {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	return r.ok
}

func (r *ready) set(ok bool) {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	r.ok = ok
	r.c.Broadcast()
}
