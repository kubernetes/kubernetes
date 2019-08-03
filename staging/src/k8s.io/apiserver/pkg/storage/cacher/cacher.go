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

package cacher

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"time"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	utiltrace "k8s.io/utils/trace"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	initCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "apiserver_init_events_total",
			Help: "Counter of init events processed in watchcache broken by resource type",
		},
		[]string{"resource"},
	)
	emptyFunc = func() {}
)

const (
	// storageWatchListPageSize is the cacher's request chunk size of
	// initial and resync watch lists to storage.
	storageWatchListPageSize = int64(10000)
)

func init() {
	prometheus.MustRegister(initCounter)
}

// Config contains the configuration for a given Cache.
type Config struct {
	// Maximum size of the history cached in memory.
	CacheCapacity int

	// An underlying storage.Interface.
	Storage storage.Interface

	// An underlying storage.Versioner.
	Versioner storage.Versioner

	// The Cache will be caching objects of a given Type and assumes that they
	// are all stored under ResourcePrefix directory in the underlying database.
	ResourcePrefix string

	// KeyFunc is used to get a key in the underlying storage for a given object.
	KeyFunc func(runtime.Object) (string, error)

	// GetAttrsFunc is used to get object labels, fields
	GetAttrsFunc func(runtime.Object) (label labels.Set, field fields.Set, err error)

	// IndexerFuncs is used for optimizing amount of watchers that
	// needs to process an incoming event.
	IndexerFuncs storage.IndexerFuncs

	// NewFunc is a function that creates new empty object storing a object of type Type.
	NewFunc func() runtime.Object

	// NewList is a function that creates new empty object storing a list of
	// objects of type Type.
	NewListFunc func() runtime.Object

	Codec runtime.Codec
}

type watchersMap map[int]*cacheWatcher

func (wm watchersMap) addWatcher(w *cacheWatcher, number int) {
	wm[number] = w
}

func (wm watchersMap) deleteWatcher(number int, done func(*cacheWatcher)) {
	if watcher, ok := wm[number]; ok {
		delete(wm, number)
		done(watcher)
	}
}

func (wm watchersMap) terminateAll(done func(*cacheWatcher)) {
	for key, watcher := range wm {
		delete(wm, key)
		done(watcher)
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

func (i *indexedWatchers) deleteWatcher(number int, value string, supported bool, done func(*cacheWatcher)) {
	if supported {
		i.valueWatchers[value].deleteWatcher(number, done)
		if len(i.valueWatchers[value]) == 0 {
			delete(i.valueWatchers, value)
		}
	} else {
		i.allWatchers.deleteWatcher(number, done)
	}
}

func (i *indexedWatchers) terminateAll(objectType reflect.Type, done func(*cacheWatcher)) {
	if len(i.allWatchers) > 0 || len(i.valueWatchers) > 0 {
		klog.Warningf("Terminating all watchers from cacher %v", objectType)
	}
	i.allWatchers.terminateAll(done)
	for _, watchers := range i.valueWatchers {
		watchers.terminateAll(done)
	}
	i.valueWatchers = map[string]watchersMap{}
}

// As we don't need a high precision here, we keep all watchers timeout within a
// second in a bucket, and pop up them once at the timeout. To be more specific,
// if you set fire time at X, you can get the bookmark within (X-1,X+1) period.
type watcherBookmarkTimeBuckets struct {
	lock            sync.Mutex
	watchersBuckets map[int64][]*cacheWatcher
	startBucketID   int64
	clock           clock.Clock
}

func newTimeBucketWatchers(clock clock.Clock) *watcherBookmarkTimeBuckets {
	return &watcherBookmarkTimeBuckets{
		watchersBuckets: make(map[int64][]*cacheWatcher),
		startBucketID:   clock.Now().Unix(),
		clock:           clock,
	}
}

// adds a watcher to the bucket, if the deadline is before the start, it will be
// added to the first one.
func (t *watcherBookmarkTimeBuckets) addWatcher(w *cacheWatcher) bool {
	nextTime, ok := w.nextBookmarkTime(t.clock.Now())
	if !ok {
		return false
	}
	bucketID := nextTime.Unix()
	t.lock.Lock()
	defer t.lock.Unlock()
	if bucketID < t.startBucketID {
		bucketID = t.startBucketID
	}
	watchers, _ := t.watchersBuckets[bucketID]
	t.watchersBuckets[bucketID] = append(watchers, w)
	return true
}

func (t *watcherBookmarkTimeBuckets) popExpiredWatchers() [][]*cacheWatcher {
	currentBucketID := t.clock.Now().Unix()
	// There should be one or two elements in almost all cases
	expiredWatchers := make([][]*cacheWatcher, 0, 2)
	t.lock.Lock()
	defer t.lock.Unlock()
	for ; t.startBucketID <= currentBucketID; t.startBucketID++ {
		if watchers, ok := t.watchersBuckets[t.startBucketID]; ok {
			delete(t.watchersBuckets, t.startBucketID)
			expiredWatchers = append(expiredWatchers, watchers)
		}
	}
	return expiredWatchers
}

type filterWithAttrsFunc func(key string, l labels.Set, f fields.Set) bool

type indexedTriggerFunc struct {
	indexName   string
	indexerFunc storage.IndexerFunc
}

// Cacher is responsible for serving WATCH and LIST requests for a given
// resource from its internal cache and updating its cache in the background
// based on the underlying storage contents.
// Cacher implements storage.Interface (although most of the calls are just
// delegated to the underlying storage).
type Cacher struct {
	// HighWaterMarks for performance debugging.
	// Important: Since HighWaterMark is using sync/atomic, it has to be at the top of the struct due to a bug on 32-bit platforms
	// See: https://golang.org/pkg/sync/atomic/ for more information
	incomingHWM storage.HighWaterMark
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
	storage storage.Interface

	// Expected type of objects in the underlying cache.
	objectType reflect.Type

	// "sliding window" of recent changes of objects and the current state.
	watchCache *watchCache
	reflector  *cache.Reflector

	// Versioner is used to handle resource versions.
	versioner storage.Versioner

	// newFunc is a function that creates new empty object storing a object of type Type.
	newFunc func() runtime.Object

	// indexedTrigger is used for optimizing amount of watchers that needs to process
	// an incoming event.
	indexedTrigger *indexedTriggerFunc
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

	clock clock.Clock
	// timer is used to avoid unnecessary allocations in underlying watchers.
	timer *time.Timer

	// dispatching determines whether there is currently dispatching of
	// any event in flight.
	dispatching bool
	// watchersBuffer is a list of watchers potentially interested in currently
	// dispatched event.
	watchersBuffer []*cacheWatcher
	// blockedWatchers is a list of watchers whose buffer is currently full.
	blockedWatchers []*cacheWatcher
	// watchersToStop is a list of watchers that were supposed to be stopped
	// during current dispatching, but stopping was deferred to the end of
	// dispatching that event to avoid race with closing channels in watchers.
	watchersToStop []*cacheWatcher
	// Maintain a timeout queue to send the bookmark event before the watcher times out.
	bookmarkWatchers *watcherBookmarkTimeBuckets
	// watchBookmark feature-gate
	watchBookmarkEnabled bool
}

// NewCacherFromConfig creates a new Cacher responsible for servicing WATCH and LIST requests from
// its internal cache and updating its cache in the background based on the
// given configuration.
func NewCacherFromConfig(config Config) (*Cacher, error) {
	stopCh := make(chan struct{})
	obj := config.NewFunc()
	// Give this error when it is constructed rather than when you get the
	// first watch item, because it's much easier to track down that way.
	if err := runtime.CheckCodec(config.Codec, obj); err != nil {
		return nil, fmt.Errorf("storage codec doesn't seem to match given type: %v", err)
	}

	var indexedTrigger *indexedTriggerFunc
	if config.IndexerFuncs != nil {
		// For now, we don't support multiple trigger functions defined
		// for a given resource.
		if len(config.IndexerFuncs) > 1 {
			return nil, fmt.Errorf("cacher %s doesn't support more than one IndexerFunc: ", reflect.TypeOf(obj).String())
		}
		for key, value := range config.IndexerFuncs {
			if value != nil {
				indexedTrigger = &indexedTriggerFunc{
					indexName:   key,
					indexerFunc: value,
				}
			}
		}
	}

	clock := clock.RealClock{}
	cacher := &Cacher{
		ready:          newReady(),
		storage:        config.Storage,
		objectType:     reflect.TypeOf(obj),
		versioner:      config.Versioner,
		newFunc:        config.NewFunc,
		indexedTrigger: indexedTrigger,
		watcherIdx:     0,
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
		stopCh:               stopCh,
		clock:                clock,
		timer:                time.NewTimer(time.Duration(0)),
		bookmarkWatchers:     newTimeBucketWatchers(clock),
		watchBookmarkEnabled: utilfeature.DefaultFeatureGate.Enabled(features.WatchBookmark),
	}

	// Ensure that timer is stopped.
	if !cacher.timer.Stop() {
		// Consume triggered (but not yet received) timer event
		// so that future reuse does not get a spurious timeout.
		<-cacher.timer.C
	}

	watchCache := newWatchCache(
		config.CacheCapacity, config.KeyFunc, cacher.processEvent, config.GetAttrsFunc, config.Versioner)
	listerWatcher := NewCacherListerWatcher(config.Storage, config.ResourcePrefix, config.NewListFunc)
	reflectorName := "storage/cacher.go:" + config.ResourcePrefix

	reflector := cache.NewNamedReflector(reflectorName, listerWatcher, obj, watchCache, 0)
	// Configure reflector's pager to for an appropriate pagination chunk size for fetching data from
	// storage. The pager falls back to full list if paginated list calls fail due to an "Expired" error.
	reflector.WatchListPageSize = storageWatchListPageSize

	cacher.watchCache = watchCache
	cacher.reflector = reflector

	go cacher.dispatchEvents()

	cacher.stopWg.Add(1)
	go func() {
		defer cacher.stopWg.Done()
		defer cacher.terminateAllWatchers()
		wait.Until(
			func() {
				if !cacher.isStopped() {
					cacher.startCaching(stopCh)
				}
			}, time.Second, stopCh,
		)
	}()

	return cacher, nil
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
		klog.Errorf("unexpected ListAndWatch error: %v", err)
	}
}

// Versioner implements storage.Interface.
func (c *Cacher) Versioner() storage.Versioner {
	return c.storage.Versioner()
}

// Create implements storage.Interface.
func (c *Cacher) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	return c.storage.Create(ctx, key, obj, out, ttl)
}

// Delete implements storage.Interface.
func (c *Cacher) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc) error {
	return c.storage.Delete(ctx, key, out, preconditions, validateDeletion)
}

// Watch implements storage.Interface.
func (c *Cacher) Watch(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	watchRV, err := c.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	c.ready.wait()

	triggerValue, triggerSupported := "", false
	if c.indexedTrigger != nil {
		for _, field := range pred.IndexFields {
			if field == c.indexedTrigger.indexName {
				if value, ok := pred.Field.RequiresExactMatch(field); ok {
					triggerValue, triggerSupported = value, true
				}
			}
		}
	}

	// If there is indexedTrigger defined, but triggerSupported is false,
	// we can't narrow the amount of events significantly at this point.
	//
	// That said, currently indexedTrigger is defined only for couple resources:
	// Pods, Nodes, Secrets and ConfigMaps and there is only a constant
	// number of watchers for which triggerSupported is false (excluding those
	// issued explicitly by users).
	// Thus, to reduce the risk of those watchers blocking all watchers of a
	// given resource in the system, we increase the sizes of buffers for them.
	chanSize := 10
	if c.indexedTrigger != nil && !triggerSupported {
		// TODO: We should tune this value and ideally make it dependent on the
		// number of objects of a given type and/or their churn.
		chanSize = 1000
	}

	// Determine watch timeout('0' means deadline is not set, ignore checking)
	deadline, _ := ctx.Deadline()
	// Create a watcher here to reduce memory allocations under lock,
	// given that memory allocation may trigger GC and block the thread.
	// Also note that emptyFunc is a placeholder, until we will be able
	// to compute watcher.forget function (which has to happen under lock).
	watcher := newCacheWatcher(chanSize, filterWithAttrsFunction(key, pred), emptyFunc, c.versioner, deadline, pred.AllowWatchBookmarks, c.objectType)

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

	// With some events already sent, update resourceVersion so that
	// events that were buffered and not yet processed won't be delivered
	// to this watcher second time causing going back in time.
	if len(initEvents) > 0 {
		watchRV = initEvents[len(initEvents)-1].ResourceVersion
	}

	func() {
		c.Lock()
		defer c.Unlock()
		// Update watcher.forget function once we can compute it.
		watcher.forget = forgetWatcher(c, c.watcherIdx, triggerValue, triggerSupported)
		c.watchers.addWatcher(watcher, c.watcherIdx, triggerValue, triggerSupported)

		// Add it to the queue only when server and client support watch bookmarks.
		if c.watchBookmarkEnabled && watcher.allowWatchBookmarks {
			c.bookmarkWatchers.addWatcher(watcher)
		}
		c.watcherIdx++
	}()

	go watcher.process(ctx, initEvents, watchRV)
	return watcher, nil
}

// WatchList implements storage.Interface.
func (c *Cacher) WatchList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	return c.Watch(ctx, key, resourceVersion, pred)
}

// Get implements storage.Interface.
func (c *Cacher) Get(ctx context.Context, key string, resourceVersion string, objPtr runtime.Object, ignoreNotFound bool) error {
	if resourceVersion == "" {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility).
		return c.storage.Get(ctx, key, resourceVersion, objPtr, ignoreNotFound)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	getRV, err := c.versioner.ParseResourceVersion(resourceVersion)
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
			return storage.NewKeyNotFoundError(key, int64(readResourceVersion))
		}
	}
	return nil
}

// GetToList implements storage.Interface.
func (c *Cacher) GetToList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	pagingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	hasContinuation := pagingEnabled && len(pred.Continue) > 0
	hasLimit := pagingEnabled && pred.Limit > 0 && resourceVersion != "0"
	if resourceVersion == "" || hasContinuation || hasLimit {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility). If a continuation is
		// requested, serve it from the underlying storage as well.
		// Limits are only sent to storage when resourceVersion is non-zero
		// since the watch cache isn't able to perform continuations, and
		// limits are ignored when resource version is zero
		return c.storage.GetToList(ctx, key, resourceVersion, pred, listObj)
	}

	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	listRV, err := c.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	if listRV == 0 && !c.ready.check() {
		// If Cacher is not yet initialized and we don't require any specific
		// minimal resource version, simply forward the request to storage.
		return c.storage.GetToList(ctx, key, resourceVersion, pred, listObj)
	}

	trace := utiltrace.New("cacher list", utiltrace.Field{"type", c.objectType.String()})
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
	filter := filterWithAttrsFunction(key, pred)

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
		if filter(elem.Key, elem.Labels, elem.Fields) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(elem.Object).Elem()))
		}
	}
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, readResourceVersion, "", nil); err != nil {
			return err
		}
	}
	return nil
}

// List implements storage.Interface.
func (c *Cacher) List(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
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
	listRV, err := c.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	if listRV == 0 && !c.ready.check() {
		// If Cacher is not yet initialized and we don't require any specific
		// minimal resource version, simply forward the request to storage.
		return c.storage.List(ctx, key, resourceVersion, pred, listObj)
	}

	trace := utiltrace.New("cacher list", utiltrace.Field{"type", c.objectType.String()})
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
	filter := filterWithAttrsFunction(key, pred)

	objs, readResourceVersion, err := c.watchCache.WaitUntilFreshAndList(listRV, trace)
	if err != nil {
		return err
	}
	trace.Step("Listed items from cache", utiltrace.Field{"count", len(objs)})
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
		if filter(elem.Key, elem.Labels, elem.Fields) {
			listVal.Set(reflect.Append(listVal, reflect.ValueOf(elem.Object).Elem()))
		}
	}
	trace.Step("Filtered items", utiltrace.Field{"count", listVal.Len()})
	if c.versioner != nil {
		if err := c.versioner.UpdateList(listObj, readResourceVersion, "", nil); err != nil {
			return err
		}
	}
	return nil
}

// GuaranteedUpdate implements storage.Interface.
func (c *Cacher) GuaranteedUpdate(
	ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, _ ...runtime.Object) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := c.watchCache.GetByKey(key); err != nil {
		klog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return c.storage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate, currObj)
	}
	// If we couldn't get the object, fallback to no-suggestion.
	return c.storage.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate)
}

// Count implements storage.Interface.
func (c *Cacher) Count(pathPrefix string) (int64, error) {
	return c.storage.Count(pathPrefix)
}

func (c *Cacher) triggerValues(event *watchCacheEvent) ([]string, bool) {
	if c.indexedTrigger == nil {
		return nil, false
	}

	result := make([]string, 0, 2)
	result = append(result, c.indexedTrigger.indexerFunc(event.Object))
	if event.PrevObject == nil {
		return result, true
	}
	prevTriggerValue := c.indexedTrigger.indexerFunc(event.PrevObject)
	if result[0] != prevTriggerValue {
		result = append(result, prevTriggerValue)
	}
	return result, true
}

func (c *Cacher) processEvent(event *watchCacheEvent) {
	if curLen := int64(len(c.incoming)); c.incomingHWM.Update(curLen) {
		// Monitor if this gets backed up, and how much.
		klog.V(1).Infof("cacher (%v): %v objects queued in incoming channel.", c.objectType.String(), curLen)
	}
	c.incoming <- *event
}

func (c *Cacher) dispatchEvents() {
	// Jitter to help level out any aggregate load.
	bookmarkTimer := c.clock.NewTimer(wait.Jitter(time.Second, 0.25))
	// Stop the timer when watchBookmarkFeatureGate is not enabled.
	if !c.watchBookmarkEnabled && !bookmarkTimer.Stop() {
		<-bookmarkTimer.C()
	}
	defer bookmarkTimer.Stop()

	lastProcessedResourceVersion := uint64(0)
	for {
		select {
		case event, ok := <-c.incoming:
			if !ok {
				return
			}
			c.dispatchEvent(&event)
			lastProcessedResourceVersion = event.ResourceVersion
		case <-bookmarkTimer.C():
			bookmarkTimer.Reset(wait.Jitter(time.Second, 0.25))
			// Never send a bookmark event if we did not see an event here, this is fine
			// because we don't provide any guarantees on sending bookmarks.
			if lastProcessedResourceVersion == 0 {
				// pop expired watchers in case there has been no update
				c.bookmarkWatchers.popExpiredWatchers()
				continue
			}
			bookmarkEvent := &watchCacheEvent{
				Type:            watch.Bookmark,
				Object:          c.newFunc(),
				ResourceVersion: lastProcessedResourceVersion,
			}
			if err := c.versioner.UpdateObject(bookmarkEvent.Object, bookmarkEvent.ResourceVersion); err != nil {
				klog.Errorf("failure to set resourceVersion to %d on bookmark event %+v", bookmarkEvent.ResourceVersion, bookmarkEvent.Object)
				continue
			}
			c.dispatchEvent(bookmarkEvent)
		case <-c.stopCh:
			return
		}
	}
}

func (c *Cacher) dispatchEvent(event *watchCacheEvent) {
	c.startDispatching(event)
	defer c.finishDispatching()
	// Watchers stopped after startDispatching will be delayed to finishDispatching,

	// Since add() can block, we explicitly add when cacher is unlocked.
	// Dispatching event in nonblocking way first, which make faster watchers
	// not be blocked by slower ones.
	if event.Type == watch.Bookmark {
		for _, watcher := range c.watchersBuffer {
			watcher.nonblockingAdd(event)
		}
	} else {
		c.blockedWatchers = c.blockedWatchers[:0]
		for _, watcher := range c.watchersBuffer {
			if !watcher.nonblockingAdd(event) {
				c.blockedWatchers = append(c.blockedWatchers, watcher)
			}
		}

		if len(c.blockedWatchers) > 0 {
			// dispatchEvent is called very often, so arrange
			// to reuse timers instead of constantly allocating.
			startTime := time.Now()
			timeout := c.dispatchTimeoutBudget.takeAvailable()
			c.timer.Reset(timeout)

			// Make sure every watcher will try to send event without blocking first,
			// even if the timer has already expired.
			timer := c.timer
			for _, watcher := range c.blockedWatchers {
				if !watcher.add(event, timer) {
					// fired, clean the timer by set it to nil.
					timer = nil
				}
			}

			// Stop the timer if it is not fired
			if timer != nil && !timer.Stop() {
				// Consume triggered (but not yet received) timer event
				// so that future reuse does not get a spurious timeout.
				<-timer.C
			}

			c.dispatchTimeoutBudget.returnUnused(timeout - time.Since(startTime))
		}
	}
}

func (c *Cacher) startDispatchingBookmarkEvents() {
	// Pop already expired watchers. However, explicitly ignore stopped ones,
	// as we don't delete watcher from bookmarkWatchers when it is stopped.
	for _, watchers := range c.bookmarkWatchers.popExpiredWatchers() {
		for _, watcher := range watchers {
			// c.Lock() is held here.
			// watcher.stopThreadUnsafe() is protected by c.Lock()
			if watcher.stopped {
				continue
			}
			c.watchersBuffer = append(c.watchersBuffer, watcher)
			// Given that we send bookmark event once at deadline-2s, never push again
			// after the watcher pops up from the buckets. Once we decide to change the
			// strategy to more sophisticated, we may need it here.
		}
	}
}

// startDispatching chooses watchers potentially interested in a given event
// a marks dispatching as true.
func (c *Cacher) startDispatching(event *watchCacheEvent) {
	triggerValues, supported := c.triggerValues(event)

	c.Lock()
	defer c.Unlock()

	c.dispatching = true
	// We are reusing the slice to avoid memory reallocations in every
	// dispatchEvent() call. That may prevent Go GC from freeing items
	// from previous phases that are sitting behind the current length
	// of the slice, but there is only a limited number of those and the
	// gain from avoiding memory allocations is much bigger.
	c.watchersBuffer = c.watchersBuffer[:0]

	if event.Type == watch.Bookmark {
		c.startDispatchingBookmarkEvents()
		// return here to reduce following code indentation and diff
		return
	}

	// Iterate over "allWatchers" no matter what the trigger function is.
	for _, watcher := range c.watchers.allWatchers {
		c.watchersBuffer = append(c.watchersBuffer, watcher)
	}
	if supported {
		// Iterate over watchers interested in the given values of the trigger.
		for _, triggerValue := range triggerValues {
			for _, watcher := range c.watchers.valueWatchers[triggerValue] {
				c.watchersBuffer = append(c.watchersBuffer, watcher)
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
				c.watchersBuffer = append(c.watchersBuffer, watcher)
			}
		}
	}
}

// finishDispatching stops all the watchers that were supposed to be
// stopped in the meantime, but it was deferred to avoid closing input
// channels of watchers, as add() may still have writing to it.
// It also marks dispatching as false.
func (c *Cacher) finishDispatching() {
	c.Lock()
	defer c.Unlock()
	c.dispatching = false
	for _, watcher := range c.watchersToStop {
		watcher.stopThreadUnsafe()
	}
	c.watchersToStop = c.watchersToStop[:0]
}

func (c *Cacher) terminateAllWatchers() {
	c.Lock()
	defer c.Unlock()
	c.watchers.terminateAll(c.objectType, c.stopWatcherThreadUnsafe)
}

func (c *Cacher) stopWatcherThreadUnsafe(watcher *cacheWatcher) {
	if c.dispatching {
		c.watchersToStop = append(c.watchersToStop, watcher)
	} else {
		watcher.stopThreadUnsafe()
	}
}

func (c *Cacher) isStopped() bool {
	c.stopLock.RLock()
	defer c.stopLock.RUnlock()
	return c.stopped
}

// Stop implements the graceful termination.
func (c *Cacher) Stop() {
	c.stopLock.Lock()
	if c.stopped {
		// avoid stopping twice (note: cachers are shared with subresources)
		c.stopLock.Unlock()
		return
	}
	c.stopped = true
	c.stopLock.Unlock()
	close(c.stopCh)
	c.stopWg.Wait()
}

func forgetWatcher(c *Cacher, index int, triggerValue string, triggerSupported bool) func() {
	return func() {
		c.Lock()
		defer c.Unlock()

		// It's possible that the watcher is already not in the structure (e.g. in case of
		// simultaneous Stop() and terminateAllWatchers(), but it is safe to call stopThreadUnsafe()
		// on a watcher multiple times.
		c.watchers.deleteWatcher(index, triggerValue, triggerSupported, c.stopWatcherThreadUnsafe)
	}
}

func filterWithAttrsFunction(key string, p storage.SelectionPredicate) filterWithAttrsFunc {
	filterFunc := func(objKey string, label labels.Set, field fields.Set) bool {
		if !hasPathPrefix(objKey, key) {
			return false
		}
		return p.MatchesObjectAttributes(label, field)
	}
	return filterFunc
}

// LastSyncResourceVersion returns resource version to which the underlying cache is synced.
func (c *Cacher) LastSyncResourceVersion() (uint64, error) {
	c.ready.wait()

	resourceVersion := c.reflector.LastSyncResourceVersion()
	return c.versioner.ParseResourceVersion(resourceVersion)
}

// cacherListerWatcher opaques storage.Interface to expose cache.ListerWatcher.
type cacherListerWatcher struct {
	storage        storage.Interface
	resourcePrefix string
	newListFunc    func() runtime.Object
}

// NewCacherListerWatcher returns a storage.Interface backed ListerWatcher.
func NewCacherListerWatcher(storage storage.Interface, resourcePrefix string, newListFunc func() runtime.Object) cache.ListerWatcher {
	return &cacherListerWatcher{
		storage:        storage,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
	}
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) List(options metav1.ListOptions) (runtime.Object, error) {
	list := lw.newListFunc()
	pred := storage.SelectionPredicate{
		Label:    labels.Everything(),
		Field:    fields.Everything(),
		Limit:    options.Limit,
		Continue: options.Continue,
	}

	if err := lw.storage.List(context.TODO(), lw.resourcePrefix, "", pred, list); err != nil {
		return nil, err
	}
	return list, nil
}

// Implements cache.ListerWatcher interface.
func (lw *cacherListerWatcher) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return lw.storage.WatchList(context.TODO(), lw.resourcePrefix, options.ResourceVersion, storage.Everything)
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

// cacheWatcher implements watch.Interface
// this is not thread-safe
type cacheWatcher struct {
	input     chan *watchCacheEvent
	result    chan watch.Event
	done      chan struct{}
	filter    filterWithAttrsFunc
	stopped   bool
	forget    func()
	versioner storage.Versioner
	// The watcher will be closed by server after the deadline,
	// save it here to send bookmark events before that.
	deadline            time.Time
	allowWatchBookmarks bool
	// Object type of the cache watcher interests
	objectType reflect.Type
}

func newCacheWatcher(chanSize int, filter filterWithAttrsFunc, forget func(), versioner storage.Versioner, deadline time.Time, allowWatchBookmarks bool, objectType reflect.Type) *cacheWatcher {
	return &cacheWatcher{
		input:               make(chan *watchCacheEvent, chanSize),
		result:              make(chan watch.Event, chanSize),
		done:                make(chan struct{}),
		filter:              filter,
		stopped:             false,
		forget:              forget,
		versioner:           versioner,
		deadline:            deadline,
		allowWatchBookmarks: allowWatchBookmarks,
		objectType:          objectType,
	}
}

// Implements watch.Interface.
func (c *cacheWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *cacheWatcher) Stop() {
	c.forget()
}

// we rely on the fact that stopThredUnsafe is actually protected by Cacher.Lock()
func (c *cacheWatcher) stopThreadUnsafe() {
	if !c.stopped {
		c.stopped = true
		close(c.done)
		close(c.input)
	}
}

func (c *cacheWatcher) nonblockingAdd(event *watchCacheEvent) bool {
	select {
	case c.input <- event:
		return true
	default:
		return false
	}
}

// Nil timer means that add will not block (if it can't send event immediately, it will break the watcher)
func (c *cacheWatcher) add(event *watchCacheEvent, timer *time.Timer) bool {
	// Try to send the event immediately, without blocking.
	if c.nonblockingAdd(event) {
		return true
	}

	closeFunc := func() {
		// This means that we couldn't send event to that watcher.
		// Since we don't want to block on it infinitely,
		// we simply terminate it.
		klog.V(1).Infof("Forcing watcher close due to unresponsiveness: %v", reflect.TypeOf(event.Object).String())
		c.forget()
	}

	if timer == nil {
		closeFunc()
		return false
	}

	// OK, block sending, but only until timer fires.
	select {
	case c.input <- event:
		return true
	case <-timer.C:
		closeFunc()
		return false
	}
}

func (c *cacheWatcher) nextBookmarkTime(now time.Time) (time.Time, bool) {
	// For now we return 2s before deadline (and maybe +infinity is now already passed this time)
	// but it gives us extensibility for the future(false when deadline is not set).
	if c.deadline.IsZero() {
		return c.deadline, false
	}
	return c.deadline.Add(-2 * time.Second), true
}

func (c *cacheWatcher) convertToWatchEvent(event *watchCacheEvent) *watch.Event {
	if event.Type == watch.Bookmark {
		return &watch.Event{Type: watch.Bookmark, Object: event.Object.DeepCopyObject()}
	}

	curObjPasses := event.Type != watch.Deleted && c.filter(event.Key, event.ObjLabels, event.ObjFields)
	oldObjPasses := false
	if event.PrevObject != nil {
		oldObjPasses = c.filter(event.Key, event.PrevObjLabels, event.PrevObjFields)
	}
	if !curObjPasses && !oldObjPasses {
		// Watcher is not interested in that object.
		return nil
	}

	switch {
	case curObjPasses && !oldObjPasses:
		return &watch.Event{Type: watch.Added, Object: event.Object.DeepCopyObject()}
	case curObjPasses && oldObjPasses:
		return &watch.Event{Type: watch.Modified, Object: event.Object.DeepCopyObject()}
	case !curObjPasses && oldObjPasses:
		// return a delete event with the previous object content, but with the event's resource version
		oldObj := event.PrevObject.DeepCopyObject()
		if err := c.versioner.UpdateObject(oldObj, event.ResourceVersion); err != nil {
			utilruntime.HandleError(fmt.Errorf("failure to version api object (%d) %#v: %v", event.ResourceVersion, oldObj, err))
		}
		return &watch.Event{Type: watch.Deleted, Object: oldObj}
	}

	return nil
}

// NOTE: sendWatchCacheEvent is assumed to not modify <event> !!!
func (c *cacheWatcher) sendWatchCacheEvent(event *watchCacheEvent) {
	watchEvent := c.convertToWatchEvent(event)
	if watchEvent == nil {
		// Watcher is not interested in that object.
		return
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
	case c.result <- *watchEvent:
	case <-c.done:
	}
}

func (c *cacheWatcher) process(ctx context.Context, initEvents []*watchCacheEvent, resourceVersion uint64) {
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
	objType := c.objectType.String()
	if len(initEvents) > 0 {
		initCounter.WithLabelValues(objType).Add(float64(len(initEvents)))
	}
	processingTime := time.Since(startTime)
	if processingTime > initProcessThreshold {
		klog.V(2).Infof("processing %d initEvents of %s took %v", len(initEvents), objType, processingTime)
	}

	defer close(c.result)
	defer c.Stop()
	for {
		select {
		case event, ok := <-c.input:
			if !ok {
				return
			}
			// only send events newer than resourceVersion
			if event.ResourceVersion > resourceVersion {
				c.sendWatchCacheEvent(event)
			}
		case <-ctx.Done():
			return
		}
	}
}

type ready struct {
	ok bool
	c  *sync.Cond
}

func newReady() *ready {
	return &ready{c: sync.NewCond(&sync.RWMutex{})}
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
	rwMutex := r.c.L.(*sync.RWMutex)
	rwMutex.RLock()
	defer rwMutex.RUnlock()
	return r.ok
}

func (r *ready) set(ok bool) {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	r.ok = ok
	r.c.Broadcast()
}
