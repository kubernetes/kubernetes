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
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/delegator"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/progress"
	"k8s.io/apiserver/pkg/storage/cacher/store"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	// blockTimeout determines how long we're willing to block the request
	// to wait for a given resource version to be propagated to cache,
	// before terminating request and returning Timeout error with retry
	// after suggestion.
	blockTimeout = 3 * time.Second

	// resourceVersionTooHighRetrySeconds is the seconds before a operation should be retried by the client
	// after receiving a 'too high resource version' error.
	resourceVersionTooHighRetrySeconds = 1
)

// watchCacheEvent is a single "watch event" that is send to users of
// watchCache. Additionally to a typical "watch.Event" it contains
// the previous value of the object to enable proper filtering in the
// upper layers.
type watchCacheEvent struct {
	Type            watch.EventType
	Object          runtime.Object
	ObjLabels       labels.Set
	ObjFields       fields.Set
	PrevObject      runtime.Object
	PrevObjLabels   labels.Set
	PrevObjFields   fields.Set
	Key             string
	ResourceVersion uint64
	RecordTime      time.Time
}

// watchCache implements a Store interface.
// However, it depends on the elements implementing runtime.Object interface.
//
// watchCache is a "sliding window" (with a limited capacity) of objects
// observed from a watch.
type watchCache struct {
	sync.RWMutex

	// Condition on which lists are waiting for the fresh enough
	// resource version.
	cond *sync.Cond

	// ResourceVersion up to which the watchCache is propagated.
	resourceVersion uint64

	// This handler is run at the end of every successful Replace() method.
	onReplace func()

	history *watchCacheHistory
	storage *store.WatchCacheStorage

	config *ImmutableWatchCacheConfig
}

type ImmutableWatchCacheConfig struct {
	// keyFunc is used to get a key in the underlying storage for a given object.
	keyFunc func(runtime.Object) (string, error)

	// getAttrsFunc is used to get labels and fields of an object.
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, error)

	// This handler is run at the end of every Add/Update/Delete method
	// and additionally gets the previous value of the object.
	eventHandler func(*watchCacheEvent)

	// for testing timeouts.
	clock clock.Clock

	// An underlying storage.Versioner.
	versioner storage.Versioner

	// cacher's group resource
	groupResource schema.GroupResource

	// For testing cache interval invalidation.
	indexValidator indexValidator

	// Requests progress notification if there are requests waiting for watch
	// to be fresh
	waitingUntilFresh *progress.ConditionalProgressRequester

	getCurrentRV func(context.Context) (uint64, error)
}

func newWatchCache(
	keyFunc func(runtime.Object) (string, error),
	eventHandler func(*watchCacheEvent),
	getAttrsFunc func(runtime.Object) (labels.Set, fields.Set, error),
	versioner storage.Versioner,
	indexers *cache.Indexers,
	clock clock.WithTicker,
	eventFreshDuration time.Duration,
	groupResource schema.GroupResource,
	progressRequester *progress.ConditionalProgressRequester,
	getCurrentRV func(context.Context) (uint64, error),
) *watchCache {
	config := &ImmutableWatchCacheConfig{
		keyFunc:           keyFunc,
		getAttrsFunc:      getAttrsFunc,
		eventHandler:      eventHandler,
		clock:             clock,
		versioner:         versioner,
		groupResource:     groupResource,
		waitingUntilFresh: progressRequester,
		getCurrentRV:      getCurrentRV,
	}

	wc := &watchCache{
		resourceVersion: 0,
		config:          config,
		history:         newWatchCacheHistory(config, eventFreshDuration),
		storage:         store.NewWatchCacheStorage(config.keyFunc, indexers),
	}
	wc.cond = sync.NewCond(wc.RLocker())
	wc.config.indexValidator = wc.history.isIndexValidLocked

	return wc
}

// Add takes runtime.Object as an argument.
func (w *watchCache) Add(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Added, Object: object}

	return w.processEvent(event, resourceVersion)
}

// Update takes runtime.Object as an argument.
func (w *watchCache) Update(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Modified, Object: object}

	return w.processEvent(event, resourceVersion)
}

// Delete takes runtime.Object as an argument.
func (w *watchCache) Delete(obj interface{}) error {
	object, resourceVersion, err := w.objectToVersionedRuntimeObject(obj)
	if err != nil {
		return err
	}
	event := watch.Event{Type: watch.Deleted, Object: object}

	return w.processEvent(event, resourceVersion)
}

func (w *watchCache) objectToVersionedRuntimeObject(obj interface{}) (runtime.Object, uint64, error) {
	object, ok := obj.(runtime.Object)
	if !ok {
		return nil, 0, fmt.Errorf("obj does not implement runtime.Object interface: %v", obj)
	}
	resourceVersion, err := w.config.versioner.ObjectResourceVersion(object)
	if err != nil {
		return nil, 0, err
	}
	return object, resourceVersion, nil
}

// processEvent is safe as long as there is at most one call to it in flight
// at any point in time.
func (w *watchCache) processEvent(event watch.Event, resourceVersion uint64) error {
	metrics.EventsReceivedCounter.WithLabelValues(w.config.groupResource.Group, w.config.groupResource.Resource).Inc()

	key, err := w.config.keyFunc(event.Object)
	if err != nil {
		return fmt.Errorf("couldn't compute key: %v", err)
	}
	elem := &store.Element{Key: key, Object: event.Object}
	elem.Labels, elem.Fields, err = w.config.getAttrsFunc(event.Object)
	if err != nil {
		return err
	}

	wcEvent := &watchCacheEvent{
		Type:            event.Type,
		Object:          elem.Object,
		ObjLabels:       elem.Labels,
		ObjFields:       elem.Fields,
		Key:             key,
		ResourceVersion: resourceVersion,
		RecordTime:      w.config.clock.Now(),
	}

	// We can call w.storage.Get() outside of a critical section,
	// because the w.storage itself is thread-safe and the only
	// place where it is modified is below (via UpdateStoreLocked)
	// and these calls are serialized because reflector is processing
	// events one-by-one.
	previous, exists, err := w.storage.Get(event.Object)
	if err != nil {
		return err
	}
	if exists {
		previousElem := previous.(*store.Element)
		wcEvent.PrevObject = previousElem.Object
		wcEvent.PrevObjLabels = previousElem.Labels
		wcEvent.PrevObjFields = previousElem.Fields
	}

	if err := func() error {
		w.Lock()
		defer w.Unlock()

		w.history.updateCache(wcEvent)
		w.resourceVersion = resourceVersion
		defer w.cond.Broadcast()

		if w.history.isCacheFullLocked() {
			oldestRV := w.history.OldestResourceVersionLocked()
			w.storage.CompactSnapshotsLocked(oldestRV)
		}
		if err := w.storage.UpdateStoreLocked(event.Type, elem, resourceVersion); err != nil {
			return err
		}
		return nil
	}(); err != nil {
		return err
	}

	// Avoid calling event handler under lock.
	// This is safe as long as there is at most one call to Add/Update/Delete and
	// UpdateResourceVersion in flight at any point in time, which is true now,
	// because reflector calls them synchronously from its main thread.
	if w.config.eventHandler != nil {
		w.config.eventHandler(wcEvent)
	}
	metrics.RecordResourceVersion(w.config.groupResource, resourceVersion)
	return nil
}

func (w *watchCache) UpdateResourceVersion(resourceVersion string) {
	rv, err := w.config.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		klog.Errorf("Couldn't parse resourceVersion: %v", err)
		return
	}

	func() {
		w.Lock()
		defer w.Unlock()
		w.resourceVersion = rv
		w.cond.Broadcast()
	}()

	// Avoid calling event handler under lock.
	// This is safe as long as there is at most one call to Add/Update/Delete and
	// UpdateResourceVersion in flight at any point in time, which is true now,
	// because reflector calls them synchronously from its main thread.
	if w.config.eventHandler != nil {
		wcEvent := &watchCacheEvent{
			Type:            watch.Bookmark,
			ResourceVersion: rv,
		}
		w.config.eventHandler(wcEvent)
	}
	metrics.RecordResourceVersion(w.config.groupResource, rv)
}

// waitUntilFreshLocked waits until cache is at least as fresh as given resourceVersion.
func (w *watchCache) waitUntilFreshLocked(ctx context.Context, consistentReadSupported bool, resourceVersion uint64) error {
	if resourceVersion == 0 || resourceVersion <= w.resourceVersion {
		return nil
	}
	if consistentReadSupported {
		w.config.waitingUntilFresh.Add()
		defer w.config.waitingUntilFresh.Remove()
	}
	startTime := w.config.clock.Now()
	defer func() {
		if resourceVersion > 0 {
			metrics.WatchCacheReadWait.WithContext(ctx).WithLabelValues(w.config.groupResource.Group, w.config.groupResource.Resource).Observe(w.config.clock.Since(startTime).Seconds())
		}
	}()

	// In case resourceVersion is 0, we accept arbitrarily stale result.
	// As a result, the condition in the below for loop will never be
	// satisfied (w.resourceVersion is never negative), this call will
	// never hit the w.cond.Wait().
	// As a result - we can optimize the code by not firing the wakeup
	// function (and avoid starting a gorotuine), especially given that
	// resourceVersion=0 is the most common case.
	if resourceVersion > 0 {
		go func() {
			// Wake us up when the time limit has expired.  The docs
			// promise that time.After (well, NewTimer, which it calls)
			// will wait *at least* the duration given. Since this go
			// routine starts sometime after we record the start time, and
			// it will wake up the loop below sometime after the broadcast,
			// we don't need to worry about waking it up before the time
			// has expired accidentally.
			<-w.config.clock.After(blockTimeout)
			w.cond.Broadcast()
		}()
	}

	span := tracing.SpanFromContext(ctx)
	span.AddEvent("watchCache locked acquired")
	for w.resourceVersion < resourceVersion {
		if w.config.clock.Since(startTime) >= blockTimeout {
			// Request that the client retry after 'resourceVersionTooHighRetrySeconds' seconds.
			return storage.NewTooLargeResourceVersionError(resourceVersion, w.resourceVersion, resourceVersionTooHighRetrySeconds)
		}
		w.cond.Wait()
	}
	span.AddEvent("watchCache fresh enough")
	return nil
}

func (c *watchCache) WaitUntilFreshAndGetList(ctx context.Context, key string, opts storage.ListOptions) (listResp, string, error) {
	if opts.Recursive {
		return c.waitUntilFreshAndList(ctx, key, opts)
	}
	return c.waitUntilFreshAndGetList(ctx, key, opts)
}

func (c *watchCache) waitUntilFreshAndGetList(ctx context.Context, key string, opts storage.ListOptions) (listResp, string, error) {
	var listRV uint64
	var err error
	if opts.ResourceVersionMatch == "" && opts.ResourceVersion == "" {
		// Consistent read
		listRV, err = c.config.getCurrentRV(ctx)
		if err != nil {
			return listResp{}, "", err
		}
	} else {
		listRV, err = c.config.versioner.ParseResourceVersion(opts.ResourceVersion)
		if err != nil {
			return listResp{}, "", err
		}
	}
	obj, exists, readResourceVersion, err := c.WaitUntilFreshAndGet(ctx, listRV, key)
	if err != nil {
		return listResp{}, "", err
	}
	if exists {
		return listResp{Items: []interface{}{obj}, ResourceVersion: readResourceVersion}, "", nil
	}
	return listResp{ResourceVersion: readResourceVersion}, "", nil
}

// WaitUntilFreshAndList returns list of pointers to `storeElement` objects along
// with their ResourceVersion and the name of the index, if any, that was used.
func (w *watchCache) WaitUntilFreshAndGetKeys(ctx context.Context, resourceVersion uint64) (keys []string, err error) {
	consistentReadSupported := delegator.ConsistentReadSupported()
	w.RLock()
	defer w.RUnlock()
	err = w.waitUntilFreshLocked(ctx, consistentReadSupported, resourceVersion)
	if err != nil {
		return nil, err
	}
	return w.storage.ListKeys(), nil
}

// NOTICE: Structure follows the shouldDelegateList function in
// staging/src/k8s.io/apiserver/pkg/storage/cacher/delegator.go
func (w *watchCache) waitUntilFreshAndList(ctx context.Context, key string, opts storage.ListOptions) (resp listResp, index string, err error) {
	listRV, err := w.config.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return listResp{}, "", err
	}
	switch opts.ResourceVersionMatch {
	case metav1.ResourceVersionMatchExact:
		return w.waitAndListExactRV(ctx, key, "", listRV)
	case metav1.ResourceVersionMatchNotOlderThan:
	case "":
		// Continue
		if len(opts.Predicate.Continue) > 0 {
			continueKey, continueRV, err := storage.DecodeContinue(opts.Predicate.Continue, key)
			if err != nil {
				return listResp{}, "", errors.NewBadRequest(fmt.Sprintf("invalid continue token: %v", err))
			}
			if continueRV > 0 {
				return w.waitAndListExactRV(ctx, key, continueKey, uint64(continueRV))
			} else {
				// Don't pass matchValues as they don't support continueKey
				return w.waitAndListConsistent(ctx, key, continueKey, nil)
			}
		}
		// Legacy exact match
		if opts.Predicate.Limit > 0 && len(opts.ResourceVersion) > 0 && opts.ResourceVersion != "0" {
			return w.waitAndListExactRV(ctx, key, "", listRV)
		}
		if opts.ResourceVersion == "" {
			return w.waitAndListConsistent(ctx, key, "", opts.Predicate.MatcherIndex(ctx))
		}
	}
	return w.waitAndListLatestRV(ctx, listRV, key, "", opts.Predicate.MatcherIndex(ctx))
}

func (w *watchCache) waitAndListExactRV(ctx context.Context, key, continueKey string, resourceVersion uint64) (resp listResp, index string, err error) {
	store, err := w.waitAndGetExactSnapshot(ctx, resourceVersion)
	if err != nil {
		return listResp{}, "", err
	}
	items, err := store.OrderedListPrefix(key, continueKey)
	return listResp{
		Items:           items,
		ResourceVersion: resourceVersion,
	}, "", err
}

func (w *watchCache) waitAndGetExactSnapshot(ctx context.Context, resourceVersion uint64) (store store.Snapshot, err error) {
	consistentReadSupported := delegator.ConsistentReadSupported()
	w.RLock()
	defer w.RUnlock()
	err = w.waitUntilFreshLocked(ctx, consistentReadSupported, resourceVersion)
	if err != nil {
		return nil, err
	}

	return w.storage.GetExactSnapshotLocked(resourceVersion)
}

func (w *watchCache) waitAndListConsistent(ctx context.Context, key, continueKey string, matchValues []storage.MatchValue) (resp listResp, index string, err error) {
	resourceVersion, err := w.config.getCurrentRV(ctx)
	if err != nil {
		return listResp{}, "", err
	}
	return w.waitAndListLatestRV(ctx, resourceVersion, key, continueKey, matchValues)
}

func (w *watchCache) waitAndListLatestRV(ctx context.Context, minResourceVersion uint64, key, continueKey string, matchValues []storage.MatchValue) (resp listResp, index string, err error) {
	snap, resourceVersion, index, err := w.waitAndGetLatestSnapshot(ctx, minResourceVersion, key, continueKey, matchValues)
	if err != nil {
		return listResp{}, "", err
	}
	items, err := snap.OrderedListPrefix(key, continueKey)
	if err != nil {
		return listResp{}, "", err
	}
	return listResp{
		Items:           items,
		ResourceVersion: resourceVersion,
	}, index, nil
}

func (w *watchCache) waitAndGetLatestSnapshot(ctx context.Context, minResourceVersion uint64, key, continueKey string, matchValues []storage.MatchValue) (snap store.Snapshot, resourceVersion uint64, index string, err error) {
	consistentReadSupported := delegator.ConsistentReadSupported()
	w.RLock()
	defer w.RUnlock()
	err = w.waitUntilFreshLocked(ctx, consistentReadSupported, minResourceVersion)
	if err != nil {
		return nil, 0, "", err
	}
	// This isn't the place where we do "final filtering" - only some "prefiltering" is happening here. So the only
	// requirement here is to NOT miss anything that should be returned. We can return as many non-matching items as we
	// want - they will be filtered out later. The fact that we return less things is only further performance improvement.
	// TODO: if multiple indexes match, return the one with the fewest items, so as to do as much filtering as possible.
	for _, matchValue := range matchValues {
		if snap, err := w.storage.GetByIndexSnapshot(matchValue.IndexName, matchValue.Value); err == nil {
			return snap, w.resourceVersion, matchValue.IndexName, nil
		}
	}
	snap, err = w.storage.GetLatestSnapshotOrBuildLocked(key, continueKey)
	return snap, w.resourceVersion, "", err
}

func (w *watchCache) notFresh(resourceVersion uint64) bool {
	w.RLock()
	defer w.RUnlock()
	return resourceVersion > w.resourceVersion
}

// WaitUntilFreshAndGet returns a pointers to <storeElement> object.
func (w *watchCache) WaitUntilFreshAndGet(ctx context.Context, resourceVersion uint64, key string) (interface{}, bool, uint64, error) {
	consistentReadSupported := delegator.ConsistentReadSupported()
	w.RLock()
	defer w.RUnlock()
	err := w.waitUntilFreshLocked(ctx, consistentReadSupported, resourceVersion)
	if err != nil {
		return nil, false, 0, err
	}
	value, exists, err := w.storage.GetByKey(key)
	return value, exists, w.resourceVersion, err
}

// Replace takes slice of runtime.Object as a parameter.
func (w *watchCache) Replace(objs []interface{}, resourceVersion string) error {
	version, err := w.config.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	toReplace := make([]interface{}, 0, len(objs))
	for _, obj := range objs {
		object, ok := obj.(runtime.Object)
		if !ok {
			return fmt.Errorf("didn't get runtime.Object for replace: %#v", obj)
		}
		key, err := w.config.keyFunc(object)
		if err != nil {
			return fmt.Errorf("couldn't compute key: %v", err)
		}
		objLabels, objFields, err := w.config.getAttrsFunc(object)
		if err != nil {
			return err
		}
		toReplace = append(toReplace, &store.Element{
			Key:    key,
			Object: object,
			Labels: objLabels,
			Fields: objFields,
		})
	}

	w.Lock()
	defer w.Unlock()

	// Ensure startIndex never decreases, so that existing watchCacheInterval
	// instances get "invalid" errors if the try to download from the buffer
	// using their own start/end indexes calculated from previous buffer
	// content.

	// Empty the cyclic buffer, ensuring startIndex doesn't decrease.
	w.history.ResetLocked()

	if err := w.storage.ReplaceLocked(toReplace, resourceVersion, version); err != nil {
		return err
	}
	w.resourceVersion = version
	if w.onReplace != nil {
		w.onReplace()
	}
	w.cond.Broadcast()

	metrics.RecordResourceVersion(w.config.groupResource, version)
	klog.V(3).Infof("Replaced watchCache (rev: %v) ", resourceVersion)
	return nil
}

func (w *watchCache) SetOnReplace(onReplace func()) {
	w.Lock()
	defer w.Unlock()
	w.onReplace = onReplace
}

func (w *watchCache) Resync() error {
	// Nothing to do
	return nil
}

func (w *watchCache) getListResourceVersion() uint64 {
	w.RLock()
	defer w.RUnlock()
	return w.storage.ListResourceVersion()
}

func (w *watchCache) suggestedWatchChannelSize(indexExists, triggerUsed bool) int {
	w.RLock()
	defer w.RUnlock()
	return w.history.suggestedWatchChannelSize(indexExists, triggerUsed)
}

// getAllEventsSinceLocked returns a watchCacheInterval that can be used to
// retrieve events since a certain resourceVersion. This function assumes to
// be called under the watchCache lock.
func (w *watchCache) getAllEventsSinceLocked(resourceVersion uint64, key string, opts storage.ListOptions) (*watchCacheInterval, error) {
	_, matchesSingle := opts.Predicate.MatchesSingle()
	matchesSingle = matchesSingle && !opts.Recursive
	if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
		return w.getIntervalFromStoreLocked(key, matchesSingle)
	}

	if resourceVersion == 0 {
		if opts.SendInitialEvents == nil {
			// resourceVersion = 0 means that we don't require any specific starting point
			// and we would like to start watching from ~now.
			// However, to keep backward compatibility, we additionally need to return the
			// current state and only then start watching from that point.
			//
			// TODO: In v2 api, we should stop returning the current state - #13969.
			return w.getIntervalFromStoreLocked(key, matchesSingle)
		}
		// SendInitialEvents = false and resourceVersion = 0
		// means that the request would like to start watching
		// from Any resourceVersion
		resourceVersion = w.resourceVersion
	}

	return w.history.GetIntervalLocked(resourceVersion, w.storage.ListResourceVersion(), w.RWMutex.RLocker())
}

// getIntervalFromStoreLocked returns a watchCacheInterval
// that covers the entire storage state.
// This function assumes to be called under the watchCache lock.
func (w *watchCache) getIntervalFromStoreLocked(key string, matchesSingle bool) (*watchCacheInterval, error) {
	// When not matching a single key, an immutable snapshot lets us
	// defer the O(N) interval build off the watchCache lock.
	if !matchesSingle {
		if snapshot, ok := w.storage.LatestSnapshotLocked(); ok {
			return newCacheIntervalFromLazySnapshot(w.resourceVersion, snapshot), nil
		}
	}
	return newCacheIntervalFromStore(w.resourceVersion, w.storage.StoreLocked(), key, matchesSingle)
}
