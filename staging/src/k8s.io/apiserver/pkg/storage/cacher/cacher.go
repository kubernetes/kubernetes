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
	"cmp"
	"context"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/grpc/metadata"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/delegator"
	"k8s.io/apiserver/pkg/storage/cacher/key"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/progress"
	"k8s.io/apiserver/pkg/storage/cacher/store"
	etcdfeature "k8s.io/apiserver/pkg/storage/feature"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

var (
	emptyFunc             = func(bool) {}
	coreNamespaceResource = schema.GroupResource{Group: "", Resource: "namespaces"}
)

const (
	// storageWatchListPageSize is the cacher's request chunk size of
	// initial and resync watch lists to storage.
	storageWatchListPageSize = int64(10000)

	// DefaultEventFreshDuration is the default time duration of events
	// we want to keep.
	// We set it to defaultBookmarkFrequency plus epsilon to maximize
	// chances that last bookmark was sent within kept history, at the
	// same time, minimizing the needed memory usage.
	DefaultEventFreshDuration = defaultBookmarkFrequency + 15*time.Second

	// defaultBookmarkFrequency defines how frequently watch bookmarks should be send
	// in addition to sending a bookmark right before watch deadline.
	defaultBookmarkFrequency = time.Minute
)

// Config contains the configuration for a given Cache.
type Config struct {
	// An underlying storage.Interface.
	Storage storage.Interface

	// An underlying storage.Versioner.
	Versioner storage.Versioner

	// The GroupResource the cacher is caching. Used for disambiguating *unstructured.Unstructured (CRDs) in logging
	// and metrics.
	GroupResource schema.GroupResource

	// EventsHistoryWindow specifies minimum history duration that storage is keeping.
	// If lower than DefaultEventFreshDuration, the cache creation will fail.
	EventsHistoryWindow time.Duration

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

	// Indexers is used to accelerate the list operation, falls back to regular list
	// operation if no indexer found.
	Indexers *cache.Indexers

	// NewFunc is a function that creates new empty object storing a object of type Type.
	NewFunc func() runtime.Object

	// NewList is a function that creates new empty object storing a list of
	// objects of type Type.
	NewListFunc func() runtime.Object

	Codec runtime.Codec

	Clock clock.WithTicker
}

type watchersMap map[int]*cacheWatcher

func (wm watchersMap) addWatcher(w *cacheWatcher, number int) {
	wm[number] = w
}

func (wm watchersMap) deleteWatcher(number int) {
	delete(wm, number)
}

func (wm watchersMap) terminateAll(done func(*cacheWatcher)) {
	for key, watcher := range wm {
		delete(wm, key)
		done(watcher)
	}
}

type indexedWatchers struct {
	allWatchers   map[namespacedName]watchersMap
	valueWatchers map[string]watchersMap
}

func (i *indexedWatchers) addWatcher(w *cacheWatcher, number int, scope namespacedName, value string, supported bool) {
	if supported {
		if _, ok := i.valueWatchers[value]; !ok {
			i.valueWatchers[value] = watchersMap{}
		}
		i.valueWatchers[value].addWatcher(w, number)
	} else {
		scopedWatchers, ok := i.allWatchers[scope]
		if !ok {
			scopedWatchers = watchersMap{}
			i.allWatchers[scope] = scopedWatchers
		}
		scopedWatchers.addWatcher(w, number)
	}
}

func (i *indexedWatchers) deleteWatcher(number int, scope namespacedName, value string, supported bool) {
	if supported {
		i.valueWatchers[value].deleteWatcher(number)
		if len(i.valueWatchers[value]) == 0 {
			delete(i.valueWatchers, value)
		}
	} else {
		i.allWatchers[scope].deleteWatcher(number)
		if len(i.allWatchers[scope]) == 0 {
			delete(i.allWatchers, scope)
		}
	}
}

func (i *indexedWatchers) terminateAll(groupResource schema.GroupResource, done func(*cacheWatcher)) {
	// note that we don't have to call setDrainInputBufferLocked method on the watchers
	// because we take advantage of the default value - stop immediately
	// also watchers that have had already its draining strategy set
	// are no longer available (they were removed from the allWatchers and the valueWatchers maps)
	if len(i.allWatchers) > 0 || len(i.valueWatchers) > 0 {
		klog.Warningf("Terminating all watchers from cacher %v", groupResource)
	}
	for _, watchers := range i.allWatchers {
		watchers.terminateAll(done)
	}
	for _, watchers := range i.valueWatchers {
		watchers.terminateAll(done)
	}
	i.allWatchers = map[namespacedName]watchersMap{}
	i.valueWatchers = map[string]watchersMap{}
}

// As we don't need a high precision here, we keep all watchers timeout within a
// second in a bucket, and pop up them once at the timeout. To be more specific,
// if you set fire time at X, you can get the bookmark within (X-1,X+1) period.
type watcherBookmarkTimeBuckets struct {
	// the key of watcherBuckets is the number of seconds since createTime
	watchersBuckets   map[int64][]*cacheWatcher
	createTime        time.Time
	startBucketID     int64
	clock             clock.Clock
	bookmarkFrequency time.Duration
}

func newTimeBucketWatchers(clock clock.Clock, bookmarkFrequency time.Duration) *watcherBookmarkTimeBuckets {
	return &watcherBookmarkTimeBuckets{
		watchersBuckets:   make(map[int64][]*cacheWatcher),
		createTime:        clock.Now(),
		startBucketID:     0,
		clock:             clock,
		bookmarkFrequency: bookmarkFrequency,
	}
}

// adds a watcher to the bucket, if the deadline is before the start, it will be
// added to the first one.
func (t *watcherBookmarkTimeBuckets) addWatcherThreadUnsafe(w *cacheWatcher) bool {
	// note that the returned time can be before t.createTime,
	// especially in cases when the nextBookmarkTime method
	// give us the zero value of type Time
	// so buckedID can hold a negative value
	nextTime, ok := w.nextBookmarkTime(t.clock.Now(), t.bookmarkFrequency)
	if !ok {
		return false
	}
	bucketID := int64(nextTime.Sub(t.createTime) / time.Second)
	if bucketID < t.startBucketID {
		bucketID = t.startBucketID
	}
	watchers := t.watchersBuckets[bucketID]
	t.watchersBuckets[bucketID] = append(watchers, w)
	return true
}

func (t *watcherBookmarkTimeBuckets) popExpiredWatchersThreadUnsafe() [][]*cacheWatcher {
	currentBucketID := int64(t.clock.Since(t.createTime) / time.Second)
	// There should be one or two elements in almost all cases
	expiredWatchers := make([][]*cacheWatcher, 0, 2)
	for ; t.startBucketID <= currentBucketID; t.startBucketID++ {
		if watchers, ok := t.watchersBuckets[t.startBucketID]; ok {
			delete(t.watchersBuckets, t.startBucketID)
			expiredWatchers = append(expiredWatchers, watchers)
		}
	}
	return expiredWatchers
}

type filterWithAttrsFunc func(key string, l labels.Set, f fields.Set, obj runtime.Object) bool

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

	resourcePrefix string

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
	// Used for logging, to disambiguate *unstructured.Unstructured (CRDs)
	groupResource schema.GroupResource

	// "sliding window" of recent changes of objects and the current state.
	watchCache *watchCache
	reflector  *cache.Reflector

	// Versioner is used to handle resource versions.
	versioner storage.Versioner

	// newFunc is a function that creates new empty object storing a object of type Type.
	newFunc func() runtime.Object

	// newListFunc is a function that creates new empty list for storing objects of type Type.
	newListFunc func() runtime.Object

	// indexedTrigger is used for optimizing amount of watchers that needs to process
	// an incoming event.
	indexedTrigger *indexedTriggerFunc
	// watchers is mapping from the value of trigger function that a
	// watcher is interested into the watchers
	watcherIdx int
	watchers   indexedWatchers

	// Defines a time budget that can be spend on waiting for not-ready watchers
	// while dispatching event before shutting them down.
	dispatchTimeoutBudget timeBudget

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
	// Note that this field when accessed MUST be protected by the Cacher.lock.
	bookmarkWatchers *watcherBookmarkTimeBuckets
	// expiredBookmarkWatchers is a list of watchers that were expired and need to be schedule for a next bookmark event
	expiredBookmarkWatchers []*cacheWatcher
	compactor               *compactor
	watcherMetrics          *metrics.WatcherMetricsObservers

	// stall is non-nil exactly when the WatchCacheStallResume feature gate
	// is enabled (read once at construction): a watcher whose input channel
	// fills up becomes unsynced and is served from the watch cache history
	// by the dispatcher's sync passes instead of being terminated. When
	// nil, dispatch behaves exactly as before the gate existed.
	stall *cacherStall
}

const (
	// syncScanBudget bounds the history events one sync pass reads, so
	// catch-up work delays live dispatch by a bounded amount; 2000 was
	// chosen so that one pass with 512 members costs about 1 ms.
	syncScanBudget = 2000
	// syncPushBudget bounds the events one pass pushes, for the same
	// reason: 512 draining members in one cohort would otherwise take up
	// to 512 times cap(input) pushes in one pass. It equals the scan
	// budget, so a pass pushes at most about what it can scan; the busy
	// period follows the pass duration, so the bound sets the slice, not
	// the throughput. The bound is soft by one cohort's width, because a
	// scanned event is offered to every cohort member or to none, plus
	// the lead's room, because the lead is served past the budget (see
	// serveSyncCohort).
	syncPushBudget = syncScanBudget
	// maxWatchersPerSync bounds the unsynced watchers one pass serves; it
	// is etcd's maxWatchersPerSync (server/storage/mvcc/watchable_store.go).
	maxWatchersPerSync = 512
	// syncIntervalOpenCost is what a cohort's interval open costs in scan
	// budget units: an open takes the watch cache read lock and allocates
	// a 100 event buffer, so it is priced like a short scan. It bounds the
	// intervals one pass opens, so many members with tiny windows cannot
	// multiply the per interval cost; the cursor carries fairness across
	// passes.
	syncIntervalOpenCost = 64
	// syncPushRetryRounds is how many rounds the end of a pass re-offers
	// the owed events to a blocked eager member whose input drained during
	// the pass, so a client that drains fast absorbs more than cap(input)
	// events per pass; it also sizes the owed events a blocked member
	// records, syncPushRetryRounds times cap(input).
	syncPushRetryRounds = 2
	// syncPassPeriod is the pass cadence while any watcher is unsynced;
	// syncPassIdlePeriod takes over after a pass that served nothing.
	// syncPassBusyPeriod is the floor of the period after a pass that left
	// a member owed events with a full input: that client is draining, and
	// a fixed period would cap its catch-up at cap(input) events per
	// period (10 per 10 ms for a trigger-indexed watcher), the same yield
	// after progress etcd's syncWatchersLoop makes. The busy period grows
	// with the measured pass duration, syncPassBusyDutyFactor times it, so
	// passes take at most about a tenth of the dispatcher's time, and is
	// capped at syncPassPeriod so that a pass descheduled by the host
	// cannot amplify into a long silence.
	syncPassPeriod         = 10 * time.Millisecond
	syncPassIdlePeriod     = 100 * time.Millisecond
	syncPassBusyPeriod     = time.Millisecond
	syncPassBusyDutyFactor = 9
)

// cacherStall is the per-Cacher stall-and-resume state; a Cacher holds one
// exactly when the WatchCacheStallResume gate is on, making the nil check
// the single mode representation. Everything but metrics and src is owned
// by the dispatcher goroutine.
type cacherStall struct {
	// metrics holds the pre-resolved metric children shared by all
	// watchers of this Cacher.
	metrics *metrics.StallResumeObservers
	// src serves catch-up intervals from the watch cache event history.
	src *historyCatchUp

	// unsynced holds the watchers the live path skips; sync passes serve
	// them until they catch up, expire or stop.
	unsynced map[*cacheWatcher]struct{}
	// syncCursor is where the last pass stopped scanning; the next pass
	// prefers a lead at or above it so one lead keeps tracking the history.
	syncCursor uint64
	// passCount numbers the passes, from 1, so a watcher's servedPass
	// tells whether the previous pass served it.
	passCount uint64

	// stalled is set by dispatchEvent when a watcher goes unsynced so the
	// scheduler returns to the short pass period.
	stalled    bool
	passPeriod time.Duration
	passTimer  clock.Timer
	passArmed  bool
	lastPass   time.Time
	// dispatcherHook runs a test function on the dispatcher goroutine, handing
	// it a func that runs one sync pass; tests drive passes and read the
	// dispatcher-owned state through it. Tests only.
	dispatcherHook chan func(runPass syncPassFunc)

	// Scratch space reused across passes.
	members    []syncMember
	scanned    []scannedEvent
	expiredBuf []*cacheWatcher
	// byScope, byTrigger and triggerFanout are the pass's lookup maps
	// over the candidates, built once per pass in sorted order; each
	// entry is a member index. lead is the cohort being served.
	byScope       map[namespacedName][]int
	byTrigger     map[string][]int
	triggerFanout []int
	lead          int
	// blockedCount counts the members of the cohort being served whose
	// input filled; pendingOpen counts the eager ones among them that can
	// still record owed events for the retry rounds. The scan stops once
	// every active member is blocked and none can record more.
	blockedCount int
	pendingOpen  int
	pushBudget   int
	// cut is set once the push budget ran out during the cohort being
	// served: the scan then goes on for the lead alone.
	cut    bool
	pushed int
}

// syncPassFunc runs one sync pass and reports how many members it served
// and how many it expired.
type syncPassFunc func() (served, expired int)

// syncMember is a candidate of one sync pass: an unsynced watcher with room
// in its input, sorted by position.
type syncMember struct {
	w *cacheWatcher
	// startPosition is the position at the start of the pass; a scanned
	// event is offered only if it is above it. room and eager are the
	// lead choice's keys, taken at the start of the pass: the free share
	// of the input in permille (a share, so watchers with different
	// input capacities compare by how far they drained, not by slots),
	// and whether the member is not served since it went unsynced or was
	// judged a fast drainer after its last service.
	startPosition uint64
	room          int
	eager         bool
	served        bool
	// blocked is set once a push failed in the member's cohort; pending
	// then collects the scanned events still owed, up to what the retry
	// rounds can push; pendingTruncated records that more were owed.
	blocked          bool
	pending          []int
	pendingTruncated bool
	// resync is set when the member accepted everything it was offered and
	// its cohort read the history to its end.
	resync bool
}

// scannedEvent is one history event read by a sync pass with its selection
// keys computed once; wrapped is the shared dispatch copy, made lazily.
type scannedEvent struct {
	raw              *watchCacheEvent
	wrapped          *watchCacheEvent
	namespace, name  string
	triggerValues    []string
	triggerSupported bool
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

	if config.Clock == nil {
		config.Clock = clock.RealClock{}
	}
	objType := reflect.TypeOf(obj)
	resourcePrefix := config.ResourcePrefix
	if resourcePrefix == "" {
		return nil, fmt.Errorf("resourcePrefix cannot be empty")
	}
	if resourcePrefix == "/" {
		return nil, fmt.Errorf("resourcePrefix cannot be /")
	}
	if !strings.HasPrefix(resourcePrefix, "/") {
		return nil, fmt.Errorf("resourcePrefix needs to start from /")
	}
	cacher := &Cacher{
		resourcePrefix: resourcePrefix,
		ready:          newReady(config.Clock),
		storage:        config.Storage,
		objectType:     objType,
		groupResource:  config.GroupResource,
		versioner:      config.Versioner,
		newFunc:        config.NewFunc,
		newListFunc:    config.NewListFunc,
		indexedTrigger: indexedTrigger,
		watcherIdx:     0,
		watchers: indexedWatchers{
			allWatchers:   make(map[namespacedName]watchersMap),
			valueWatchers: make(map[string]watchersMap),
		},
		// TODO: Figure out the correct value for the buffer size.
		incoming:              make(chan watchCacheEvent, 100),
		dispatchTimeoutBudget: newTimeBudget(),
		// We need to (potentially) stop both:
		// - wait.Until go-routine
		// - reflector.ListAndWatch
		// and there are no guarantees on the order that they will stop.
		// So we will be simply closing the channel, and synchronizing on the WaitGroup.
		stopCh:           stopCh,
		clock:            config.Clock,
		timer:            time.NewTimer(time.Duration(0)),
		bookmarkWatchers: newTimeBucketWatchers(config.Clock, defaultBookmarkFrequency),
		watcherMetrics:   metrics.NewWatcherMetricsObservers(config.GroupResource),
	}

	// Ensure that timer is stopped.
	if !cacher.timer.Stop() {
		// Consume triggered (but not yet received) timer event
		// so that future reuse does not get a spurious timeout.
		<-cacher.timer.C
	}
	var contextMetadata metadata.MD
	if utilfeature.DefaultFeatureGate.Enabled(features.SeparateCacheWatchRPC) {
		// Add grpc context metadata to watch and progress notify requests done by cacher to:
		// * Prevent starvation of watch opened by cacher, by moving it to separate Watch RPC than watch request that bypass cacher.
		// * Ensure that progress notification requests are executed on the same Watch RPC as their watch, which is required for it to work.
		contextMetadata = metadata.New(map[string]string{"source": "cache"})
	}

	eventFreshDuration := config.EventsHistoryWindow
	if eventFreshDuration < DefaultEventFreshDuration {
		return nil, fmt.Errorf("config.EventsHistoryWindow (%v) must not be below %v", eventFreshDuration, DefaultEventFreshDuration)
	}

	progressRequester := progress.NewConditionalProgressRequester(config.Storage.RequestWatchProgress, config.Clock, contextMetadata)
	watchCache := newWatchCache(
		config.KeyFunc, cacher.processEvent, config.GetAttrsFunc, config.Versioner, config.Indexers,
		config.Clock, eventFreshDuration, config.GroupResource, progressRequester, config.Storage.GetCurrentResourceVersion)
	listerWatcher := NewListerWatcher(config.Storage, resourcePrefix, config.NewListFunc, contextMetadata)
	reflectorName := "storage/cacher.go:" + resourcePrefix

	reflector := cache.NewNamedReflector(reflectorName, listerWatcher, nil, watchCache, 0)
	// Configure reflector's pager to for an appropriate pagination chunk size for fetching data from
	// storage. The pager falls back to full list if paginated list calls fail due to an "Expired" error.
	reflector.WatchListPageSize = storageWatchListPageSize
	// When etcd loses leader for 3 cycles, it returns error "no leader".
	// We don't want to terminate all watchers as recreating all watchers puts high load on api-server.
	// In most of the cases, leader is reelected within few cycles.
	reflector.MaxInternalErrorRetryDuration = time.Second * 30

	cacher.watchCache = watchCache
	cacher.reflector = reflector
	if utilfeature.DefaultFeatureGate.Enabled(features.WatchCacheStallResume) {
		cacher.stall = &cacherStall{
			metrics:        metrics.NewStallResumeObservers(config.GroupResource),
			src:            &historyCatchUp{cache: watchCache},
			unsynced:       map[*cacheWatcher]struct{}{},
			passPeriod:     syncPassPeriod,
			dispatcherHook: make(chan func(syncPassFunc)),
			byScope:        map[namespacedName][]int{},
			byTrigger:      map[string][]int{},
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.SizeBasedListCostEstimate) {
		err := config.Storage.EnableResourceSizeEstimation(cacher.getKeys)
		if err != nil {
			return nil, fmt.Errorf("failed to enable resource size estimation: %w", err)
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
		cacher.compactor = newCompactor(config.Storage, watchCache, config.Clock)
		go cacher.compactor.Run(stopCh)
	}

	go cacher.dispatchEvents()
	go progressRequester.Run(stopCh)

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
	startTime := time.Now()
	c.watchCache.SetOnReplace(func() {
		c.ready.setReady()
		duration := time.Since(startTime)
		klog.V(1).InfoS("cacher initialized", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "duration", duration)
		metrics.WatchCacheInitializations.WithLabelValues(c.groupResource.Group, c.groupResource.Resource).Inc()
		metrics.WatchCacheInitializationDuration.WithLabelValues(c.groupResource.Group, c.groupResource.Resource).Observe(duration.Seconds())
	})
	var err error
	defer func() {
		c.ready.setError(err)
	}()

	c.terminateAllWatchers()
	err = c.reflector.ListAndWatch(stopChannel)
	if err != nil {
		klog.Errorf("cacher (%v): unexpected ListAndWatch error: %v; reinitializing...", c.groupResource.String(), err)
		metrics.WatchCacheInitializationErrors.WithLabelValues(c.groupResource.Group, c.groupResource.Resource).Inc()
	}
}

type namespacedName struct {
	namespace string
	name      string
}

func (c *Cacher) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	ctx, span := tracing.Start(ctx, "cacher.Watch",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.Stringer("type", c.groupResource))
	defer span.End(500 * time.Millisecond)
	key, err := c.prepareKey(key, opts.Recursive)
	if err != nil {
		return nil, err
	}
	pred := opts.Predicate
	requestedWatchRV, err := c.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return nil, err
	}

	readyGeneration, downtime, err := c.ready.checkAndReadGeneration()
	if err != nil {
		return nil, errors.NewTooManyRequests(err.Error(), calculateRetryAfterForUnreadyCache(downtime))
	}

	// determine the namespace and name scope of the watch, first from the request, secondarily from the field selector
	scope := namespacedName{}
	if requestNamespace, ok := request.NamespaceFrom(ctx); ok && len(requestNamespace) > 0 {
		scope.namespace = requestNamespace
	} else if selectorNamespace, ok := pred.Field.RequiresExactMatch("metadata.namespace"); ok {
		scope.namespace = selectorNamespace
	}
	if requestInfo, ok := request.RequestInfoFrom(ctx); ok && requestInfo != nil && len(requestInfo.Name) > 0 {
		scope.name = requestInfo.Name
	} else if selectorName, ok := pred.Field.RequiresExactMatch("metadata.name"); ok {
		scope.name = selectorName
	}

	// for request like '/api/v1/watch/namespaces/*', set scope.namespace to empty.
	// namespaces don't populate metadata.namespace in ObjFields.
	if c.groupResource == coreNamespaceResource && len(scope.namespace) > 0 && scope.namespace == scope.name {
		scope.namespace = ""
	}

	triggerValue, triggerSupported := "", false
	if c.indexedTrigger != nil {
		for _, field := range pred.IndexFields {
			if field == c.indexedTrigger.indexName {
				if value, ok := pred.Field.RequiresExactMatch(field); ok {
					triggerValue, triggerSupported = value, true
					break
				}
			}
		}
	}

	// It boils down to a tradeoff between:
	// - having it as small as possible to reduce memory usage
	// - having it large enough to ensure that watchers that need to process
	//   a bunch of changes have enough buffer to avoid from blocking other
	//   watchers on our watcher having a processing hiccup
	// (with WatchCacheStallResume the size no longer decides whether a slow
	// watcher survives; see suggestedWatchChannelSize)
	chanSize := c.watchCache.suggestedWatchChannelSize(c.indexedTrigger != nil, triggerSupported)

	// client-go is going to fall back to a standard LIST on any error
	// returned for watch-list requests
	if isListWatchRequest(opts) && !etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress) {
		return newErrWatcher(fmt.Errorf("a watch stream was requested by the client but the required storage feature %s is disabled", storage.RequestWatchProgress)), nil
	}

	// Determine the ResourceVersion to which the watch cache must be synchronized
	requiredResourceVersion, err := c.getWatchCacheResourceVersion(ctx, requestedWatchRV, opts)
	if err != nil {
		return newErrWatcher(err), nil
	}

	// Determine a function that computes the bookmarkAfterResourceVersion
	bookmarkAfterResourceVersionFn, err := c.getBookmarkAfterResourceVersionLockedFunc(requestedWatchRV, requiredResourceVersion, opts)
	if err != nil {
		return newErrWatcher(err), nil
	}

	// Determine watch timeout('0' means deadline is not set, ignore checking)
	deadline, _ := ctx.Deadline()

	identifier := fmt.Sprintf("key: %q, labels: %q, fields: %q", key, pred.Label, pred.Field)

	// Create a watcher here to reduce memory allocations under lock,
	// given that memory allocation may trigger GC and block the thread.
	// Also note that emptyFunc is a placeholder, until we will be able
	// to compute watcher.forget function (which has to happen under lock).
	watcher := newCacheWatcher(
		chanSize,
		filterWithAttrsAndPrefixFunction(key, pred, c.groupResource),
		emptyFunc,
		c.versioner,
		deadline,
		pred.AllowWatchBookmarks,
		c.groupResource,
		c.watcherMetrics,
		c.clock,
		identifier,
	)
	if c.stall != nil {
		watcher.stallMetrics = c.stall.metrics
	}

	// note that c.waitUntilWatchCacheFreshAndForceAllEvents must be called without
	// the c.watchCache.RLock held otherwise we are at risk of a deadlock
	// mainly because c.watchCache.processEvent method won't be able to make progress
	//
	// moreover even though the c.waitUntilWatchCacheFreshAndForceAllEvents acquires a lock
	// it is safe to release the lock after the method finishes because we don't require
	// any atomicity between the call to the method and further calls that actually get the events.
	err = c.waitUntilWatchCacheFreshAndForceAllEvents(ctx, requiredResourceVersion, opts)
	if err != nil {
		return newErrWatcher(err), nil
	}

	// We explicitly use thread unsafe version and do locking ourself to ensure that
	// no new events will be processed in the meantime. The watchCache will be unlocked
	// on return from this function.
	// Note that we cannot do it under Cacher lock, to avoid a deadlock, since the
	// underlying watchCache is calling processEvent under its lock.
	c.watchCache.RLock()
	defer c.watchCache.RUnlock()

	var cacheInterval *watchCacheInterval
	cacheInterval, err = c.watchCache.getAllEventsSinceLocked(requiredResourceVersion, key, opts)
	if err != nil {
		// To match the uncached watch implementation, once we have passed authn/authz/admission,
		// and successfully parsed a resource version, other errors must fail with a watch event of type ERROR,
		// rather than a directly returned error.
		return newErrWatcher(err), nil
	}

	c.setInitialEventsEndBookmarkIfRequested(cacheInterval, opts, c.watchCache.resourceVersion)

	addedWatcher := false
	func() {
		c.Lock()
		defer c.Unlock()

		if generation, _, err := c.ready.checkAndReadGeneration(); generation != readyGeneration || err != nil {
			// We went unready or are already on a different generation.
			// Avoid registering and starting the watch as it will have to be
			// terminated immediately anyway.
			return
		}

		// The registration keys, read by forget and by the sync passes.
		watcher.scope, watcher.triggerValue, watcher.triggerSupported = scope, triggerValue, triggerSupported
		// Update watcher.forget function once we can compute it.
		watcher.forget = forgetWatcher(c, watcher, c.watcherIdx)
		// Update the bookMarkAfterResourceVersion
		watcher.setBookmarkAfterResourceVersion(bookmarkAfterResourceVersionFn())
		c.watchers.addWatcher(watcher, c.watcherIdx, scope, triggerValue, triggerSupported)
		addedWatcher = true

		// Add it to the queue only when the client support watch bookmarks.
		if watcher.allowWatchBookmarks {
			c.bookmarkWatchers.addWatcherThreadUnsafe(watcher)
		}
		c.watcherIdx++
	}()

	if !addedWatcher {
		// Watcher isn't really started at this point, so it's safe to just drop it.
		//
		// We're simulating the immediate watch termination, which boils down to simply
		// closing the watcher.
		return newImmediateCloseWatcher(), nil
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch) && pred.ShardSelector != nil && !pred.ShardSelector.Empty() {
		metrics.RecordShardedWatchStarted(c.groupResource)
		originalForget := watcher.forget
		watcher.forget = func(drainWatcher bool) {
			metrics.RecordShardedWatchStopped(c.groupResource)
			originalForget(drainWatcher)
		}
	}

	go watcher.processInterval(ctx, cacheInterval, requiredResourceVersion)
	return watcher, nil
}

func (c *Cacher) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	ctx, span := tracing.Start(ctx, "cacher.Get",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.String("key", key),
		attribute.String("resource-version", opts.ResourceVersion))
	key, err := c.prepareKey(key, false)
	if err != nil {
		return err
	}
	getRV, err := c.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return err
	}

	objVal, err := conversion.EnforcePtr(objPtr)
	if err != nil {
		return err
	}

	obj, exists, readResourceVersion, err := c.watchCache.WaitUntilFreshAndGet(ctx, getRV, key)
	if err != nil {
		return err
	}
	// Get long processing is >500ms, however wait for fresh cache timeout is 3s so want to avoid traces just showing waits.
	defer span.End(500 * time.Millisecond)

	if exists {
		elem, ok := obj.(*store.Element)
		if !ok {
			return fmt.Errorf("non *store.Element returned from storage: %v", obj)
		}
		objVal.Set(reflect.ValueOf(elem.Object).Elem())
	} else {
		objVal.Set(reflect.Zero(objVal.Type()))
		if !opts.IgnoreNotFound {
			return storage.NewKeyNotFoundError(key, int64(readResourceVersion))
		}
	}
	return nil
}

// computeListLimit determines whether the cacher should
// apply a limit to an incoming LIST request and returns its value.
//
// note that this function doesn't check RVM nor the Continuation token.
// these parameters are validated by the shouldDelegateList function.
//
// as of today, the limit is ignored for requests that set RV == 0
func computeListLimit(opts storage.ListOptions) int64 {
	if opts.Predicate.Limit <= 0 || opts.ResourceVersion == "0" {
		return 0
	}
	return opts.Predicate.Limit
}

type listResp struct {
	ResourceVersion uint64
	store.Range
}

// GetList implements storage.Interface
func (c *Cacher) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	preparedKey, err := c.prepareKey(key, opts.Recursive)
	if err != nil {
		return err
	}
	_, err = c.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return err
	}

	ctx, span := tracing.Start(ctx, "cacher.GetList",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.Stringer("type", c.groupResource))
	defer span.End(500 * time.Millisecond)

	if downtime, err := c.ready.check(); err != nil {
		// If Cacher is not initialized, reject List requests
		// as described in https://kep.k8s.io/4568
		return errors.NewTooManyRequests(err.Error(), calculateRetryAfterForUnreadyCache(downtime))
	}
	span.AddEvent("Ready")

	// List elements with at least 'listRV' from cache.
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	listVal, err := conversion.EnforcePtr(listPtr)
	if err != nil {
		return err
	}
	if listVal.Kind() != reflect.Slice {
		return fmt.Errorf("need a pointer to slice, got %v", listVal.Kind())
	}

	resp, indexUsed, err := c.watchCache.WaitUntilFreshAndGetList(ctx, preparedKey, opts)
	if err != nil {
		return err
	}
	span.AddEvent("Listed items from cache")
	var lastSelectedObjectKey string
	var hasMoreListItems bool
	var numFetched int
	var totalCount int64
	limit := computeListLimit(opts)
	if opts.Predicate.Empty() {
		// Every item matches, so the result size is known upfront and items can be
		// copied directly into the result list without an intermediate slice.
		count := resp.Count()
		totalCount = int64(count)
		if limit > 0 && int64(count) > limit {
			count = int(limit)
			hasMoreListItems = true
		}
		listVal.Set(reflect.MakeSlice(listVal.Type(), count, count))
		for elem, err := range resp.All() {
			if err != nil {
				return err
			}
			listVal.Index(numFetched).Set(reflect.ValueOf(elem.Object).Elem())
			lastSelectedObjectKey = elem.Key
			numFetched++
			if numFetched == count {
				break
			}
		}
	} else {
		// store pointer of eligible objects,
		// Why not directly put object in the items of listObj?
		//   the elements in ListObject are Struct type, making slice will bring excessive memory consumption.
		//   so we try to delay this action as much as possible
		var selectedObjects []runtime.Object
		shardingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch)
		for elem, err := range resp.All() {
			if err != nil {
				return err
			}
			numFetched++
			if limit > 0 && int64(len(selectedObjects)) >= limit {
				// Reaching an item past a full page is how we learn a continuation is needed.
				hasMoreListItems = true
				break
			}
			shardMatch := true
			if shardingEnabled {
				shardMatch, err = opts.Predicate.MatchesSharding(elem.Object)
				if err != nil {
					return fmt.Errorf("shard matching failed: %w", err)
				}
			}
			if shardMatch && opts.Predicate.MatchesObjectAttributes(elem.Labels, elem.Fields) {
				selectedObjects = append(selectedObjects, elem.Object)
				lastSelectedObjectKey = elem.Key
			}
		}
		if len(selectedObjects) == 0 {
			// Ensure that we never return a nil Items pointer in the result for consistency.
			listVal.Set(reflect.MakeSlice(listVal.Type(), 0, 0))
		} else {
			// Resize the slice appropriately, since we already know that size of result set
			listVal.Set(reflect.MakeSlice(listVal.Type(), len(selectedObjects), len(selectedObjects)))
			span.AddEvent("Resized result")
			for i, o := range selectedObjects {
				listVal.Index(i).Set(reflect.ValueOf(o).Elem())
			}
		}
	}
	span.AddEvent("Filtered items", attribute.Int("count", listVal.Len()))
	if c.versioner != nil {
		continueValue, remainingItemCount, err := storage.PrepareContinueToken(lastSelectedObjectKey, key, int64(resp.ResourceVersion), totalCount, hasMoreListItems, opts)
		if err != nil {
			return err
		}

		if err = c.versioner.UpdateList(listObj, resp.ResourceVersion, continueValue, remainingItemCount); err != nil {
			return err
		}
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch) {
		opts.Predicate.SetShardInfoOnList(listObj)
	}
	metrics.RecordListCacheMetrics(c.groupResource, indexUsed, numFetched, listVal.Len())
	return nil
}

// baseObjectThreadUnsafe omits locking for cachingObject.
func baseObjectThreadUnsafe(object runtime.Object) runtime.Object {
	if co, ok := object.(*cachingObject); ok {
		return co.object
	}
	return object
}

func (c *Cacher) triggerValuesThreadUnsafe(event *watchCacheEvent) ([]string, bool) {
	if c.indexedTrigger == nil {
		return nil, false
	}

	result := make([]string, 0, 2)
	result = append(result, c.indexedTrigger.indexerFunc(baseObjectThreadUnsafe(event.Object)))
	if event.PrevObject == nil {
		return result, true
	}
	prevTriggerValue := c.indexedTrigger.indexerFunc(baseObjectThreadUnsafe(event.PrevObject))
	if result[0] != prevTriggerValue {
		result = append(result, prevTriggerValue)
	}
	return result, true
}

func (c *Cacher) processEvent(event *watchCacheEvent) {
	if curLen := int64(len(c.incoming)); c.incomingHWM.Update(curLen) {
		// Monitor if this gets backed up, and how much.
		klog.V(1).Infof("cacher (%v): %v objects queued in incoming channel.", c.groupResource.String(), curLen)
	}
	c.incoming <- *event
}

func (c *Cacher) dispatchEvents() {
	// Jitter to help level out any aggregate load.
	bookmarkTimer := c.clock.NewTimer(wait.Jitter(time.Second, 0.25))
	defer bookmarkTimer.Stop()

	// The internal informer populates the RV as soon as it conducts
	// The first successful sync with the underlying store.
	// The cache must wait until this first sync is completed to be deemed ready.
	// Since we cannot send a bookmark when the lastProcessedResourceVersion is 0,
	// we poll aggressively for the first list RV before entering the dispatch loop.
	lastProcessedResourceVersion := uint64(0)
	if err := wait.PollUntilContextCancel(wait.ContextForChannel(c.stopCh), 10*time.Millisecond, true, func(_ context.Context) (bool, error) {
		if rv := c.watchCache.getListResourceVersion(); rv != 0 {
			lastProcessedResourceVersion = rv
			return true, nil
		}
		return false, nil
	}); err != nil {
		// given the function above never returns error,
		// the non-empty error means that the stopCh was closed
		return
	}
	var passC <-chan time.Time
	var dispatcherHook chan func(syncPassFunc)
	if c.stall != nil {
		c.stall.passTimer = c.clock.NewTimer(syncPassPeriod)
		c.stall.passTimer.Stop()
		defer c.stall.passTimer.Stop()
		dispatcherHook = c.stall.dispatcherHook
	}
	for {
		select {
		case event, ok := <-c.incoming:
			if !ok {
				return
			}
			// Don't dispatch bookmarks coming from the storage layer.
			// They can be very frequent (even to the level of subseconds)
			// to allow efficient watch resumption on kube-apiserver restarts,
			// and propagating them down may overload the whole system.
			//
			// TODO: If at some point we decide the performance and scalability
			// footprint is acceptable, this is the place to hook them in.
			// However, we then need to check if this was called as a result
			// of a bookmark event or regular Add/Update/Delete operation by
			// checking if resourceVersion here has changed.
			if event.Type != watch.Bookmark {
				event.timeline.MarkAt(metrics.PointDispatchStarted, c.clock.Now())
				c.dispatchEvent(&event)
				metrics.EventsCounter.WithLabelValues(c.groupResource.Group, c.groupResource.Resource).Inc()
			}
			lastProcessedResourceVersion = event.ResourceVersion
			if c.stall != nil {
				passC = c.syncAfterDispatch()
			}
		case <-bookmarkTimer.C():
			bookmarkTimer.Reset(wait.Jitter(time.Second, 0.25))
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
			if c.stall != nil {
				passC = c.syncAfterDispatch()
			}
		case <-passC:
			c.stall.passArmed = false
			c.syncPass(c.clock.Now())
			passC = c.armSyncTimer()
		case fn := <-dispatcherHook:
			// Re-arm only after a pass: re-arming after a hook that ran
			// none would drain a tick the test's clock step just produced.
			ran := false
			fn(func() (int, int) {
				ran = true
				return c.syncPass(c.clock.Now())
			})
			if ran {
				passC = c.armSyncTimer()
			}
		case <-c.stopCh:
			return
		}
	}
}

// syncAfterDispatch runs a sync pass when one is due after a dispatched
// event and keeps the pass timer armed while any watcher is unsynced. It
// returns the timer channel to select on, nil while nothing is unsynced.
func (c *Cacher) syncAfterDispatch() <-chan time.Time {
	s := c.stall
	rearm := false
	stalled := s.stalled
	if stalled {
		s.stalled = false
		if s.passPeriod != syncPassPeriod {
			s.passPeriod = syncPassPeriod
			rearm = true
		}
	}
	if len(s.unsynced) == 0 {
		return c.armSyncTimer()
	}
	now := c.clock.Now()
	if now.Sub(s.lastPass) >= s.passPeriod {
		c.syncPass(now)
		if stalled && s.passPeriod == syncPassIdlePeriod {
			// The pass that follows a stall finds the new member with a
			// full input and serves nothing; keep the short period so its
			// first catch-up is not delayed by the idle backoff.
			s.passPeriod = syncPassPeriod
		}
		return c.armSyncTimer()
	}
	if s.passArmed && !rearm {
		return s.passTimer.C()
	}
	return c.armSyncTimer()
}

// armSyncTimer re-arms the pass timer for the current period while any
// watcher is unsynced and stops it otherwise, returning the channel to
// select on (nil when stopped). A stale tick is drained so that a fake
// clock, whose timer channel holds one tick, never blocks on the next fire.
func (c *Cacher) armSyncTimer() <-chan time.Time {
	s := c.stall
	if s.passArmed {
		s.passTimer.Stop()
		select {
		case <-s.passTimer.C():
		default:
		}
		s.passArmed = false
	}
	if len(s.unsynced) == 0 {
		return nil
	}
	s.passTimer.Reset(s.passPeriod)
	s.passArmed = true
	return s.passTimer.C()
}

// syncPass serves the unsynced watchers from the watch cache history, the
// way etcd's syncWatchers serves its unsynced group: candidates are the
// unsynced watchers with room in their input (the ones the lead choice
// prefers when there are more than maxWatchersPerSync), sorted by
// position; they are served in cohorts, led by the candidate the lead
// choice prefers, that share one history read and one wrapped copy of
// every event, under a scan budget, a push budget and a cohort budget for
// the whole pass.
// A member that accepted everything it was offered and whose cohort read
// the history to its end is synced again; one whose position aged out of
// the history is expired with an in-stream 410. Runs on the dispatcher
// goroutine between dispatches; it never holds the Cacher lock while taking
// the watch cache lock, because Watch holds the latter across the former.
func (c *Cacher) syncPass(now time.Time) (served, expired int) {
	s := c.stall
	s.lastPass = now
	s.passCount++
	oldest, ok := s.src.oldest()
	if !ok {
		return 0, 0
	}

	c.Lock()
	c.dispatching = true
	s.expiredBuf = s.expiredBuf[:0]
	s.members = s.members[:0]
	for w := range s.unsynced {
		// A stop deferred by an earlier dispatch has run by now, so a
		// stopped member has a closed input that must never be pushed to.
		if w.forgotten || w.stopped {
			delete(s.unsynced, w)
			continue
		}
		// The same boundary GetIntervalLocked rejects, so an unexpired
		// lead is always servable. The reason is the watcher's state now;
		// the termination is counted when the 410 is sent.
		if w.position+1 < oldest {
			w.expired = true
			if w.live.Load() {
				w.expiredReason = metrics.TerminationReasonResourceExpired
			} else {
				w.expiredReason = metrics.TerminationReasonResourceExpiredInitial
			}
			delete(s.unsynced, w)
			s.expiredBuf = append(s.expiredBuf, w)
			continue
		}
		if len(w.input) < cap(w.input) {
			s.members = append(s.members, syncMember{w: w})
		}
	}
	c.Unlock()

	// forget takes the Cacher lock itself; with dispatching set the stop
	// is deferred to finishDispatching, and in drain mode the watcher
	// goroutine delivers the backlog and then the 410.
	for _, w := range s.expiredBuf {
		w.forget(true)
	}
	expired = len(s.expiredBuf)
	clear(s.expiredBuf)

	for i := range s.members {
		m := &s.members[i]
		m.room = (cap(m.w.input) - len(m.w.input)) * 1000 / cap(m.w.input)
		// The first pass a served member has room again judges it: a
		// fast drainer emptied its input, a slow one freed a slot or a
		// few. The judgement holds until the next service, so a pass
		// that leaves the member out does not change it.
		if m.w.servedPass != 0 && !m.w.judged {
			m.w.judged, m.w.fastDrainer = true, m.room == 1000
		}
		m.eager = m.w.servedPass == 0 || m.w.fastDrainer
	}
	if len(s.members) > maxWatchersPerSync {
		// Keep the members the lead choice would pick first (the same
		// keys), so that a client that can catch up is never left out
		// behind maxWatchersPerSync laggards; among equals the order
		// rotates with the cursor (at or above it first), so every member
		// gets its turn across passes.
		slices.SortFunc(s.members, func(a, b syncMember) int {
			if c := cmp.Compare(b.room, a.room); c != 0 {
				return c
			}
			if a.eager != b.eager {
				if a.eager {
					return -1
				}
				return 1
			}
			return cmp.Compare(a.w.position-s.syncCursor, b.w.position-s.syncCursor)
		})
		clear(s.members[maxWatchersPerSync:])
		s.members = s.members[:maxWatchersPerSync]
	}
	slices.SortFunc(s.members, func(a, b syncMember) int {
		return cmp.Compare(a.w.position, b.w.position)
	})
	c.buildSyncIndexes()

	budget := syncScanBudget
	s.pushBudget = syncPushBudget
	s.pushed = 0
	s.scanned = s.scanned[:0]
	for budget >= syncIntervalOpenCost && s.pushBudget > 0 {
		lead := c.pickSyncLead()
		if lead < 0 {
			break
		}
		c.serveSyncCohort(lead, &budget, now)
	}
	// A spent push budget means a member is still owed events it has
	// room for; read before the retry rounds get their own allowance.
	pushBudgetSpent := s.pushBudget <= 0
	// Retry rounds over the whole pass, with their own push allowance: a
	// fast client drained what its cohort pushed while the later cohorts
	// ran (microseconds against the rest of the pass), so it takes the
	// events its cohort recorded as owed now, up to two rounds of its
	// room. This is what lets a client with a small input catch up with
	// a churn faster than cap(input) per pass period.
	s.pushBudget = syncPushBudget
	for round := 0; round < syncPushRetryRounds && s.pushBudget > 0; round++ {
		for i := range s.members {
			m := &s.members[i]
			if !m.blocked || !m.eager || len(m.w.input) == cap(m.w.input) {
				continue
			}
			if c.reofferPending(m, now) {
				m.blocked = false
			}
		}
	}
	clear(s.scanned)
	s.scanned = s.scanned[:0]
	if s.pushed > 0 {
		s.metrics.DeferredEvents.Add(float64(s.pushed))
	}

	blocked := false
	c.Lock()
	for i := range s.members {
		m := &s.members[i]
		if m.served {
			served++
		}
		blocked = blocked || m.blocked
		if !m.resync || m.w.forgotten || m.w.stopped {
			continue
		}
		m.w.unsynced = false
		m.w.servedPass, m.w.judged, m.w.fastDrainer = 0, false, false
		delete(s.unsynced, m.w)
		s.metrics.CatchupRounds.Inc()
		s.metrics.CatchupEvents.Observe(float64(m.w.catchupEvents))
		m.w.catchupEvents = 0
	}
	c.Unlock()
	c.finishDispatching()

	clear(s.members)
	s.members = s.members[:0]
	// Busy: a member is still owed events it has room for, because its
	// input was full when offered (and is draining) or the push budget
	// ran out.
	s.passPeriod = nextPassPeriod(served, expired, blocked || pushBudgetSpent, c.clock.Since(now))
	return served, expired
}

// nextPassPeriod picks the period until the next pass from what this one
// did: busy while a member is still owed events it has room for, scaled to
// the pass duration so that passes take at most about a tenth of the
// dispatcher's time, but never above the normal period, so a pass that
// the host descheduled cannot amplify into a long silence; idle after a
// pass that served and expired nothing; the normal period otherwise.
func nextPassPeriod(served, expired int, busy bool, passDuration time.Duration) time.Duration {
	switch {
	case busy:
		return min(syncPassPeriod, max(syncPassBusyPeriod, syncPassBusyDutyFactor*passDuration))
	case served == 0 && expired == 0:
		return syncPassIdlePeriod
	default:
		return syncPassPeriod
	}
}

// buildSyncIndexes builds the pass's lookup maps over the candidates in
// sorted order, mirroring the indexes startDispatching consults: scope key
// to members, trigger value to members, and the fan-out list of every
// trigger indexed member for an event without a supported trigger value.
// Every list is in position order, so offerToAll can stop at the first
// member above the scanned event.
func (c *Cacher) buildSyncIndexes() {
	s := c.stall
	clear(s.byScope)
	clear(s.byTrigger)
	s.triggerFanout = s.triggerFanout[:0]
	for i := range s.members {
		m := &s.members[i]
		m.startPosition = m.w.position
		if m.w.triggerSupported {
			s.byTrigger[m.w.triggerValue] = append(s.byTrigger[m.w.triggerValue], i)
			s.triggerFanout = append(s.triggerFanout, i)
			continue
		}
		s.byScope[m.w.scope] = append(s.byScope[m.w.scope], i)
	}
}

// pickSyncLead returns the index of the next cohort's lead, or -1 when
// every member was served. Among the unserved members it prefers, in this
// order: the largest free share of the input (the member that drained
// the most of what it holds, whatever its capacity); then an eager
// member (not served since it went unsynced, or judged a fast drainer:
// its input was empty at the first pass it had room after its last
// service) over one judged slow (it had freed a slot or a few by then);
// then the first member at or above the cursor, else the first. The
// middle key depends only on the member's own behaviour, so being left
// out by the candidate cap or the push budget for a pass does not demote
// it. Without these keys a client draining at full speed would wait its
// turn behind every slow client that had time to free a few slots, and
// could never catch up with the churn; a burst of clients stalling
// together ties on all three and still forms one cohort. "At or above"
// lets a lead that accepted its whole window be picked again and keep
// tracking the history.
func (c *Cacher) pickSyncLead() int {
	s := c.stall
	lead := -1
	for i := range s.members {
		m := &s.members[i]
		if m.served {
			continue
		}
		if lead < 0 {
			lead = i
			continue
		}
		l := &s.members[lead]
		switch {
		case m.room > l.room || (m.room == l.room && m.eager && !l.eager):
			lead = i
		case m.room == l.room && m.eager == l.eager && l.w.position < s.syncCursor && m.w.position >= s.syncCursor:
			lead = i
		}
	}
	return lead
}

// serveSyncCohort opens one history interval at the lead's position, scans
// it up to the remaining budget and pushes every scanned event to the
// unserved members above the lead that it is selected for. The cohort is
// the lead plus the unserved members whose position is below the last
// scanned resourceVersion; every one of them counts as served, and the
// cursor moves to that resourceVersion. A member that rejected a push
// keeps its last accepted resourceVersion and records the events still
// owed; the end of the pass re-offers them.
func (c *Cacher) serveSyncCohort(lead int, budget *int, now time.Time) {
	s := c.stall
	members := s.members
	leadW := members[lead].w
	// The open is priced as a short scan (see syncIntervalOpenCost); the
	// caller made sure the budget covers it.
	*budget -= syncIntervalOpenCost
	interval, err := s.src.intervalSince(leadW.position)
	if err != nil {
		// The history advanced past the lead since it was checked: it is
		// served with nothing offered, the cursor stays, and step 1 of the
		// next pass expires it.
		members[lead].served = true
		leadW.servedPass, leadW.judged = s.passCount, false
		return
	}

	s.lead = lead
	s.cut = false
	s.blockedCount, s.pendingOpen = 0, 0
	lastScannedRV := leadW.position
	historyEnd := false
	// The cohort is members[lead:end) less the ones an earlier cohort
	// served; end advances with the scan over the members whose start
	// position is below the scanned event, and active counts the cohort
	// members so far, i.e. the members the scan has offered something to.
	// Once the push budget is spent (checked between events, so every
	// event scanned before the cut was offered to every active member)
	// the scan goes on for the lead alone, up to its room: the lead is the
	// member the pass chose to serve, and the followers it picked up on
	// the way must not take the whole budget before it got its share.
	// Followers are offered nothing past cutRV.
	end, active := lead, 0
	budgetSpent := false
	cutRV := uint64(0)
	for {
		if *budget == 0 {
			budgetSpent = true
			break
		}
		if s.cut && members[lead].blocked {
			break
		}
		if !s.cut && s.pushBudget <= 0 {
			s.cut, cutRV = true, lastScannedRV
		}
		ev, err := interval.Next()
		if err != nil {
			// Invalidated mid scan: what was scanned stands, the rest is
			// retried by the next pass from the members' positions.
			break
		}
		if ev == nil {
			historyEnd = true
			break
		}
		*budget--
		rv := ev.ResourceVersion
		for !s.cut && end < len(members) && members[end].startPosition < rv {
			if !members[end].served {
				active++
			}
			end++
		}
		s.scanned = append(s.scanned, scannedEvent{
			raw:       ev,
			namespace: ev.ObjFields["metadata.namespace"],
			name:      ev.ObjFields["metadata.name"],
		})
		idx := len(s.scanned) - 1
		s.scanned[idx].triggerValues, s.scanned[idx].triggerSupported = c.triggerValuesThreadUnsafe(ev)
		lastScannedRV = rv
		c.offerScanned(idx, now)
		if !s.cut && s.blockedCount == active && s.pendingOpen == 0 {
			// Every member offered something has a full input, and every
			// eager one among them holds all the owed events the retry
			// rounds can push; a member above this point gets its own
			// cohort.
			break
		}
	}
	if budgetSpent {
		// A budget ran out exactly at the last event scanned: one more
		// read tells whether the history end was reached, so the cohort
		// can resync this pass. An event returned here is not consumed;
		// the next pass reads it again.
		if ev, err := interval.Next(); err == nil && ev == nil {
			historyEnd = true
		}
	}

	// The lead is a cohort member even when the interval was empty.
	if end == lead {
		end = lead + 1
	}
	for i := lead; i < end; i++ {
		m := &members[i]
		if m.served {
			continue
		}
		m.served = true
		m.w.servedPass, m.w.judged = s.passCount, false
		if m.blocked {
			continue
		}
		// Position follows the scan: everything in (position, limit] was
		// scanned and either accepted or not selected; for a follower
		// the scan ends at the cut, and it reached the history end only
		// if nothing was scanned past the cut.
		limit, atEnd := lastScannedRV, historyEnd
		if s.cut && i != lead {
			limit, atEnd = cutRV, historyEnd && cutRV == lastScannedRV
		}
		if limit > m.w.position {
			m.w.position = limit
		}
		m.resync = atEnd
	}
	s.syncCursor = lastScannedRV
}

// offerScanned pushes the scanned event at idx to every cohort member it is
// selected for, by the rules of startDispatching.
func (c *Cacher) offerScanned(idx int, now time.Time) {
	s := c.stall
	se := &s.scanned[idx]
	forEachSelectionKey(se.namespace, se.name, se.triggerValues, se.triggerSupported,
		func(key namespacedName) { c.offerToAll(s.byScope[key], idx, now) },
		func(value string) { c.offerToAll(s.byTrigger[value], idx, now) },
		func() { c.offerToAll(s.triggerFanout, idx, now) })
}

// offerToAll offers the scanned event at idx to the cohort members among
// hits: the entries at or above the lead that no earlier cohort served,
// or the lead alone once the push budget is spent. hits is in position
// order, so the walk stops at the first member whose start position is at
// or above the event.
func (c *Cacher) offerToAll(hits []int, idx int, now time.Time) {
	s := c.stall
	rv := s.scanned[idx].raw.ResourceVersion
	for _, i := range hits {
		m := &s.members[i]
		if m.startPosition >= rv {
			return
		}
		if i < s.lead || m.served || (s.cut && i != s.lead) {
			continue
		}
		c.offerTo(m, idx, now)
	}
}

// offerTo pushes the scanned event at idx to the member; a blocked member
// only records it as owed, up to what the retry rounds can push.
func (c *Cacher) offerTo(m *syncMember, idx int, now time.Time) {
	s := c.stall
	pendingCap := syncPushRetryRounds * cap(m.w.input)
	if m.blocked {
		if len(m.pending) < pendingCap {
			m.pending = append(m.pending, idx)
			if m.eager && len(m.pending) == pendingCap {
				s.pendingOpen--
			}
		} else {
			m.pendingTruncated = true
		}
		return
	}
	if !c.pushScanned(m, idx, now) {
		m.blocked = true
		m.pending = append(m.pending, idx)
		s.blockedCount++
		if m.eager {
			// pendingCap is at least two, so the first owed event never
			// fills it.
			s.pendingOpen++
		}
	}
}

// reofferPending pushes the events owed to a blocked member in order and
// reports whether all of them were accepted; a member owed more than
// pending holds, or cut off by the push budget, stays blocked at its last
// accepted resourceVersion.
func (c *Cacher) reofferPending(m *syncMember, now time.Time) bool {
	for len(m.pending) > 0 {
		if c.stall.pushBudget <= 0 || !c.pushScanned(m, m.pending[0], now) {
			return false
		}
		m.pending = m.pending[1:]
	}
	return !m.pendingTruncated
}

// pushScanned wraps the scanned event at idx once, exactly like the live
// path does per dispatch, and offers the shared copy to the member; the
// scanned events live for the whole pass, so the retry rounds at its end
// reuse the copies.
func (c *Cacher) pushScanned(m *syncMember, idx int, now time.Time) bool {
	s := c.stall
	se := &s.scanned[idx]
	if se.wrapped == nil {
		wcEvent := *se.raw
		setCachingObjects(&wcEvent, c.versioner)
		wcEvent.timeline.MarkAt(metrics.PointDispatchStarted, now)
		wcEvent.timeline.MarkAt(metrics.PointWatcherEnqueued, now)
		se.wrapped = &wcEvent
	}
	if !m.w.nonblockingAdd(se.wrapped) {
		return false
	}
	m.w.position = se.raw.ResourceVersion
	m.w.catchupEvents++
	s.pushed++
	s.pushBudget--
	return true
}

func setCachingObjects(event *watchCacheEvent, versioner storage.Versioner) {
	switch event.Type {
	case watch.Added, watch.Modified:
		if object, err := newCachingObject(event.Object); err == nil {
			event.Object = object
		} else {
			klog.Errorf("couldn't create cachingObject from: %#v", event.Object)
		}
		// Don't wrap PrevObject for update event (for create events it is nil).
		// We only encode those to deliver DELETE watch events, so if
		// event.Object is not nil it can be used only for watchers for which
		// selector was satisfied for its previous version and is no longer
		// satisfied for the current version.
		// This is rare enough that it doesn't justify making deep-copy of the
		// object (done by newCachingObject) every time.
	case watch.Deleted:
		// Don't wrap Object for delete events - these are not to deliver any
		// events. Only wrap PrevObject.
		if object, err := newCachingObject(event.PrevObject); err == nil {
			// Update resource version of the object.
			// event.PrevObject is used to deliver DELETE watch events and
			// for them, we set resourceVersion to <current> instead of
			// the resourceVersion of the last modification of the object.
			updateResourceVersion(object, versioner, event.ResourceVersion)
			event.PrevObject = object
		} else {
			klog.Errorf("couldn't create cachingObject from: %#v", event.Object)
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
		if c.stall != nil {
			// A bookmark below the position would move the client's last
			// seen resourceVersion backwards after a resync that scanned
			// past the events still queued in c.incoming.
			for _, watcher := range c.watchersBuffer {
				if watcher.unsynced || event.ResourceVersion < watcher.position {
					continue
				}
				if watcher.nonblockingAdd(event) {
					watcher.position = event.ResourceVersion
				}
			}
			return
		}
		for _, watcher := range c.watchersBuffer {
			watcher.nonblockingAdd(event)
		}
	} else {
		// Set up caching of object serializations only for dispatching this event.
		//
		// Storing serializations in memory would result in increased memory usage,
		// but it would help for caching encodings for watches started from old
		// versions. However, we still don't have a convincing data that the gain
		// from it justifies increased memory usage, so for now we drop the cached
		// serializations after dispatching this event.
		//
		// Given that CachingObject is just wrapping the object and not perfoming
		// deep-copying (until some field is explicitly being modified), we create
		// it unconditionally to ensure safety and reduce deep-copying.
		//
		// Make a shallow copy to allow overwriting Object and PrevObject.
		wcEvent := *event
		setCachingObjects(&wcEvent, c.versioner)
		wcEvent.timeline.MarkAt(metrics.PointWatcherEnqueued, c.clock.Now())
		event = &wcEvent

		if c.stall != nil {
			// Never wait for a slow watcher and never terminate it here: a
			// watcher whose input is full becomes unsynced and the sync
			// passes serve it from the history. Events at or below the
			// position were already pushed by a pass.
			for _, watcher := range c.watchersBuffer {
				if watcher.unsynced || event.ResourceVersion <= watcher.position {
					continue
				}
				if watcher.nonblockingAdd(event) {
					watcher.position = event.ResourceVersion
					continue
				}
				// Everything below this event was accepted, so the pass
				// resumes from just below it.
				watcher.position = max(watcher.position, event.ResourceVersion-1)
				watcher.unsynced = true
				watcher.servedPass, watcher.judged, watcher.fastDrainer = 0, false, false
				c.stall.unsynced[watcher] = struct{}{}
				c.stall.stalled = true
				c.stall.metrics.Stalls.Inc()
			}
			return
		}

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

			// Send event to all blocked watchers. As long as timer is running,
			// `add` will wait for the watcher to unblock. After timeout,
			// `add` will not wait, but immediately close a still blocked watcher.
			// Hence, every watcher gets the chance to unblock itself while timer
			// is running, not only the first ones in the list.
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

func (c *Cacher) startDispatchingBookmarkEventsLocked() {
	// Pop already expired watchers. However, explicitly ignore stopped ones,
	// as we don't delete watcher from bookmarkWatchers when it is stopped.
	for _, watchers := range c.bookmarkWatchers.popExpiredWatchersThreadUnsafe() {
		for _, watcher := range watchers {
			// c.Lock() is held here.
			// watcher.stopThreadUnsafe() is protected by c.Lock()
			if watcher.stopped {
				continue
			}
			c.watchersBuffer = append(c.watchersBuffer, watcher)
			c.expiredBookmarkWatchers = append(c.expiredBookmarkWatchers, watcher)
		}
	}
}

// startDispatching chooses watchers potentially interested in a given event
// a marks dispatching as true.
func (c *Cacher) startDispatching(event *watchCacheEvent) {
	// It is safe to call triggerValuesThreadUnsafe here, because at this
	// point only this thread can access this event (we create a separate
	// watchCacheEvent for every dispatch).
	triggerValues, supported := c.triggerValuesThreadUnsafe(event)

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
		c.startDispatchingBookmarkEventsLocked()
		// return here to reduce following code indentation and diff
		return
	}

	namespace := event.ObjFields["metadata.namespace"]
	name := event.ObjFields["metadata.name"]
	forEachSelectionKey(namespace, name, triggerValues, supported,
		func(key namespacedName) {
			for _, watcher := range c.watchers.allWatchers[key] {
				c.watchersBuffer = append(c.watchersBuffer, watcher)
			}
		},
		func(triggerValue string) {
			for _, watcher := range c.watchers.valueWatchers[triggerValue] {
				c.watchersBuffer = append(c.watchersBuffer, watcher)
			}
		},
		func() {
			for _, watchers := range c.watchers.valueWatchers {
				for _, watcher := range watchers {
					c.watchersBuffer = append(c.watchersBuffer, watcher)
				}
			}
		})
}

// forEachSelectionKey encodes how an event selects watchers, for the live
// dispatch and the sync passes alike: scoped is called with every
// namespace/name key the event's watchers are indexed under (namespaced
// watchers scoped by name, namespaced watchers not scoped by name,
// cluster-wide watchers scoped by name, cluster-wide watchers unscoped by
// name), then trigger with each of the event's trigger values, or, when the
// trigger is not supported, allTriggers once for the fan-out to every
// watcher interested in an exact trigger value.
func forEachSelectionKey(namespace, name string, triggerValues []string, triggerSupported bool, scoped func(namespacedName), trigger func(string), allTriggers func()) {
	if len(namespace) > 0 {
		if len(name) > 0 {
			scoped(namespacedName{namespace: namespace, name: name})
		}
		scoped(namespacedName{namespace: namespace})
	}
	if len(name) > 0 {
		scoped(namespacedName{name: name})
	}
	scoped(namespacedName{})

	if triggerSupported {
		for _, triggerValue := range triggerValues {
			trigger(triggerValue)
		}
		return
	}
	// supported equal to false generally means that trigger function
	// is not defined (or not aware of any indexes). In this case,
	// watchers filters should generally also don't generate any
	// trigger values, but can cause problems in case of some
	// misconfiguration. Thus we paranoidly leave this branch.
	allTriggers()
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
		watcher.stopLocked()
	}
	c.watchersToStop = c.watchersToStop[:0]

	for _, watcher := range c.expiredBookmarkWatchers {
		if watcher.stopped {
			continue
		}
		// requeue the watcher for the next bookmark if needed.
		c.bookmarkWatchers.addWatcherThreadUnsafe(watcher)
	}
	c.expiredBookmarkWatchers = c.expiredBookmarkWatchers[:0]
}

func (c *Cacher) terminateAllWatchers() {
	c.Lock()
	defer c.Unlock()
	c.watchers.terminateAll(c.groupResource, c.stopWatcherLocked)
}

func (c *Cacher) stopWatcherLocked(watcher *cacheWatcher) {
	// A non-draining stop is final: a later draining stop (a sync pass
	// expiring the watcher) must not reopen the drain window, or done
	// would never be closed and the goroutine could leak on a full result
	// channel. Latched here so that terminateAllWatchers, which stops
	// without forgetting, is covered too.
	if !watcher.drainInputBuffer {
		watcher.hardStop = true
	}
	if c.dispatching {
		c.watchersToStop = append(c.watchersToStop, watcher)
	} else {
		watcher.stopLocked()
	}
}

func (c *Cacher) isStopped() bool {
	c.stopLock.RLock()
	defer c.stopLock.RUnlock()
	return c.stopped
}

func (c *Cacher) Compact(resourceVersion string) error {
	rv, err := c.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}
	c.watchCache.storage.Compact(rv)
	return nil
}

func (c *Cacher) MarkConsistent(consistent bool) {
	c.watchCache.storage.MarkConsistent(consistent)
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
	c.ready.stop()
	c.stopLock.Unlock()
	close(c.stopCh)
	c.stopWg.Wait()
}

func (c *Cacher) prepareKey(key string, recursive bool) (string, error) {
	return storage.PrepareKey(c.resourcePrefix, key, recursive)
}

func forgetWatcher(c *Cacher, w *cacheWatcher, index int) func(bool) {
	return func(drainWatcher bool) {
		c.Lock()
		defer c.Unlock()

		w.setDrainInputBufferLocked(drainWatcher)
		w.forgotten = true

		// It's possible that the watcher is already not in the structure (e.g. in case of
		// simultaneous Stop() and terminateAllWatchers(), but it is safe to call stopLocked()
		// on a watcher multiple times.
		c.watchers.deleteWatcher(index, w.scope, w.triggerValue, w.triggerSupported)
		c.stopWatcherLocked(w)
	}
}

func filterWithAttrsAndPrefixFunction(prefix string, p storage.SelectionPredicate, groupResource schema.GroupResource) filterWithAttrsFunc {
	isSharded := utilfeature.DefaultFeatureGate.Enabled(features.ShardedListAndWatch) && p.ShardSelector != nil && !p.ShardSelector.Empty()
	filterFunc := func(objKey string, label labels.Set, field fields.Set, obj runtime.Object) bool {
		if !key.HasPathPrefix(objKey, prefix) {
			return false
		}
		if isSharded {
			matches, err := p.MatchesSharding(obj)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("shard matching failed for %v: %w", groupResource, err))
				return false
			}
			if !matches {
				metrics.RecordWatchFilteredEvent(groupResource)
				return false
			}
		}
		return p.MatchesObjectAttributes(label, field)
	}
	return filterFunc
}

// LastSyncResourceVersion returns resource version to which the underlying cache is synced.
func (c *Cacher) LastSyncResourceVersion() (uint64, error) {
	if err := c.ready.wait(context.Background()); err != nil {
		return 0, errors.NewServiceUnavailable(err.Error())
	}

	resourceVersion := c.reflector.LastSyncResourceVersion()
	return c.versioner.ParseResourceVersion(resourceVersion)
}

// getBookmarkAfterResourceVersionLockedFunc returns a function that
// spits a ResourceVersion after which the bookmark event will be delivered.
//
// The returned function must be called under the watchCache lock.
func (c *Cacher) getBookmarkAfterResourceVersionLockedFunc(parsedResourceVersion, requiredResourceVersion uint64, opts storage.ListOptions) (func() uint64, error) {
	if !isListWatchRequest(opts) {
		return func() uint64 { return 0 }, nil
	}

	switch {
	case len(opts.ResourceVersion) == 0:
		return func() uint64 { return requiredResourceVersion }, nil
	case parsedResourceVersion == 0:
		// here we assume that watchCache locked is already held
		return func() uint64 { return c.watchCache.resourceVersion }, nil
	default:
		return func() uint64 { return parsedResourceVersion }, nil
	}
}

// isListWatchRequest is mirrored in staging/src/k8s.io/apiserver/pkg/endpoints/handlers/get.go
func isListWatchRequest(opts storage.ListOptions) bool {
	return opts.SendInitialEvents != nil && *opts.SendInitialEvents && opts.Predicate.AllowWatchBookmarks
}

// getWatchCacheResourceVersion returns a ResourceVersion to which the watch cache must be synchronized to
//
// Depending on the input parameters, the semantics of the returned ResourceVersion are:
//   - must be at Exact RV (when parsedWatchResourceVersion > 0)
//   - can be at Any RV (when parsedWatchResourceVersion = 0)
//   - must be at Most Recent RV (return an RV from etcd)
//
// note that the above semantic is enforced by the API validation (defined elsewhere):
//
//	if SendInitiaEvents != nil => ResourceVersionMatch = NotOlderThan
//	if ResourceVersionmatch != nil => ResourceVersionMatch = NotOlderThan & SendInitialEvents != nil
func (c *Cacher) getWatchCacheResourceVersion(ctx context.Context, parsedWatchResourceVersion uint64, opts storage.ListOptions) (uint64, error) {
	if len(opts.ResourceVersion) != 0 {
		return parsedWatchResourceVersion, nil
	}
	// legacy case
	if opts.SendInitialEvents == nil && opts.ResourceVersion == "" {
		return 0, nil
	}
	rv, err := c.storage.GetCurrentResourceVersion(ctx)
	return rv, err
}

// waitUntilWatchCacheFreshAndForceAllEvents waits until cache is at least
// as fresh as given requestedWatchRV if sendInitialEvents was requested.
// otherwise, we allow for establishing the connection because the clients
// can wait for events without unnecessary blocking.
func (c *Cacher) waitUntilWatchCacheFreshAndForceAllEvents(ctx context.Context, requestedWatchRV uint64, opts storage.ListOptions) error {
	if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
		// Here be dragons:
		// Since the etcd feature checker needs to check all members
		// to determine whether a given feature is supported,
		// we may receive a positive response even if the feature is not supported.
		//
		// In this very rare scenario, the worst case will be that this
		// request will wait for 3 seconds before it fails.
		span := tracing.SpanFromContext(ctx)
		consistentReadSupported := delegator.ConsistentReadSupported()
		c.watchCache.RLock()
		span.AddEvent("watchCache locked acquired")
		defer c.watchCache.RUnlock()
		err := c.watchCache.waitUntilFreshLocked(ctx, consistentReadSupported, requestedWatchRV)
		if err != nil {
			return err
		}
		span.AddEvent("watchCache fresh enough")
		return nil
	}
	return nil
}

// Wait blocks until the cacher is Ready or Stopped, it returns an error if Stopped.
func (c *Cacher) Wait(ctx context.Context) error {
	return c.ready.wait(ctx)
}

// setInitialEventsEndBookmarkIfRequested sets initialEventsEndBookmark field in watchCacheInterval for watchlist request
func (c *Cacher) setInitialEventsEndBookmarkIfRequested(cacheInterval *watchCacheInterval, opts storage.ListOptions, currentResourceVersion uint64) {
	if opts.SendInitialEvents != nil && *opts.SendInitialEvents && opts.Predicate.AllowWatchBookmarks {
		// We don't need to set the InitialEventsAnnotation for this bookmark event,
		// because this will be automatically set during event conversion in cacheWatcher.convertToWatchEvent method
		initialEventsEndBookmark := &watchCacheEvent{
			Type:            watch.Bookmark,
			Object:          c.newFunc(),
			ResourceVersion: currentResourceVersion,
		}

		if err := c.versioner.UpdateObject(initialEventsEndBookmark.Object, initialEventsEndBookmark.ResourceVersion); err != nil {
			klog.Errorf("failure to set resourceVersion to %d on initialEventsEndBookmark event %+v for watchlist request and wait for bookmark trigger to send", initialEventsEndBookmark.ResourceVersion, initialEventsEndBookmark.Object)
			initialEventsEndBookmark = nil
		}

		cacheInterval.initialEventsEndBookmark = initialEventsEndBookmark
	}
}

func (c *Cacher) getKeys(ctx context.Context) ([]string, error) {
	ctx, span := tracing.Start(ctx, "cacher.getKeys",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)))
	defer span.End(500 * time.Millisecond)
	rev, err := c.storage.GetCurrentResourceVersion(ctx)
	if err != nil {
		return nil, err
	}
	span.AddEvent("GetCurrentResourceVersion succeed", attribute.Int64("resource-version", int64(rev)))
	return c.watchCache.WaitUntilFreshAndGetKeys(ctx, rev)
}

func (c *Cacher) Ready() bool {
	_, err := c.ready.check()
	return err == nil
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

func (c *Cacher) ShouldDelegateExactRV(resourceVersion string, recursive bool) (delegator.Result, error) {
	// Not Recursive is not supported unitl exact RV is implemented for WaitUntilFreshAndGet.
	if !recursive || !c.watchCache.storage.SnapshottingEnabled() {
		return delegator.Result{ShouldDelegate: true}, nil
	}
	listRV, err := c.versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return delegator.Result{}, err
	}
	return c.shouldDelegateExactRV(listRV)
}

func (c *Cacher) ShouldDelegateContinue(continueToken string, recursive bool) (delegator.Result, error) {
	// Not Recursive is not supported unitl exact RV is implemented for WaitUntilFreshAndGet.
	if !recursive || !c.watchCache.storage.SnapshottingEnabled() {
		return delegator.Result{ShouldDelegate: true}, nil
	}
	_, continueRV, err := storage.DecodeContinue(continueToken, c.resourcePrefix)
	if err != nil {
		return delegator.Result{}, err
	}
	if continueRV > 0 {
		return c.shouldDelegateExactRV(uint64(continueRV))
	} else {
		// Continue with negative RV is a consistent read.
		return c.ShouldDelegateConsistentRead()
	}
}

func (c *Cacher) shouldDelegateExactRV(rv uint64) (delegator.Result, error) {
	// Exact requests on future revision require support for consistent read, but are not a consistent read by themselves.
	if c.watchCache.notFresh(rv) {
		return delegator.Result{
			ShouldDelegate: !delegator.ConsistentReadSupported(),
		}, nil
	}
	canServe := c.watchCache.storage.CanServeExactRV(rv)
	return delegator.Result{
		ShouldDelegate: !canServe,
	}, nil
}

func (c *Cacher) ShouldDelegateConsistentRead() (delegator.Result, error) {
	return delegator.Result{
		ConsistentRead: true,
		ShouldDelegate: !delegator.ConsistentReadSupported(),
	}, nil
}

// Implements watch.Interface.
func (c *errWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *errWatcher) Stop() {
	// no-op
}

// immediateCloseWatcher implements watch.Interface that is immediately closed
type immediateCloseWatcher struct {
	result chan watch.Event
}

func newImmediateCloseWatcher() *immediateCloseWatcher {
	watcher := &immediateCloseWatcher{result: make(chan watch.Event)}
	close(watcher.result)
	return watcher
}

// Implements watch.Interface.
func (c *immediateCloseWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *immediateCloseWatcher) Stop() {
	// no-op
}
