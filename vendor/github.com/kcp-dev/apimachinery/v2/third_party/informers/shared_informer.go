/*
Copyright 2015 The Kubernetes Authors.
Modifications Copyright 2022 The KCP Authors.

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

package informers

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/cache/synctrack"
	"k8s.io/klog/v2"
	"k8s.io/utils/buffer"
	"k8s.io/utils/clock"

	kcpcache "github.com/kcp-dev/apimachinery/v2/pkg/cache"
)

// NewSharedInformer creates a new instance for the listwatcher.
func NewSharedInformer(lw cache.ListerWatcher, exampleObject runtime.Object, defaultEventHandlerResyncPeriod time.Duration) cache.SharedInformer {
	return NewSharedIndexInformer(lw, exampleObject, defaultEventHandlerResyncPeriod, cache.Indexers{})
}

// NewSharedIndexInformer creates a new instance for the listwatcher.
// The created informer will not do resyncs if the given
// defaultEventHandlerResyncPeriod is zero.  Otherwise: for each
// handler that with a non-zero requested resync period, whether added
// before or after the informer starts, the nominal resync period is
// the requested resync period rounded up to a multiple of the
// informer's resync checking period.  Such an informer's resync
// checking period is established when the informer starts running,
// and is the maximum of (a) the minimum of the resync periods
// requested before the informer starts and the
// defaultEventHandlerResyncPeriod given here and (b) the constant
// `minimumResyncPeriod` defined in this file.
func NewSharedIndexInformer(lw cache.ListerWatcher, exampleObject runtime.Object, defaultEventHandlerResyncPeriod time.Duration, indexers cache.Indexers) kcpcache.ScopeableSharedIndexInformer {
	realClock := &clock.RealClock{}
	sharedIndexInformer := &sharedIndexInformer{
		processor: &sharedProcessor{clock: realClock},
		// KCP modification: We changed the keyfunction passed to NewIndexer
		indexer:                         cache.NewIndexer(kcpcache.MetaClusterNamespaceKeyFunc, indexers),
		listerWatcher:                   lw,
		objectType:                      exampleObject,
		resyncCheckPeriod:               defaultEventHandlerResyncPeriod,
		defaultEventHandlerResyncPeriod: defaultEventHandlerResyncPeriod,
		cacheMutationDetector:           cache.NewCacheMutationDetector(fmt.Sprintf("%T", exampleObject)),
		clock:                           realClock,
	}
	return sharedIndexInformer
}

const (
	// syncedPollPeriod controls how often you look at the status of your sync funcs
	syncedPollPeriod = 100 * time.Millisecond

	// initialBufferSize is the initial number of event notifications that can be buffered.
	initialBufferSize = 1024
)

// `*sharedIndexInformer` implements SharedIndexInformer and has three
// main components.  One is an indexed local cache, `indexer cache.Indexer`.
// The second main component is a cache.Controller that pulls
// objects/notifications using the cache.ListerWatcher and pushes them into
// a DeltaFIFO --- whose knownObjects is the informer's local cache
// --- while concurrently Popping Deltas values from that fifo and
// processing them with `sharedIndexInformer::HandleDeltas`.  Each
// invocation of HandleDeltas, which is done with the fifo's lock
// held, processes each Delta in turn.  For each Delta this both
// updates the local cache and stuffs the relevant notification into
// the sharedProcessor.  The third main component is that
// sharedProcessor, which is responsible for relaying those
// notifications to each of the informer's clients.
type sharedIndexInformer struct {
	indexer    cache.Indexer
	controller cache.Controller

	processor             *sharedProcessor
	cacheMutationDetector cache.MutationDetector

	listerWatcher cache.ListerWatcher

	// objectType is an example object of the type this informer is
	// expected to handle.  Only the type needs to be right, except
	// that when that is `unstructured.Unstructured` the object's
	// `"apiVersion"` and `"kind"` must also be right.
	objectType runtime.Object

	// resyncCheckPeriod is how often we want the reflector's resync timer to fire so it can call
	// shouldResync to check if any of our listeners need a resync.
	resyncCheckPeriod time.Duration
	// defaultEventHandlerResyncPeriod is the default resync period for any handlers added via
	// AddEventHandler (i.e. they don't specify one and just want to use the shared informer's default
	// value).
	defaultEventHandlerResyncPeriod time.Duration
	// clock allows for testability
	clock clock.Clock

	started, stopped bool
	startedLock      sync.Mutex

	// blockDeltas gives a way to stop all event distribution so that a late event handler
	// can safely join the shared informer.
	blockDeltas sync.Mutex

	// Called whenever the ListAndWatch drops the connection with an error.
	watchErrorHandler cache.WatchErrorHandler

	transform cache.TransformFunc
}

func (s *sharedIndexInformer) Cluster(cluster logicalcluster.Name) cache.SharedIndexInformer {
	return &scopedSharedIndexInformer{
		sharedIndexInformer: s,
		clusterName:         cluster,
	}
}

// dummyController hides the fact that a SharedInformer is different from a dedicated one
// where a caller can `Run`.  The run method is disconnected in this case, because higher
// level logic will decide when to start the SharedInformer and related controller.
// Because returning information back is always asynchronous, the legacy callers shouldn't
// notice any change in behavior.
type dummyController struct {
	informer *sharedIndexInformer
}

func (v *dummyController) Run(stopCh <-chan struct{}) {
}

func (v *dummyController) HasSynced() bool {
	return v.informer.HasSynced()
}

func (v *dummyController) LastSyncResourceVersion() string {
	return ""
}

type updateNotification struct {
	oldObj interface{}
	newObj interface{}
}

type addNotification struct {
	newObj          interface{}
	isInInitialList bool
}

type deleteNotification struct {
	oldObj interface{}
}

func (s *sharedIndexInformer) SetWatchErrorHandler(handler cache.WatchErrorHandler) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.started {
		return fmt.Errorf("informer has already started")
	}

	s.watchErrorHandler = handler
	return nil
}

func (s *sharedIndexInformer) SetTransform(handler cache.TransformFunc) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.started {
		return fmt.Errorf("informer has already started")
	}

	s.transform = handler
	return nil
}

func (s *sharedIndexInformer) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	if s.HasStarted() {
		klog.Warningf("The sharedIndexInformer has started, run more than once is not allowed")
		return
	}
	fifo := cache.NewDeltaFIFOWithOptions(cache.DeltaFIFOOptions{
		KnownObjects:          s.indexer,
		EmitDeltaTypeReplaced: true,
		// KCP modification: We changed the keyfunction passed to NewDeltaFIFOWithOptions
		KeyFunction: kcpcache.MetaClusterNamespaceKeyFunc,
	})

	cfg := &cache.Config{
		Queue:            fifo,
		ListerWatcher:    s.listerWatcher,
		ObjectType:       s.objectType,
		FullResyncPeriod: s.resyncCheckPeriod,
		RetryOnError:     false,
		ShouldResync:     s.processor.shouldResync,

		Process:           s.HandleDeltas,
		WatchErrorHandler: s.watchErrorHandler,
	}

	func() {
		s.startedLock.Lock()
		defer s.startedLock.Unlock()

		s.controller = cache.New(cfg)

		// KCP modification: we removed setting the s.controller.clock here as it's an unexported field we can't access

		s.started = true
	}()

	// Separate stop channel because Processor should be stopped strictly after controller
	processorStopCh := make(chan struct{})
	var wg wait.Group
	defer wg.Wait()              // Wait for Processor to stop
	defer close(processorStopCh) // Tell Processor to stop
	wg.StartWithChannel(processorStopCh, s.cacheMutationDetector.Run)
	wg.StartWithChannel(processorStopCh, s.processor.run)

	defer func() {
		s.startedLock.Lock()
		defer s.startedLock.Unlock()
		s.stopped = true // Don't want any new listeners
	}()
	s.controller.Run(stopCh)
}

func (s *sharedIndexInformer) HasStarted() bool {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()
	return s.started
}

func (s *sharedIndexInformer) HasSynced() bool {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.controller == nil {
		return false
	}
	return s.controller.HasSynced()
}

func (s *sharedIndexInformer) LastSyncResourceVersion() string {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.controller == nil {
		return ""
	}
	return s.controller.LastSyncResourceVersion()
}

func (s *sharedIndexInformer) GetStore() cache.Store {
	return s.indexer
}

func (s *sharedIndexInformer) GetIndexer() cache.Indexer {
	return s.indexer
}

func (s *sharedIndexInformer) AddIndexers(indexers cache.Indexers) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.stopped {
		return fmt.Errorf("indexer was not added because it has stopped already")
	}

	return s.indexer.AddIndexers(indexers)
}

func (s *sharedIndexInformer) GetController() cache.Controller {
	return &dummyController{informer: s}
}

func (s *sharedIndexInformer) AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error) {
	return s.AddEventHandlerWithResyncPeriod(handler, s.defaultEventHandlerResyncPeriod)
}

func determineResyncPeriod(desired, check time.Duration) time.Duration {
	if desired == 0 {
		return desired
	}
	if check == 0 {
		klog.Warningf("The specified resyncPeriod %v is invalid because this shared informer doesn't support resyncing", desired)
		return 0
	}
	if desired < check {
		klog.Warningf("The specified resyncPeriod %v is being increased to the minimum resyncCheckPeriod %v", desired, check)
		return check
	}
	return desired
}

const minimumResyncPeriod = 1 * time.Second

func (s *sharedIndexInformer) AddEventHandlerWithResyncPeriod(handler cache.ResourceEventHandler, resyncPeriod time.Duration) (cache.ResourceEventHandlerRegistration, error) {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.stopped {
		return nil, fmt.Errorf("handler %v was not added to shared informer because it has stopped already", handler)
	}

	if resyncPeriod > 0 {
		if resyncPeriod < minimumResyncPeriod {
			klog.Warningf("resyncPeriod %v is too small. Changing it to the minimum allowed value of %v", resyncPeriod, minimumResyncPeriod)
			resyncPeriod = minimumResyncPeriod
		}

		if resyncPeriod < s.resyncCheckPeriod {
			if s.started {
				klog.Warningf("resyncPeriod %v is smaller than resyncCheckPeriod %v and the informer has already started. Changing it to %v", resyncPeriod, s.resyncCheckPeriod, s.resyncCheckPeriod)
				resyncPeriod = s.resyncCheckPeriod
			} else {
				// if the event handler's resyncPeriod is smaller than the current resyncCheckPeriod, update
				// resyncCheckPeriod to match resyncPeriod and adjust the resync periods of all the listeners
				// accordingly
				s.resyncCheckPeriod = resyncPeriod
				s.processor.resyncCheckPeriodChanged(resyncPeriod)
			}
		}
	}

	listener := newProcessListener(handler, resyncPeriod, determineResyncPeriod(resyncPeriod, s.resyncCheckPeriod), s.clock.Now(), initialBufferSize, func() bool { return true })

	if !s.started {
		return s.processor.addListener(listener), nil
	}

	// in order to safely join, we have to
	// 1. stop sending add/update/delete notifications
	// 2. do a list against the store
	// 3. send synthetic "Add" events to the new handler
	// 4. unblock
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	handle := s.processor.addListener(listener)
	for _, item := range s.indexer.List() {
		listener.add(addNotification{newObj: item})
	}

	return handle, nil
}

func (s *sharedIndexInformer) HandleDeltas(obj interface{}, isInInitialList bool) error {
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	if deltas, ok := obj.(cache.Deltas); ok {
		return processDeltas(s, s.indexer, s.transform, deltas, isInInitialList)
	}
	return errors.New("object given as Process argument is not Deltas")
}

func (s *sharedIndexInformer) RemoveEventHandler(handle cache.ResourceEventHandlerRegistration) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	// in order to safely remove, we have to
	// 1. stop sending add/update/delete notifications
	// 2. remove and stop listener
	// 3. unblock
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()
	return s.processor.removeListener(handle)
}

// Conforms to cache.ResourceEventHandler
func (s *sharedIndexInformer) OnAdd(obj interface{}, isInInitialList bool) {
	// Invocation of this function is locked under s.blockDeltas, so it is
	// save to distribute the notification
	s.cacheMutationDetector.AddObject(obj)
	s.processor.distribute(addNotification{newObj: obj, isInInitialList: isInInitialList}, false)
}

// Conforms to cache.ResourceEventHandler
func (s *sharedIndexInformer) OnUpdate(old, new interface{}) {
	isSync := false

	// If is a Sync event, isSync should be true
	// If is a Replaced event, isSync is true if resource version is unchanged.
	// If RV is unchanged: this is a Sync/Replaced event, so isSync is true

	if accessor, err := meta.Accessor(new); err == nil {
		if oldAccessor, err := meta.Accessor(old); err == nil {
			// Events that didn't change resourceVersion are treated as resync events
			// and only propagated to listeners that requested resync
			isSync = accessor.GetResourceVersion() == oldAccessor.GetResourceVersion()
		}
	}

	// Invocation of this function is locked under s.blockDeltas, so it is
	// save to distribute the notification
	s.cacheMutationDetector.AddObject(new)
	s.processor.distribute(updateNotification{oldObj: old, newObj: new}, isSync)
}

// Conforms to cache.ResourceEventHandler
func (s *sharedIndexInformer) OnDelete(old interface{}) {
	// Invocation of this function is locked under s.blockDeltas, so it is
	// save to distribute the notification
	s.processor.distribute(deleteNotification{oldObj: old}, false)
}

// IsStopped reports whether the informer has already been stopped
func (s *sharedIndexInformer) IsStopped() bool {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()
	return s.stopped
}

// sharedProcessor has a collection of processorListener and can
// distribute a notification object to its listeners.  There are two
// kinds of distribute operations.  The sync distributions go to a
// subset of the listeners that (a) is recomputed in the occasional
// calls to shouldResync and (b) every listener is initially put in.
// The non-sync distributions go to every listener.
type sharedProcessor struct {
	listenersStarted bool
	listenersLock    sync.RWMutex
	// Map from listeners to whether or not they are currently syncing
	listeners map[*processorListener]bool
	clock     clock.Clock
	wg        wait.Group
}

func (p *sharedProcessor) getListener(registration cache.ResourceEventHandlerRegistration) *processorListener {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	if p.listeners == nil {
		return nil
	}

	if result, ok := registration.(*processorListener); ok {
		if _, exists := p.listeners[result]; exists {
			return result
		}
	}

	return nil
}

func (p *sharedProcessor) addListener(listener *processorListener) cache.ResourceEventHandlerRegistration {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	if p.listeners == nil {
		p.listeners = make(map[*processorListener]bool)
	}

	p.listeners[listener] = true

	if p.listenersStarted {
		p.wg.Start(listener.run)
		p.wg.Start(listener.pop)
	}

	return listener
}

func (p *sharedProcessor) removeListener(handle cache.ResourceEventHandlerRegistration) error {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	listener, ok := handle.(*processorListener)
	if !ok {
		return fmt.Errorf("invalid key type %t", handle)
	} else if p.listeners == nil {
		// No listeners are registered, do nothing
		return nil
	} else if _, exists := p.listeners[listener]; !exists {
		// Listener is not registered, just do nothing
		return nil
	}

	delete(p.listeners, listener)

	if p.listenersStarted {
		close(listener.addCh)
	}

	return nil
}

func (p *sharedProcessor) distribute(obj interface{}, sync bool) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for listener, isSyncing := range p.listeners {
		switch {
		case !sync:
			// non-sync messages are delivered to every listener
			listener.add(obj)
		case isSyncing:
			// sync messages are delivered to every syncing listener
			listener.add(obj)
		default:
			// skipping a sync obj for a non-syncing listener
		}
	}
}

func (p *sharedProcessor) run(stopCh <-chan struct{}) {
	func() {
		p.listenersLock.RLock()
		defer p.listenersLock.RUnlock()
		for listener := range p.listeners {
			p.wg.Start(listener.run)
			p.wg.Start(listener.pop)
		}
		p.listenersStarted = true
	}()
	<-stopCh

	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()
	for listener := range p.listeners {
		close(listener.addCh) // Tell .pop() to stop. .pop() will tell .run() to stop
	}

	// Wipe out list of listeners since they are now closed
	// (processorListener cannot be re-used)
	p.listeners = nil

	// Reset to false since no listeners are running
	p.listenersStarted = false

	p.wg.Wait() // Wait for all .pop() and .run() to stop
}

// shouldResync queries every listener to determine if any of them need a resync, based on each
// listener's resyncPeriod.
func (p *sharedProcessor) shouldResync() bool {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	resyncNeeded := false
	now := p.clock.Now()
	for listener := range p.listeners {
		// need to loop through all the listeners to see if they need to resync so we can prepare any
		// listeners that are going to be resyncing.
		shouldResync := listener.shouldResync(now)
		p.listeners[listener] = shouldResync

		if shouldResync {
			resyncNeeded = true
			listener.determineNextResync(now)
		}
	}
	return resyncNeeded
}

func (p *sharedProcessor) resyncCheckPeriodChanged(resyncCheckPeriod time.Duration) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for listener := range p.listeners {
		resyncPeriod := determineResyncPeriod(listener.requestedResyncPeriod, resyncCheckPeriod)
		listener.setResyncPeriod(resyncPeriod)
	}
}

// processorListener relays notifications from a sharedProcessor to
// one cache.ResourceEventHandler --- using two goroutines, two unbuffered
// channels, and an unbounded ring buffer.  The `add(notification)`
// function sends the given notification to `addCh`.  One goroutine
// runs `pop()`, which pumps notifications from `addCh` to `nextCh`
// using storage in the ring buffer while `nextCh` is not keeping up.
// Another goroutine runs `run()`, which receives notifications from
// `nextCh` and synchronously invokes the appropriate handler method.
//
// processorListener also keeps track of the adjusted requested resync
// period of the listener.
type processorListener struct {
	nextCh chan interface{}
	addCh  chan interface{}

	handler cache.ResourceEventHandler

	syncTracker *synctrack.SingleFileTracker

	// pendingNotifications is an unbounded ring buffer that holds all notifications not yet distributed.
	// There is one per listener, but a failing/stalled listener will have infinite pendingNotifications
	// added until we OOM.
	// TODO: This is no worse than before, since reflectors were backed by unbounded DeltaFIFOs, but
	// we should try to do something better.
	pendingNotifications buffer.RingGrowing

	// requestedResyncPeriod is how frequently the listener wants a
	// full resync from the shared informer, but modified by two
	// adjustments.  One is imposing a lower bound,
	// `minimumResyncPeriod`.  The other is another lower bound, the
	// sharedIndexInformer's `resyncCheckPeriod`, that is imposed (a) only
	// in AddEventHandlerWithResyncPeriod invocations made after the
	// sharedIndexInformer starts and (b) only if the informer does
	// resyncs at all.
	requestedResyncPeriod time.Duration
	// resyncPeriod is the threshold that will be used in the logic
	// for this listener.  This value differs from
	// requestedResyncPeriod only when the sharedIndexInformer does
	// not do resyncs, in which case the value here is zero.  The
	// actual time between resyncs depends on when the
	// sharedProcessor's `shouldResync` function is invoked and when
	// the sharedIndexInformer processes `Sync` type Delta objects.
	resyncPeriod time.Duration
	// nextResync is the earliest time the listener should get a full resync
	nextResync time.Time
	// resyncLock guards access to resyncPeriod and nextResync
	resyncLock sync.Mutex
}

// HasSynced returns true if the source informer has synced, and all
// corresponding events have been delivered.
func (p *processorListener) HasSynced() bool {
	return p.syncTracker.HasSynced()
}

func newProcessListener(handler cache.ResourceEventHandler, requestedResyncPeriod, resyncPeriod time.Duration, now time.Time, bufferSize int, hasSynced func() bool) *processorListener {
	ret := &processorListener{
		nextCh:                make(chan interface{}),
		addCh:                 make(chan interface{}),
		handler:               handler,
		syncTracker:           &synctrack.SingleFileTracker{UpstreamHasSynced: hasSynced},
		pendingNotifications:  *buffer.NewRingGrowing(bufferSize),
		requestedResyncPeriod: requestedResyncPeriod,
		resyncPeriod:          resyncPeriod,
	}

	ret.determineNextResync(now)

	return ret
}

func (p *processorListener) add(notification interface{}) {
	p.addCh <- notification
}

func (p *processorListener) pop() {
	defer utilruntime.HandleCrash()
	defer close(p.nextCh) // Tell .run() to stop

	var nextCh chan<- interface{}
	var notification interface{}
	for {
		select {
		case nextCh <- notification:
			// Notification dispatched
			var ok bool
			notification, ok = p.pendingNotifications.ReadOne()
			if !ok { // Nothing to pop
				nextCh = nil // Disable this select case
			}
		case notificationToAdd, ok := <-p.addCh:
			if !ok {
				return
			}
			if notification == nil { // No notification to pop (and pendingNotifications is empty)
				// Optimize the case - skip adding to pendingNotifications
				notification = notificationToAdd
				nextCh = p.nextCh
			} else { // There is already a notification waiting to be dispatched
				p.pendingNotifications.WriteOne(notificationToAdd)
			}
		}
	}
}

func (p *processorListener) run() {
	// this call blocks until the channel is closed.  When a panic happens during the notification
	// we will catch it, **the offending item will be skipped!**, and after a short delay (one second)
	// the next notification will be attempted.  This is usually better than the alternative of never
	// delivering again.
	stopCh := make(chan struct{})
	wait.Until(func() {
		for next := range p.nextCh {
			switch notification := next.(type) {
			case updateNotification:
				p.handler.OnUpdate(notification.oldObj, notification.newObj)
			case addNotification:
				p.handler.OnAdd(notification.newObj, notification.isInInitialList)
			case deleteNotification:
				p.handler.OnDelete(notification.oldObj)
			default:
				utilruntime.HandleError(fmt.Errorf("unrecognized notification: %T", next))
			}
		}
		// the only way to get here is if the p.nextCh is empty and closed
		close(stopCh)
	}, 1*time.Second, stopCh)
}

// shouldResync deterimines if the listener needs a resync. If the listener's resyncPeriod is 0,
// this always returns false.
func (p *processorListener) shouldResync(now time.Time) bool {
	p.resyncLock.Lock()
	defer p.resyncLock.Unlock()

	if p.resyncPeriod == 0 {
		return false
	}

	return now.After(p.nextResync) || now.Equal(p.nextResync)
}

func (p *processorListener) determineNextResync(now time.Time) {
	p.resyncLock.Lock()
	defer p.resyncLock.Unlock()

	p.nextResync = now.Add(p.resyncPeriod)
}

func (p *processorListener) setResyncPeriod(resyncPeriod time.Duration) {
	p.resyncLock.Lock()
	defer p.resyncLock.Unlock()

	p.resyncPeriod = resyncPeriod
}

// Multiplexes updates in the form of a list of Deltas into a Store, and informs
// a given handler of events OnUpdate, OnAdd, OnDelete
// taken from k8s.io/client-go/tools/cache/controller.go
// KCP modification: we added this function from controller.go
func processDeltas(
	// Object which receives event notifications from the given deltas
	handler cache.ResourceEventHandler,
	clientState cache.Store,
	transformer cache.TransformFunc,
	deltas cache.Deltas,
	isInInitialList bool,
) error {
	// from oldest to newest
	for _, d := range deltas {
		obj := d.Object
		if transformer != nil {
			var err error
			obj, err = transformer(obj)
			if err != nil {
				return err
			}
		}

		switch d.Type {
		case cache.Sync, cache.Replaced, cache.Added, cache.Updated:
			if old, exists, err := clientState.Get(obj); err == nil && exists {
				if err := clientState.Update(obj); err != nil {
					return err
				}
				handler.OnUpdate(old, obj)
			} else {
				if err := clientState.Add(obj); err != nil {
					return err
				}
				handler.OnAdd(obj, isInInitialList)
			}
		case cache.Deleted:
			if err := clientState.Delete(obj); err != nil {
				return err
			}
			handler.OnDelete(obj)
		}
	}
	return nil
}
