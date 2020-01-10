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

package cache

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/buffer"

	"k8s.io/klog"
)

// SharedInformer provides eventually consistent linkage of its
// clients to the authoritative state of a given collection of
// objects.  An object is identified by its API group, kind/resource,
// namespace, and name; the `ObjectMeta.UID` is not part of an
// object's ID as far as this contract is concerned.  One
// SharedInformer provides linkage to objects of a particular API
// group and kind/resource.  The linked object collection of a
// SharedInformer may be further restricted to one namespace and/or by
// label selector and/or field selector.
//
// The authoritative state of an object is what apiservers provide
// access to, and an object goes through a strict sequence of states.
// An object state is either "absent" or present with a
// ResourceVersion and other appropriate content.
//
// A SharedInformer gets object states from apiservers using a
// sequence of LIST and WATCH operations.  Through this sequence the
// apiservers provide a sequence of "collection states" to the
// informer, where each collection state defines the state of every
// object of the collection.  No promise --- beyond what is implied by
// other remarks here --- is made about how one informer's sequence of
// collection states relates to a different informer's sequence of
// collection states.
//
// A SharedInformer maintains a local cache, exposed by GetStore() and
// by GetIndexer() in the case of an indexed informer, of the state of
// each relevant object.  This cache is eventually consistent with the
// authoritative state.  This means that, unless prevented by
// persistent communication problems, if ever a particular object ID X
// is authoritatively associated with a state S then for every
// SharedInformer I whose collection includes (X, S) eventually either
// (1) I's cache associates X with S or a later state of X, (2) I is
// stopped, or (3) the authoritative state service for X terminates.
// To be formally complete, we say that the absent state meets any
// restriction by label selector or field selector.
//
// The local cache starts out empty, and gets populated and updated
// during `Run()`.
//
// As a simple example, if a collection of objects is henceforth
// unchanging, a SharedInformer is created that links to that
// collection, and that SharedInformer is `Run()` then that
// SharedInformer's cache eventually holds an exact copy of that
// collection (unless it is stopped too soon, the authoritative state
// service ends, or communication problems between the two
// persistently thwart achievement).
//
// As another simple example, if the local cache ever holds a
// non-absent state for some object ID and the object is eventually
// removed from the authoritative state then eventually the object is
// removed from the local cache (unless the SharedInformer is stopped
// too soon, the authoritative state service ends, or communication
// problems persistently thwart the desired result).
//
// The keys in the Store are of the form namespace/name for namespaced
// objects, and are simply the name for non-namespaced objects.
// Clients can use `MetaNamespaceKeyFunc(obj)` to extract the key for
// a given object, and `SplitMetaNamespaceKey(key)` to split a key
// into its constituent parts.
//
// A client is identified here by a ResourceEventHandler.  For every
// update to the SharedInformer's local cache and for every client
// added before `Run()`, eventually either the SharedInformer is
// stopped or the client is notified of the update.  A client added
// after `Run()` starts gets a startup batch of notifications of
// additions of the object existing in the cache at the time that
// client was added; also, for every update to the SharedInformer's
// local cache after that client was added, eventually either the
// SharedInformer is stopped or that client is notified of that
// update.  Client notifications happen after the corresponding cache
// update and, in the case of a SharedIndexInformer, after the
// corresponding index updates.  It is possible that additional cache
// and index updates happen before such a prescribed notification.
// For a given SharedInformer and client, the notifications are
// delivered sequentially.  For a given SharedInformer, client, and
// object ID, the notifications are delivered in order.  Because
// `ObjectMeta.UID` has no role in identifying objects, it is possible
// that when (1) object O1 with ID (e.g. namespace and name) X and
// `ObjectMeta.UID` U1 in the SharedInformer's local cache is deleted
// and later (2) another object O2 with ID X and ObjectMeta.UID U2 is
// created the informer's clients are not notified of (1) and (2) but
// rather are notified only of an update from O1 to O2. Clients that
// need to detect such cases might do so by comparing the `ObjectMeta.UID`
// field of the old and the new object in the code that handles update
// notifications (i.e. `OnUpdate` method of ResourceEventHandler).
//
// A client must process each notification promptly; a SharedInformer
// is not engineered to deal well with a large backlog of
// notifications to deliver.  Lengthy processing should be passed off
// to something else, for example through a
// `client-go/util/workqueue`.
//
// Each query to an informer's local cache --- whether a single-object
// lookup, a list operation, or a use of one of its indices --- is
// answered entirely from one of the collection states received by
// that informer.
//
// A delete notification exposes the last locally known non-absent
// state, except that its ResourceVersion is replaced with a
// ResourceVersion in which the object is actually absent.
type SharedInformer interface {
	// AddEventHandler adds an event handler to the shared informer using the shared informer's resync
	// period.  Events to a single handler are delivered sequentially, but there is no coordination
	// between different handlers.
	AddEventHandler(handler ResourceEventHandler)
	// AddEventHandlerWithResyncPeriod adds an event handler to the
	// shared informer using the specified resync period.  The resync
	// operation consists of delivering to the handler an update
	// notification for every object in the informer's local cache; it
	// does not add any interactions with the authoritative storage.
	AddEventHandlerWithResyncPeriod(handler ResourceEventHandler, resyncPeriod time.Duration)
	// GetStore returns the informer's local cache as a Store.
	GetStore() Store
	// GetController is deprecated, it does nothing useful
	GetController() Controller
	// Run starts and runs the shared informer, returning after it stops.
	// The informer will be stopped when stopCh is closed.
	Run(stopCh <-chan struct{})
	// HasSynced returns true if the shared informer's store has been
	// informed by at least one full LIST of the authoritative state
	// of the informer's object collection.  This is unrelated to "resync".
	HasSynced() bool
	// LastSyncResourceVersion is the resource version observed when last synced with the underlying
	// store. The value returned is not synchronized with access to the underlying store and is not
	// thread-safe.
	LastSyncResourceVersion() string
}

// SharedIndexInformer provides add and get Indexers ability based on SharedInformer.
type SharedIndexInformer interface {
	SharedInformer
	// AddIndexers add indexers to the informer before it starts.
	AddIndexers(indexers Indexers) error
	GetIndexer() Indexer
}

// NewSharedInformer creates a new instance for the listwatcher.
func NewSharedInformer(lw ListerWatcher, exampleObject runtime.Object, resyncPeriod time.Duration) SharedInformer {
	return NewSharedIndexInformer(lw, exampleObject, resyncPeriod, Indexers{})
}

// NewSharedIndexInformer creates a new instance for the listwatcher.
func NewSharedIndexInformer(lw ListerWatcher, exampleObject runtime.Object, defaultEventHandlerResyncPeriod time.Duration, indexers Indexers) SharedIndexInformer {
	realClock := &clock.RealClock{}
	sharedIndexInformer := &sharedIndexInformer{
		processor:                       &sharedProcessor{clock: realClock},
		indexer:                         NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers),
		listerWatcher:                   lw,
		objectType:                      exampleObject,
		resyncCheckPeriod:               defaultEventHandlerResyncPeriod,
		defaultEventHandlerResyncPeriod: defaultEventHandlerResyncPeriod,
		cacheMutationDetector:           NewCacheMutationDetector(fmt.Sprintf("%T", exampleObject)),
		clock:                           realClock,
	}
	return sharedIndexInformer
}

// InformerSynced is a function that can be used to determine if an informer has synced.  This is useful for determining if caches have synced.
type InformerSynced func() bool

const (
	// syncedPollPeriod controls how often you look at the status of your sync funcs
	syncedPollPeriod = 100 * time.Millisecond

	// initialBufferSize is the initial number of event notifications that can be buffered.
	initialBufferSize = 1024
)

// WaitForNamedCacheSync is a wrapper around WaitForCacheSync that generates log messages
// indicating that the caller identified by name is waiting for syncs, followed by
// either a successful or failed sync.
func WaitForNamedCacheSync(controllerName string, stopCh <-chan struct{}, cacheSyncs ...InformerSynced) bool {
	klog.Infof("Waiting for caches to sync for %s", controllerName)

	if !WaitForCacheSync(stopCh, cacheSyncs...) {
		utilruntime.HandleError(fmt.Errorf("unable to sync caches for %s", controllerName))
		return false
	}

	klog.Infof("Caches are synced for %s ", controllerName)
	return true
}

// WaitForCacheSync waits for caches to populate.  It returns true if it was successful, false
// if the controller should shutdown
// callers should prefer WaitForNamedCacheSync()
func WaitForCacheSync(stopCh <-chan struct{}, cacheSyncs ...InformerSynced) bool {
	err := wait.PollImmediateUntil(syncedPollPeriod,
		func() (bool, error) {
			for _, syncFunc := range cacheSyncs {
				if !syncFunc() {
					return false, nil
				}
			}
			return true, nil
		},
		stopCh)
	if err != nil {
		klog.V(2).Infof("stop requested")
		return false
	}

	klog.V(4).Infof("caches populated")
	return true
}

// `*sharedIndexInformer` implements SharedIndexInformer and has three
// main components.  One is an indexed local cache, `indexer Indexer`.
// The second main component is a Controller that pulls
// objects/notifications using the ListerWatcher and pushes them into
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
	indexer    Indexer
	controller Controller

	processor             *sharedProcessor
	cacheMutationDetector MutationDetector

	listerWatcher ListerWatcher

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
	newObj interface{}
}

type deleteNotification struct {
	oldObj interface{}
}

func (s *sharedIndexInformer) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	fifo := NewDeltaFIFO(MetaNamespaceKeyFunc, s.indexer)

	cfg := &Config{
		Queue:            fifo,
		ListerWatcher:    s.listerWatcher,
		ObjectType:       s.objectType,
		FullResyncPeriod: s.resyncCheckPeriod,
		RetryOnError:     false,
		ShouldResync:     s.processor.shouldResync,

		Process: s.HandleDeltas,
	}

	func() {
		s.startedLock.Lock()
		defer s.startedLock.Unlock()

		s.controller = New(cfg)
		s.controller.(*controller).clock = s.clock
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

func (s *sharedIndexInformer) GetStore() Store {
	return s.indexer
}

func (s *sharedIndexInformer) GetIndexer() Indexer {
	return s.indexer
}

func (s *sharedIndexInformer) AddIndexers(indexers Indexers) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.started {
		return fmt.Errorf("informer has already started")
	}

	return s.indexer.AddIndexers(indexers)
}

func (s *sharedIndexInformer) GetController() Controller {
	return &dummyController{informer: s}
}

func (s *sharedIndexInformer) AddEventHandler(handler ResourceEventHandler) {
	s.AddEventHandlerWithResyncPeriod(handler, s.defaultEventHandlerResyncPeriod)
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

func (s *sharedIndexInformer) AddEventHandlerWithResyncPeriod(handler ResourceEventHandler, resyncPeriod time.Duration) {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if s.stopped {
		klog.V(2).Infof("Handler %v was not added to shared informer because it has stopped already", handler)
		return
	}

	if resyncPeriod > 0 {
		if resyncPeriod < minimumResyncPeriod {
			klog.Warningf("resyncPeriod %d is too small. Changing it to the minimum allowed value of %d", resyncPeriod, minimumResyncPeriod)
			resyncPeriod = minimumResyncPeriod
		}

		if resyncPeriod < s.resyncCheckPeriod {
			if s.started {
				klog.Warningf("resyncPeriod %d is smaller than resyncCheckPeriod %d and the informer has already started. Changing it to %d", resyncPeriod, s.resyncCheckPeriod, s.resyncCheckPeriod)
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

	listener := newProcessListener(handler, resyncPeriod, determineResyncPeriod(resyncPeriod, s.resyncCheckPeriod), s.clock.Now(), initialBufferSize)

	if !s.started {
		s.processor.addListener(listener)
		return
	}

	// in order to safely join, we have to
	// 1. stop sending add/update/delete notifications
	// 2. do a list against the store
	// 3. send synthetic "Add" events to the new handler
	// 4. unblock
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	s.processor.addListener(listener)
	for _, item := range s.indexer.List() {
		listener.add(addNotification{newObj: item})
	}
}

func (s *sharedIndexInformer) HandleDeltas(obj interface{}) error {
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	// from oldest to newest
	for _, d := range obj.(Deltas) {
		switch d.Type {
		case Sync, Added, Updated:
			isSync := d.Type == Sync
			s.cacheMutationDetector.AddObject(d.Object)
			if old, exists, err := s.indexer.Get(d.Object); err == nil && exists {
				if err := s.indexer.Update(d.Object); err != nil {
					return err
				}
				s.processor.distribute(updateNotification{oldObj: old, newObj: d.Object}, isSync)
			} else {
				if err := s.indexer.Add(d.Object); err != nil {
					return err
				}
				s.processor.distribute(addNotification{newObj: d.Object}, isSync)
			}
		case Deleted:
			if err := s.indexer.Delete(d.Object); err != nil {
				return err
			}
			s.processor.distribute(deleteNotification{oldObj: d.Object}, false)
		}
	}
	return nil
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
	listeners        []*processorListener
	syncingListeners []*processorListener
	clock            clock.Clock
	wg               wait.Group
}

func (p *sharedProcessor) addListener(listener *processorListener) {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	p.addListenerLocked(listener)
	if p.listenersStarted {
		p.wg.Start(listener.run)
		p.wg.Start(listener.pop)
	}
}

func (p *sharedProcessor) addListenerLocked(listener *processorListener) {
	p.listeners = append(p.listeners, listener)
	p.syncingListeners = append(p.syncingListeners, listener)
}

func (p *sharedProcessor) distribute(obj interface{}, sync bool) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	if sync {
		for _, listener := range p.syncingListeners {
			listener.add(obj)
		}
	} else {
		for _, listener := range p.listeners {
			listener.add(obj)
		}
	}
}

func (p *sharedProcessor) run(stopCh <-chan struct{}) {
	func() {
		p.listenersLock.RLock()
		defer p.listenersLock.RUnlock()
		for _, listener := range p.listeners {
			p.wg.Start(listener.run)
			p.wg.Start(listener.pop)
		}
		p.listenersStarted = true
	}()
	<-stopCh
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()
	for _, listener := range p.listeners {
		close(listener.addCh) // Tell .pop() to stop. .pop() will tell .run() to stop
	}
	p.wg.Wait() // Wait for all .pop() and .run() to stop
}

// shouldResync queries every listener to determine if any of them need a resync, based on each
// listener's resyncPeriod.
func (p *sharedProcessor) shouldResync() bool {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	p.syncingListeners = []*processorListener{}

	resyncNeeded := false
	now := p.clock.Now()
	for _, listener := range p.listeners {
		// need to loop through all the listeners to see if they need to resync so we can prepare any
		// listeners that are going to be resyncing.
		if listener.shouldResync(now) {
			resyncNeeded = true
			p.syncingListeners = append(p.syncingListeners, listener)
			listener.determineNextResync(now)
		}
	}
	return resyncNeeded
}

func (p *sharedProcessor) resyncCheckPeriodChanged(resyncCheckPeriod time.Duration) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for _, listener := range p.listeners {
		resyncPeriod := determineResyncPeriod(listener.requestedResyncPeriod, resyncCheckPeriod)
		listener.setResyncPeriod(resyncPeriod)
	}
}

// processorListener relays notifications from a sharedProcessor to
// one ResourceEventHandler --- using two goroutines, two unbuffered
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

	handler ResourceEventHandler

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
	// sharedProcessor's `resyncCheckPeriod`, that is imposed (a) only
	// in AddEventHandlerWithResyncPeriod invocations made after the
	// sharedProcessor starts and (b) only if the informer does
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

func newProcessListener(handler ResourceEventHandler, requestedResyncPeriod, resyncPeriod time.Duration, now time.Time, bufferSize int) *processorListener {
	ret := &processorListener{
		nextCh:                make(chan interface{}),
		addCh:                 make(chan interface{}),
		handler:               handler,
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
	// we will catch it, **the offending item will be skipped!**, and after a short delay (one minute)
	// the next notification will be attempted.  This is usually better than the alternative of never
	// delivering again.
	stopCh := make(chan struct{})
	wait.Until(func() {
		// this gives us a few quick retries before a long pause and then a few more quick retries
		err := wait.ExponentialBackoff(retry.DefaultRetry, func() (bool, error) {
			for next := range p.nextCh {
				switch notification := next.(type) {
				case updateNotification:
					p.handler.OnUpdate(notification.oldObj, notification.newObj)
				case addNotification:
					p.handler.OnAdd(notification.newObj)
				case deleteNotification:
					p.handler.OnDelete(notification.oldObj)
				default:
					utilruntime.HandleError(fmt.Errorf("unrecognized notification: %T", next))
				}
			}
			// the only way to get here is if the p.nextCh is empty and closed
			return true, nil
		})

		// the only way to get here is if the p.nextCh is empty and closed
		if err == nil {
			close(stopCh)
		}
	}, 1*time.Minute, stopCh)
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
