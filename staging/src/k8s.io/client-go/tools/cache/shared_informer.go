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
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/clock"

	"github.com/golang/glog"
)

// SharedInformer has a shared data cache and is capable of distributing notifications for changes
// to the cache to multiple listeners who registered via AddEventHandler. If you use this, there is
// one behavior change compared to a standard Informer.  When you receive a notification, the cache
// will be AT LEAST as fresh as the notification, but it MAY be more fresh.  You should NOT depend
// on the contents of the cache exactly matching the notification you've received in handler
// functions.  If there was a create, followed by a delete, the cache may NOT have your item.  This
// has advantages over the broadcaster since it allows us to share a common cache across many
// controllers. Extending the broadcaster would have required us keep duplicate caches for each
// watch.
type SharedInformer interface {
	// AddEventHandler adds an event handler to the shared informer using the shared informer's resync
	// period.  Events to a single handler are delivered sequentially, but there is no coordination
	// between different handlers.
	AddEventHandler(handler ResourceEventHandler)
	// AddEventHandlerWithResyncPeriod adds an event handler to the shared informer using the
	// specified resync period.  Events to a single handler are delivered sequentially, but there is
	// no coordination between different handlers.
	AddEventHandlerWithResyncPeriod(handler ResourceEventHandler, resyncPeriod time.Duration)
	// GetStore returns the Store.
	GetStore() Store
	// GetController gives back a synthetic interface that "votes" to start the informer
	GetController() Controller
	// Run starts the shared informer, which will be stopped when stopCh is closed.
	Run(stopCh <-chan struct{})
	// HasSynced returns true if the shared informer's store has synced.
	HasSynced() bool

	// LastSyncResourceVersion is the resource version observed when last synced with the underlying
	// store. The value returned is not synchronized with access to the underlying store and is not
	// thread-safe.
	LastSyncResourceVersion() string
}

type SharedIndexInformer interface {
	SharedInformer
	// AddIndexers add indexers to the informer before it starts.
	AddIndexers(indexers Indexers) error
	GetIndexer() Indexer
}

const DefaultResyncCheckPeriod = 5 * time.Minute

// NewSharedInformer creates a new instance for the listwatcher.
func NewSharedInformer(lw ListerWatcher, objType runtime.Object, resyncCheckPeriod, defaultEventHandlerResyncPeriod time.Duration) SharedInformer {
	return NewSharedIndexInformer(lw, objType, resyncCheckPeriod, defaultEventHandlerResyncPeriod, Indexers{})
}

// NewSharedIndexInformer creates a new instance for the listwatcher.
func NewSharedIndexInformer(lw ListerWatcher, objType runtime.Object, resyncCheckPeriod, defaultEventHandlerResyncPeriod time.Duration, indexers Indexers) SharedIndexInformer {
	realClock := &clock.RealClock{}
	sharedIndexInformer := &sharedIndexInformer{
		processor:                       &sharedProcessor{clock: realClock},
		indexer:                         NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers),
		listerWatcher:                   lw,
		objectType:                      objType,
		resyncCheckPeriod:               resyncCheckPeriod,
		defaultEventHandlerResyncPeriod: defaultEventHandlerResyncPeriod,
		cacheMutationDetector:           NewCacheMutationDetector(fmt.Sprintf("%T", objType)),
		clock: realClock,
	}
	return sharedIndexInformer
}

// InformerSynced is a function that can be used to determine if an informer has synced.  This is useful for determining if caches have synced.
type InformerSynced func() bool

// syncedPollPeriod controls how often you look at the status of your sync funcs
const syncedPollPeriod = 100 * time.Millisecond

// WaitForCacheSync waits for caches to populate.  It returns true if it was successful, false
// if the contoller should shutdown
func WaitForCacheSync(stopCh <-chan struct{}, cacheSyncs ...InformerSynced) bool {
	err := wait.PollUntil(syncedPollPeriod,
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
		glog.V(2).Infof("stop requested")
		return false
	}

	glog.V(4).Infof("caches populated")
	return true
}

type sharedIndexInformer struct {
	indexer    Indexer
	controller Controller

	processor             *sharedProcessor
	cacheMutationDetector CacheMutationDetector

	// This block is tracked to handle late initialization of the controller
	listerWatcher ListerWatcher
	objectType    runtime.Object

	// resyncCheckPeriod is how often we want the reflector's resync timer to fire so it can call
	// shouldResync to check if any of our listeners need a resync.
	resyncCheckPeriod time.Duration
	// defaultEventHandlerResyncPeriod is the default resync period for any handlers added via
	// AddEventHandler (i.e. they don't specify one and just want to use the shared informer's default
	// value).
	defaultEventHandlerResyncPeriod time.Duration
	// clock allows for testability
	clock clock.Clock

	started     bool
	startedLock sync.Mutex

	// blockDeltas gives a way to stop all event distribution so that a late event handler
	// can safely join the shared informer.
	blockDeltas sync.Mutex
	// stopCh is the channel used to stop the main Run process.  We have to track it so that
	// late joiners can have a proper stop
	stopCh <-chan struct{}
}

// dummyController hides the fact that a SharedInformer is different from a dedicated one
// where a caller can `Run`.  The run method is disonnected in this case, because higher
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

func (c *dummyController) LastSyncResourceVersion() string {
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

	fifo := NewDeltaFIFO(MetaNamespaceKeyFunc, nil, s.indexer)
	fifo.SyncRunning = s.syncRunning

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

	s.stopCh = stopCh
	s.cacheMutationDetector.Run(stopCh)
	s.processor.run(stopCh)
	s.controller.Run(stopCh)
}

func (s *sharedIndexInformer) isStarted() bool {
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
		glog.Warningf("The specified resyncPeriod %v is invalid because this shared informer doesn't support resyncing", desired)
		return 0
	}
	if desired < check {
		glog.Warningf("The specified resyncPeriod %v is being increased to the minimum resyncCheckPeriod %v", desired, check)
		return check
	}
	if desired%check != 0 {
		// e.g. if resyncPeriod is 5s and resyncCheckPeriod is 3s, we want (floor(5/3) * 3) + 3 = 6
		actual := desired / check
		actual *= check
		actual += check
		glog.Warningf("The specified resyncPeriod %v is not a multiple of resyncCheckPeriod %v. This event handler will resync every %v", desired, check, actual)
		return actual
	}
	return desired
}

func (s *sharedIndexInformer) AddEventHandlerWithResyncPeriod(handler ResourceEventHandler, resyncPeriod time.Duration) {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	listener := newProcessListener(handler, determineResyncPeriod(resyncPeriod, s.resyncCheckPeriod))
	listener.clock = s.clock

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

	go listener.run(s.stopCh)
	go listener.pop(s.stopCh)

	items := s.indexer.List()
	for i := range items {
		listener.add(addNotification{newObj: items[i]})
	}
}

// syncRunning is called at the start and end of a sync. It passes the
// event to the processor.
func (s *sharedIndexInformer) syncRunning(running bool) {
	s.processor.syncRunning(running)
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

type sharedProcessor struct {
	listenersLock sync.RWMutex
	listeners     []*processorListener
	clock         clock.Clock
}

func (p *sharedProcessor) addListener(listener *processorListener) {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	p.listeners = append(p.listeners, listener)
}

func (p *sharedProcessor) distribute(obj interface{}, isSync bool) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for _, listener := range p.listeners {
		if isSync && !listener.shouldHandleSyncItem() {
			// Skip if the listener doesn't want to resync now
			continue
		}
		listener.add(obj)
	}
}

// syncRunning is called at the start and end of a sync. It passes the
// event to all the listeners.
func (p *sharedProcessor) syncRunning(running bool) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for _, listener := range p.listeners {
		listener.syncRunning(running)
	}
}

func (p *sharedProcessor) run(stopCh <-chan struct{}) {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	for _, listener := range p.listeners {
		go listener.run(stopCh)
		go listener.pop(stopCh)
	}
}

// shouldResync queries every listener to determine if any of them need a resync, based on each
// listener's resyncPeriod.
func (p *sharedProcessor) shouldResync() bool {
	p.listenersLock.RLock()
	defer p.listenersLock.RUnlock()

	resyncNeeded := false
	now := p.clock.Now()
	for _, listener := range p.listeners {
		// need to loop through all the listeners to see if they need to resync so we can prepare any
		// listeners that are going to be resyncing.
		if listener.shouldResync(now) {
			resyncNeeded = true
			listener.prepareForResync(now)
		}
	}
	return resyncNeeded
}

type processorListener struct {
	// lock/cond protects access to 'pendingNotifications'.
	lock sync.RWMutex
	cond sync.Cond

	// pendingNotifications is an unbounded slice that holds all notifications not yet distributed
	// there is one per listener, but a failing/stalled listener will have infinite pendingNotifications
	// added until we OOM.
	// TODO This is no worse that before, since reflectors were backed by unbounded DeltaFIFOs, but
	// we should try to do something better
	pendingNotifications []interface{}

	nextCh chan interface{}

	handler ResourceEventHandler

	// clock exists to allow unit tests to simulate and control the time
	clock clock.Clock
	// resyncPeriod is how frequently the listener wants a full resync from the shared informer
	resyncPeriod time.Duration
	// nextResync is the earliest time the listener should get a full resync
	nextResync time.Time
	// wantSyncItems indicates if the listener should receive Sync items from the delta fifo (i.e.
	// during a resync)
	wantSyncItems bool
	// syncing is set to true for the duration of a resync, even if the listener's nextResync is in
	// the future
	syncing bool
	// hasSynced is set to true as soon as the listener receives its first sync item. In the event
	// that the shared informer's reflector's ListAndWatch is called multiple times, this allows the
	// listener to continue attempting to handle a Sync if it hasn't ever received anything.
	hasSynced bool
}

func newProcessListener(handler ResourceEventHandler, resyncPeriod time.Duration) *processorListener {
	ret := &processorListener{
		pendingNotifications: []interface{}{},
		nextCh:               make(chan interface{}),
		handler:              handler,
		resyncPeriod:         resyncPeriod,
		wantSyncItems:        true, // true because we want the initial sync (List)
	}

	ret.cond.L = &ret.lock
	return ret
}

func (p *processorListener) add(notification interface{}) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.pendingNotifications = append(p.pendingNotifications, notification)
	p.cond.Broadcast()
}

func (p *processorListener) pop(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	for {
		blockingGet := func() (interface{}, bool) {
			p.lock.Lock()
			defer p.lock.Unlock()

			for len(p.pendingNotifications) == 0 {
				// check if we're shutdown
				select {
				case <-stopCh:
					return nil, true
				default:
				}
				p.cond.Wait()
			}

			nt := p.pendingNotifications[0]
			p.pendingNotifications = p.pendingNotifications[1:]
			return nt, false
		}

		notification, stopped := blockingGet()
		if stopped {
			return
		}

		select {
		case <-stopCh:
			return
		case p.nextCh <- notification:
		}
	}
}

func (p *processorListener) run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	for {
		var next interface{}
		select {
		case <-stopCh:
			func() {
				p.lock.Lock()
				defer p.lock.Unlock()
				p.cond.Broadcast()
			}()
			return
		case next = <-p.nextCh:
		}

		switch notification := next.(type) {
		case updateNotification:
			p.handler.OnUpdate(notification.oldObj, notification.newObj)
		case addNotification:
			p.handler.OnAdd(notification.newObj)
		case deleteNotification:
			p.handler.OnDelete(notification.oldObj)
		default:
			utilruntime.HandleError(fmt.Errorf("unrecognized notification: %#v", next))
		}
	}
}

// shouldResync deterimines if the listener needs a resync. If the listener's resyncPeriod is 0,
// this always returns false.
func (p *processorListener) shouldResync(now time.Time) bool {
	if p.resyncPeriod == 0 {
		return false
	}

	return now.After(p.nextResync) || now.Equal(p.nextResync)
}

// shouldHandleSyncItem returns true if the reflector has started resyncing and the listener is
// participating in the current resync.
func (p *processorListener) shouldHandleSyncItem() bool {
	handle := p.syncing && p.wantSyncItems
	if handle && !p.hasSynced {
		p.hasSynced = true
	}
	return handle
}

// prepareForResync sets the listener as participating in the current resync.
func (p *processorListener) prepareForResync(now time.Time) {
	p.wantSyncItems = true
}

// syncRunning is called at the beginning and end of a sync. If it is the end of a sync and the
// listener is participating in the current resync, the listener's nextResync time is recalculated.
func (p *processorListener) syncRunning(running bool) {
	p.syncing = running
	if !running && p.wantSyncItems && p.hasSynced {
		if p.resyncPeriod > 0 {
			now := p.clock.Now()
			p.nextResync = now.Add(p.resyncPeriod)
		}
		p.wantSyncItems = false
	}
}
