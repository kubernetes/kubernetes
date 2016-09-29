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

	"k8s.io/client-go/1.5/pkg/runtime"
	utilruntime "k8s.io/client-go/1.5/pkg/util/runtime"
	"k8s.io/client-go/1.5/pkg/util/wait"

	"github.com/golang/glog"
)

// if you use this, there is one behavior change compared to a standard Informer.
// When you receive a notification, the cache will be AT LEAST as fresh as the
// notification, but it MAY be more fresh.  You should NOT depend on the contents
// of the cache exactly matching the notification you've received in handler
// functions.  If there was a create, followed by a delete, the cache may NOT
// have your item.  This has advantages over the broadcaster since it allows us
// to share a common cache across many controllers. Extending the broadcaster
// would have required us keep duplicate caches for each watch.
type SharedInformer interface {
	// events to a single handler are delivered sequentially, but there is no coordination between different handlers
	// You may NOT add a handler *after* the SharedInformer is running.  That will result in an error being returned.
	// TODO we should try to remove this restriction eventually.
	AddEventHandler(handler ResourceEventHandler) error
	GetStore() Store
	// GetController gives back a synthetic interface that "votes" to start the informer
	GetController() ControllerInterface
	Run(stopCh <-chan struct{})
	HasSynced() bool
	LastSyncResourceVersion() string
}

type SharedIndexInformer interface {
	SharedInformer
	// AddIndexers add indexers to the informer before it starts.
	AddIndexers(indexers Indexers) error
	GetIndexer() Indexer
}

// NewSharedInformer creates a new instance for the listwatcher.
// TODO: create a cache/factory of these at a higher level for the list all, watch all of a given resource that can
// be shared amongst all consumers.
func NewSharedInformer(lw ListerWatcher, objType runtime.Object, resyncPeriod time.Duration) SharedInformer {
	return NewSharedIndexInformer(lw, objType, resyncPeriod, Indexers{})
}

// NewSharedIndexInformer creates a new instance for the listwatcher.
// TODO: create a cache/factory of these at a higher level for the list all, watch all of a given resource that can
// be shared amongst all consumers.
func NewSharedIndexInformer(lw ListerWatcher, objType runtime.Object, resyncPeriod time.Duration, indexers Indexers) SharedIndexInformer {
	sharedIndexInformer := &sharedIndexInformer{
		processor:        &sharedProcessor{},
		indexer:          NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers),
		listerWatcher:    lw,
		objectType:       objType,
		fullResyncPeriod: resyncPeriod,
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
	controller *Controller

	processor *sharedProcessor

	// This block is tracked to handle late initialization of the controller
	listerWatcher    ListerWatcher
	objectType       runtime.Object
	fullResyncPeriod time.Duration

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

	cfg := &Config{
		Queue:            fifo,
		ListerWatcher:    s.listerWatcher,
		ObjectType:       s.objectType,
		FullResyncPeriod: s.fullResyncPeriod,
		RetryOnError:     false,

		Process: s.HandleDeltas,
	}

	func() {
		s.startedLock.Lock()
		defer s.startedLock.Unlock()

		s.controller = New(cfg)
		s.started = true
	}()

	s.stopCh = stopCh
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
	return s.controller.reflector.LastSyncResourceVersion()
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

func (s *sharedIndexInformer) GetController() ControllerInterface {
	return &dummyController{informer: s}
}

func (s *sharedIndexInformer) AddEventHandler(handler ResourceEventHandler) error {
	s.startedLock.Lock()
	defer s.startedLock.Unlock()

	if !s.started {
		listener := newProcessListener(handler)
		s.processor.listeners = append(s.processor.listeners, listener)
		return nil
	}

	// in order to safely join, we have to
	// 1. stop sending add/update/delete notifications
	// 2. do a list against the store
	// 3. send synthetic "Add" events to the new handler
	// 4. unblock
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	listener := newProcessListener(handler)
	s.processor.listeners = append(s.processor.listeners, listener)

	go listener.run(s.stopCh)
	go listener.pop(s.stopCh)

	items := s.indexer.List()
	for i := range items {
		listener.add(addNotification{newObj: items[i]})
	}

	return nil
}

func (s *sharedIndexInformer) HandleDeltas(obj interface{}) error {
	s.blockDeltas.Lock()
	defer s.blockDeltas.Unlock()

	// from oldest to newest
	for _, d := range obj.(Deltas) {
		switch d.Type {
		case Sync, Added, Updated:
			if old, exists, err := s.indexer.Get(d.Object); err == nil && exists {
				if err := s.indexer.Update(d.Object); err != nil {
					return err
				}
				s.processor.distribute(updateNotification{oldObj: old, newObj: d.Object})
			} else {
				if err := s.indexer.Add(d.Object); err != nil {
					return err
				}
				s.processor.distribute(addNotification{newObj: d.Object})
			}
		case Deleted:
			if err := s.indexer.Delete(d.Object); err != nil {
				return err
			}
			s.processor.distribute(deleteNotification{oldObj: d.Object})
		}
	}
	return nil
}

type sharedProcessor struct {
	listeners []*processorListener
}

func (p *sharedProcessor) distribute(obj interface{}) {
	for _, listener := range p.listeners {
		listener.add(obj)
	}
}

func (p *sharedProcessor) run(stopCh <-chan struct{}) {
	for _, listener := range p.listeners {
		go listener.run(stopCh)
		go listener.pop(stopCh)
	}
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
}

func newProcessListener(handler ResourceEventHandler) *processorListener {
	ret := &processorListener{
		pendingNotifications: []interface{}{},
		nextCh:               make(chan interface{}),
		handler:              handler,
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
