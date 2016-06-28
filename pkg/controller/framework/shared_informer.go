/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package framework

import (
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
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
	// The handler can be blocking. SharedInformer is responsible for queuing events underlying for each handler.
	AddEventHandler(handler ResourceEventHandler) error
	GetStore() cache.Store
	// GetController gives back a synthetic interface that "votes" to start the informer
	GetController() ControllerInterface
	Run(stopCh <-chan struct{})
	HasSynced() bool
}

type SharedIndexInformer interface {
	SharedInformer
	// AddIndexers add indexers to the informer before it starts.
	AddIndexers(indexers cache.Indexers) error
	GetIndexer() cache.Indexer
}

// NewSharedInformer creates a new instance for the listwatcher.
// TODO: create a cache/factory of these at a higher level for the list all, watch all of a given resource that can
// be shared amongst all consumers.
func NewSharedInformer(lw cache.ListerWatcher, objType runtime.Object, resyncPeriod time.Duration) SharedInformer {
	return NewSharedIndexInformer(lw, objType, resyncPeriod, cache.Indexers{})
}

// NewSharedIndexInformer creates a new instance for the listwatcher.
// TODO: create a cache/factory of these at a higher level for the list all, watch all of a given resource that can
// be shared amongst all consumers.
func NewSharedIndexInformer(lw cache.ListerWatcher, objType runtime.Object, resyncPeriod time.Duration, indexers cache.Indexers) SharedIndexInformer {
	return &sharedInformer{
		coreInformer: newCoreInformer(lw, objType, resyncPeriod, indexers),
	}
}

type sharedInformer struct {
	*coreInformer
}

func (s *sharedInformer) Run(stopCh <-chan struct{}) {
	s.mu.Lock()
	for _, h := range s.handlers {
		qh := h.(*queuedEventHandler)
		go qh.run(stopCh)
	}
	s.mu.Unlock()

	s.coreInformer.Run(stopCh)
}

func (s *sharedInformer) AddEventHandler(handler ResourceEventHandler) error {
	return s.addEventHandler(newQueuedEventHandler(handler))
}

func (s *sharedInformer) GetStore() cache.Store {
	return s.GetIndexer()
}

func (s *sharedInformer) GetController() ControllerInterface {
	return &dummyController{informer: s}
}

// dummyController hides the fact that a SharedInformer is different from a dedicated one
// where a caller can `Run`.  The run method is disonnected in this case, because higher
// level logic will decide when to start the SharedInformer and related controller.
// Because returning information back is always asynchronous, the legacy callers shouldn't
// notice any change in behavior.
type dummyController struct {
	informer *sharedInformer
}

func (v *dummyController) Run(stopCh <-chan struct{}) {
}

func (v *dummyController) HasSynced() bool {
	return v.informer.HasSynced()
}
