/*
Copyright 2017 The Kubernetes Authors.

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

package watch

import (
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
)

func newTicketer() *ticketer {
	return &ticketer{
		cond: sync.NewCond(&sync.Mutex{}),
	}
}

type ticketer struct {
	counter uint64

	cond    *sync.Cond
	current uint64
}

func (t *ticketer) GetTicket() uint64 {
	// -1 to start from 0
	return atomic.AddUint64(&t.counter, 1) - 1
}

func (t *ticketer) WaitForTicket(ticket uint64, f func()) {
	t.cond.L.Lock()
	defer t.cond.L.Unlock()
	for ticket != t.current {
		t.cond.Wait()
	}

	f()

	t.current++
	t.cond.Broadcast()
}

// NewIndexerInformerWatcher will create an IndexerInformer and wrap it into watch.Interface
// so you can use it anywhere where you'd have used a regular Watcher returned from Watch method.
// it also returns a channel you can use to wait for the informers to fully shutdown.
func NewIndexerInformerWatcher(lw cache.ListerWatcher, objType runtime.Object) (cache.Indexer, cache.Controller, watch.Interface, <-chan struct{}) {
	ch := make(chan watch.Event)
	doneCh := make(chan struct{})
	w := watch.NewProxyWatcher(ch)
	t := newTicketer()

	indexer, informer := cache.NewIndexerInformer(lw, objType, 0, cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			go t.WaitForTicket(t.GetTicket(), func() {
				select {
				case ch <- watch.Event{
					Type:   watch.Added,
					Object: obj.(runtime.Object),
				}:
				case <-w.StopChan():
				}
			})
		},
		UpdateFunc: func(old, new interface{}) {
			go t.WaitForTicket(t.GetTicket(), func() {
				select {
				case ch <- watch.Event{
					Type:   watch.Modified,
					Object: new.(runtime.Object),
				}:
				case <-w.StopChan():
				}
			})
		},
		DeleteFunc: func(obj interface{}) {
			go t.WaitForTicket(t.GetTicket(), func() {
				staleObj, stale := obj.(cache.DeletedFinalStateUnknown)
				if stale {
					// We have no means of passing the additional information down using watch API based on watch.Event
					// but the caller can filter such objects by checking if metadata.deletionTimestamp is set
					obj = staleObj
				}

				select {
				case ch <- watch.Event{
					Type:   watch.Deleted,
					Object: obj.(runtime.Object),
				}:
				case <-w.StopChan():
				}
			})
		},
	}, cache.Indexers{})

	go func() {
		defer close(doneCh)
		informer.Run(w.StopChan())
	}()

	return indexer, informer, w, doneCh
}
