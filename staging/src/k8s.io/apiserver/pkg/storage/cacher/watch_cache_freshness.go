/*
Copyright The Kubernetes Authors.

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
	"container/heap"
	"context"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/progress"
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

func newFreshnessWaiter(c clock.Clock, groupResource schema.GroupResource, pr *progress.ConditionalProgressRequester) *freshnessWaiter {
	w := &freshnessWaiter{
		clock:             c,
		groupResource:     groupResource,
		progressRequester: pr,
	}
	w.cond = sync.NewCond(&w.lock)
	return w
}

type freshnessWaiter struct {
	clock             clock.Clock
	groupResource     schema.GroupResource
	progressRequester *progress.ConditionalProgressRequester

	lock       sync.Mutex
	cond       *sync.Cond
	latestRV   uint64
	waiterHeap rvWaiterHeap
	stopped    bool
}

type rvWaiter struct {
	targetRV    uint64
	satisfiedCh chan struct{}
	heapIndex   int
}

func (w *freshnessWaiter) broadcast(resourceVersion uint64) {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.latestRV = resourceVersion
	if w.waiterReadyLocked() {
		w.cond.Signal()
	}
}

func (w *freshnessWaiter) waitUntilFreshLockedWithLocker(ctx context.Context, resourceVersion uint64, locker sync.Locker) error {
	if resourceVersion == 0 || resourceVersion <= w.latestRV {
		return nil
	}
	w.progressRequester.Add()
	defer w.progressRequester.Remove()
	startTime := w.clock.Now()
	defer func() {
		if resourceVersion > 0 {
			metrics.WatchCacheReadWait.WithContext(ctx).WithLabelValues(w.groupResource.Group, w.groupResource.Resource).Observe(w.clock.Since(startTime).Seconds())
		}
	}()

	timer := w.clock.NewTimer(blockTimeout)
	defer timer.Stop()

	waiter := w.registerWaiter(resourceVersion)
	locker.Unlock()
	select {
	case <-waiter.satisfiedCh:
		locker.Lock()
		return nil
	case <-timer.C():
		w.unregisterWaiter(waiter)
		locker.Lock()
		if w.latestRV >= resourceVersion {
			return nil
		}
		return storage.NewTooLargeResourceVersionError(resourceVersion, w.latestRV, resourceVersionTooHighRetrySeconds)
	}
}

func (w *freshnessWaiter) registerWaiter(targetRV uint64) *rvWaiter {
	waiter := &rvWaiter{
		targetRV:    targetRV,
		satisfiedCh: make(chan struct{}),
	}
	w.lock.Lock()
	defer w.lock.Unlock()
	heap.Push(&w.waiterHeap, waiter)
	if w.waiterHeap[0].targetRV <= w.latestRV {
		w.cond.Signal()
	}
	return waiter
}

func (w *freshnessWaiter) unregisterWaiter(waiter *rvWaiter) {
	w.lock.Lock()
	defer w.lock.Unlock()
	if waiter.heapIndex >= 0 {
		heap.Remove(&w.waiterHeap, waiter.heapIndex)
	}
}

func (w *freshnessWaiter) Run(stopCh <-chan struct{}) {
	ctx := wait.ContextForChannel(stopCh)
	go func() {
		defer utilruntime.HandleCrashWithContext(ctx)
		<-stopCh
		w.lock.Lock()
		defer w.lock.Unlock()
		w.stopped = true
		w.cond.Signal()
	}()

	w.lock.Lock()
	defer w.lock.Unlock()
	for !w.stopped {
		for w.waiterReadyLocked() {
			waiter := heap.Pop(&w.waiterHeap).(*rvWaiter)
			close(waiter.satisfiedCh)
		}
		w.cond.Wait()
	}
}

func (w *freshnessWaiter) waiterReadyLocked() bool {
	return len(w.waiterHeap) > 0 && w.waiterHeap[0].targetRV <= w.latestRV
}

type rvWaiterHeap []*rvWaiter

func (h rvWaiterHeap) Len() int { return len(h) }

func (h rvWaiterHeap) Less(i, j int) bool { return h[i].targetRV < h[j].targetRV }

func (h rvWaiterHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].heapIndex = i
	h[j].heapIndex = j
}

func (h *rvWaiterHeap) Push(x any) {
	waiter := x.(*rvWaiter)
	waiter.heapIndex = len(*h)
	*h = append(*h, waiter)
}

func (h *rvWaiterHeap) Pop() any {
	old := *h
	n := len(old)
	waiter := old[n-1]
	old[n-1] = nil
	waiter.heapIndex = -1
	*h = old[:n-1]
	return waiter
}
