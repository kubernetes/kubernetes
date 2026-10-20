/*
Copyright 2026 The Kubernetes Authors.

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

package dynamicresources

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// failingSliceLister fails the next N list calls so the real worker sees
// listError results without touching publication, which uses its own lister.
type failingSliceLister struct {
	*mutableSliceLister
	failures atomic.Int32
	calls    atomic.Int32
}

func (l *failingSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	l.calls.Add(1)
	if l.failures.Add(-1) >= 0 {
		return nil, errors.New("injected list failure")
	}
	l.failures.Store(0)
	return l.mutableSliceLister.ListPatchedResourceSlices()
}

func (l *failingSliceLister) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return l.ListPatchedResourceSlices()
}

type workerPass struct {
	pool      poolName
	requeues  int
	pending   bool
	reason    string
	queueSize int
}

// List errors requeue with rate-limited backoff and
// then settle; incomplete pools park without requeue until one publication
// pass; cancellation stops the worker.
func TestSharingAffinityReconciliation_WorkerRetryAndCancellation(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	c := newSharingAffinityAccountingTracker(t, true)
	a := c.allocatedDevices
	base := c.affinity.slices.(accountingSliceLister).slices
	lister := &mutableSliceLister{slices: base}
	extract := &failingSliceLister{mutableSliceLister: lister}
	c.affinity.slices, c.metadataSlices = extract, lister
	c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
	a.metadata = c.metadata
	_, tCtx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()

	key := poolName{accountingDriver, accountingPool}
	owner := sharingAffinityAccountingClaim("owner", "blue")
	device := sharingAffinityAccountingDevice(owner)
	ownerKey := ownerPoolKey{uid: owner.UID, pool: key}

	var mu sync.Mutex
	var passes []workerPass
	passed := make(chan struct{}, 100)
	c.reconcileAfterCommitHook = func(pool poolName) {
		a.mutex.RLock()
		p := workerPass{pool: pool, requeues: a.reconcile.NumRequeues(pool)}
		if rec := a.owners[ownerKey]; rec != nil {
			p.pending, p.reason = rec.pending, rec.reason
		}
		a.mutex.RUnlock()
		mu.Lock()
		passes = append(passes, p)
		mu.Unlock()
		passed <- struct{}{}
	}
	snapshot := func() []workerPass {
		mu.Lock()
		defer mu.Unlock()
		return append([]workerPass(nil), passes...)
	}
	drain := func() {
		for {
			select {
			case <-passed:
			default:
				return
			}
		}
	}

	c.initReconciler(ctx)
	c.cache.AddEventHandler(a.handlers())
	a.claimsSynced = true
	c.publishMetadata(true, nil)
	assumecache.AddTestObject(c.cache, owner)
	reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

	t.Run("list-error-retries-then-settles", func(t *testing.T) {
		drain()
		mu.Lock()
		passes = nil
		mu.Unlock()
		extract.failures.Store(2)
		updated := reconcileSlices(base, 2)
		lister.set(updated...)
		c.publishMetadata(false, affectedPools(updated[0]))
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		got := snapshot()
		var retries []int
		settled := false
		for _, p := range got {
			if p.pool != key {
				continue
			}
			if p.reason == "listError" {
				if !p.pending {
					t.Errorf("listError pass left owner settled: %+v", p)
				}
				retries = append(retries, p.requeues)
			} else if !p.pending {
				settled = true
				if p.requeues != 0 {
					t.Errorf("settled pass kept requeue count %d, want Forget", p.requeues)
				}
			}
		}
		if len(retries) != 2 || retries[0] != 1 || retries[1] != 2 {
			t.Errorf("listError passes NumRequeues=%v, want [1 2]", retries)
		}
		if !settled {
			t.Errorf("no settled pass after list recovered: %+v", got)
		}
	})

	t.Run("incomplete-pool-parks-without-busy-loop", func(t *testing.T) {
		drain()
		mu.Lock()
		passes = nil
		mu.Unlock()
		// Publish generation 3 for only one of the pool's slices: the pool
		// is incomplete, so the owner must wait for an event, not spin.
		incomplete := reconcileSlices(base, 3)
		if len(incomplete) < 2 {
			t.Skip("fixture pool has a single slice")
		}
		lister.set(append(reconcileSlices(base, 2)[1:], incomplete[0])...)
		callsBefore := extract.calls.Load()
		c.publishMetadata(false, affectedPools(incomplete[0]))
		// Exactly one initial pass parks the owner; nothing follows it.
		select {
		case <-passed:
		case <-time.After(5 * time.Second):
			t.Fatal("no pass for incomplete publication")
		}
		time.Sleep(300 * time.Millisecond)
		got := snapshot()
		if len(got) != 1 {
			t.Fatalf("incomplete pool processed %d times over 300ms, want exactly 1: %+v", len(got), got)
		}
		if p := got[0]; p.pool != key || !p.pending || p.requeues != 0 {
			t.Errorf("parked pass = %+v, want pending owner of %v with NumRequeues=0", p, key)
		}
		if calls := extract.calls.Load() - callsBefore; calls > 1 {
			t.Errorf("incomplete pool listed slices %d times, want at most 1", calls)
		}
		a.mutex.RLock()
		rec := a.owners[ownerKey]
		parked := rec != nil && rec.pending
		requeues, qlen := a.reconcile.NumRequeues(key), a.reconcile.Len()
		a.mutex.RUnlock()
		if !parked {
			t.Errorf("owner not pending while pool incomplete: %+v", rec)
		}
		if requeues != 0 || qlen != 0 {
			t.Errorf("parked pool NumRequeues=%d Len=%d, want 0/0", requeues, qlen)
		}
		// One completing publication resolves the wait with exactly one pass.
		lister.set(reconcileSlices(base, 3)...)
		c.publishMetadata(false, affectedPools(incomplete[0]))
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		time.Sleep(300 * time.Millisecond)
		got = snapshot()[1:]
		if len(got) != 1 {
			t.Fatalf("completing publication caused %d passes, want exactly 1: %+v", len(got), got)
		}
		if p := got[0]; p.pending || p.requeues != 0 {
			t.Errorf("settlement pass = %+v, want settled with NumRequeues=0", p)
		}
	})

	t.Run("enqueue-during-compute-runs-second-pass", func(t *testing.T) {
		drain()
		entered, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		hook := func(pool poolName) {
			once.Do(func() {
				close(entered)
				<-release
			})
		}
		c.reconcileProcessHook = hook
		defer func() { c.reconcileProcessHook = nil }()
		a.mutex.Lock()
		a.enqueuePoolLocked(key)
		a.mutex.Unlock()
		reconcileAwait(t, entered, "worker processing")
		// The item is being processed; a second enqueue must be retained
		// by the workqueue's dirty set and produce another pass.
		a.mutex.Lock()
		a.enqueuePoolLocked(key)
		a.mutex.Unlock()
		close(release)
		for i := range 2 {
			select {
			case <-passed:
			case <-time.After(5 * time.Second):
				t.Fatalf("pass %d did not run after enqueue during compute", i+1)
			}
		}
	})

	t.Run("cancellation-stops-worker", func(t *testing.T) {
		cancel()
		reconcileAwait(t, c.reconcilerDone, "reconciler shutdown")
		if !a.reconcile.ShuttingDown() {
			t.Error("reconcile queue not shut down after cancellation")
		}
	})
}
