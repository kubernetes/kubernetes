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

package dynamicresources

import (
	"fmt"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
)

func assertAffinityWaiterIndexes(t *testing.T, w *affinityWaiters, want map[poolName]map[types.UID]*v1.Pod, reverse map[types.UID]sets.Set[poolName]) {
	t.Helper()
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.byPool) != len(want) {
		t.Errorf("pool index size = %d, want %d", len(w.byPool), len(want))
	}
	for pool, pods := range want {
		got, found := w.byPool[pool]
		if !found || len(got) != len(pods) {
			t.Errorf("pool %v entry count = %d (present=%t), want %d", pool, len(got), found, len(pods))
		}
		for uid, pod := range pods {
			if got[uid] != pod {
				t.Errorf("pool %v pod %s pointer = %p, want %p", pool, uid, got[uid], pod)
			}
		}
	}
	if len(w.byPod) != len(reverse) {
		t.Errorf("reverse index size = %d, want %d: %+v", len(w.byPod), len(reverse), w.byPod)
	}
	for uid, pools := range reverse {
		if got, found := w.byPod[uid]; !found || !got.Equal(pools) {
			t.Errorf("pod %s pool set = %v (present=%t), want %v", uid, got, found, pools)
		}
	}
}

func assertAffinityResolvedPods(t *testing.T, got []*v1.Pod, want ...*v1.Pod) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("resolved Pod count = %d, want %d", len(got), len(want))
	}
	expected := sets.New(want...)
	seen := sets.New[*v1.Pod]()
	for _, pod := range got {
		if !expected.Has(pod) {
			t.Errorf("unexpected resolved Pod pointer %p", pod)
		}
		if seen.Has(pod) {
			t.Errorf("Pod %p returned more than once", pod)
		}
		seen.Insert(pod)
	}
	for _, pod := range want {
		if !seen.Has(pod) {
			t.Errorf("missing resolved Pod %s/%s (%s)", pod.Namespace, pod.Name, pod.UID)
		}
	}
}

type affinityWaiterPodLister struct {
	corelisters.PodLister
	beforeGet func()
}

func (l affinityWaiterPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return affinityWaiterPodNamespaceLister{PodNamespaceLister: l.PodLister.Pods(namespace), beforeGet: l.beforeGet}
}

type affinityWaiterPodNamespaceLister struct {
	corelisters.PodNamespaceLister
	beforeGet func()
}

func (l affinityWaiterPodNamespaceLister) Get(name string) (*v1.Pod, error) {
	l.beforeGet()
	return l.PodNamespaceLister.Get(name)
}

// These tests cover the registry, not informer callbacks or scheduler queue wiring.
func TestSharingAffinityReconciliation_Waiters(t *testing.T) {
	p := poolName{driver: "driver.example", pool: "p"}
	q := poolName{driver: "driver.example", pool: "q"}
	r := poolName{driver: "other.example", pool: "p"}
	a := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "a", UID: "a"}}
	b := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "b", UID: "b"}}

	t.Run("register-initializes-both-indexes-and-duplicate-is-idempotent", func(t *testing.T) {
		var w affinityWaiters
		for i := 0; i < 2; i++ {
			if !w.register(a, sets.New(p, q)) {
				t.Errorf("registration %d rejected by open registry", i)
			}
			assertAffinityWaiterIndexes(t, &w,
				map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, q: {a.UID: a}},
				map[types.UID]sets.Set[poolName]{a.UID: sets.New(p, q)})
		}
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q), func(poolName) bool { return true }), a)
		assertAffinityWaiterIndexes(t, &w, nil, nil)
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q), func(poolName) bool { return true }))
	})

	t.Run("remove-cleans-all-indexes-and-preserves-other-pods", func(t *testing.T) {
		w := affinityWaiters{
			byPool: map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, q: {a.UID: a, b.UID: b}},
			byPod:  map[types.UID]sets.Set[poolName]{a.UID: sets.New(p, q), b.UID: sets.New(q)},
		}
		w.remove(a.UID)
		assertAffinityWaiterIndexes(t, &w,
			map[poolName]map[types.UID]*v1.Pod{q: {b.UID: b}},
			map[types.UID]sets.Set[poolName]{b.UID: sets.New(q)})
		w.remove(a.UID)
		w.remove("unknown")
		assertAffinityWaiterIndexes(t, &w,
			map[poolName]map[types.UID]*v1.Pod{q: {b.UID: b}},
			map[types.UID]sets.Set[poolName]{b.UID: sets.New(q)})
		w.remove(b.UID)
		assertAffinityWaiterIndexes(t, &w, nil, nil)
	})

	t.Run("any-resolved-pool-removes-pod-from-every-index", func(t *testing.T) {
		w := affinityWaiters{
			byPool: map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, q: {a.UID: a, b.UID: b}, r: {a.UID: a}},
			byPod:  map[types.UID]sets.Set[poolName]{a.UID: sets.New(p, q, r), b.UID: sets.New(q)},
		}
		got := w.takeResolved(sets.New(p, q), func(pool poolName) bool { return pool == p })
		assertAffinityResolvedPods(t, got, a)
		assertAffinityWaiterIndexes(t, &w,
			map[poolName]map[types.UID]*v1.Pod{q: {b.UID: b}},
			map[types.UID]sets.Set[poolName]{b.UID: sets.New(q)})
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, r), func(poolName) bool { return true }))
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(q), func(poolName) bool { return true }), b)
		assertAffinityWaiterIndexes(t, &w, nil, nil)
	})

	t.Run("only-changed-pools-can-resolve-waiters", func(t *testing.T) {
		w := affinityWaiters{
			byPool: map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, r: {b.UID: b}},
			byPod:  map[types.UID]sets.Set[poolName]{a.UID: sets.New(p), b.UID: sets.New(r)},
		}
		assertAffinityResolvedPods(t, w.takeResolved(sets.New[poolName](), func(pool poolName) bool {
			t.Errorf("evaluated %v without a changed pool", pool)
			return true
		}))
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p), func(pool poolName) bool {
			if pool != p {
				t.Errorf("evaluated unchanged pool %v", pool)
			}
			return true
		}), a)
		assertAffinityWaiterIndexes(t, &w,
			map[poolName]map[types.UID]*v1.Pod{r: {b.UID: b}},
			map[types.UID]sets.Set[poolName]{b.UID: sets.New(r)})
	})

	t.Run("notification-alone-does-not-resolve-lister-ahead-analogue", func(t *testing.T) {
		w := affinityWaiters{
			byPool: map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}},
			byPod:  map[types.UID]sets.Set[poolName]{a.UID: sets.New(p)},
		}
		// The predicate models held publication; no real lister is exercised here.
		publicationMatchesLister := false
		calls := 0
		resolved := func(pool poolName) bool {
			calls++
			if pool != p {
				t.Errorf("resolved predicate pool = %v, want %v", pool, p)
			}
			return publicationMatchesLister
		}
		for i := 0; i < 3; i++ {
			before := calls
			assertAffinityResolvedPods(t, w.takeResolved(sets.New(p), resolved))
			if calls == before {
				t.Error("changed registered pool was not checked for resolution")
			}
			assertAffinityWaiterIndexes(t, &w,
				map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}},
				map[types.UID]sets.Set[poolName]{a.UID: sets.New(p)})
		}
		publicationMatchesLister = true
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p), resolved), a)
		assertAffinityWaiterIndexes(t, &w, nil, nil)
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p), resolved))
	})

	t.Run("close-clears-indexes-and-rejects-future-registers", func(t *testing.T) {
		w := affinityWaiters{
			byPool: map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, q: {a.UID: a}},
			byPod:  map[types.UID]sets.Set[poolName]{a.UID: sets.New(p, q)},
		}
		w.close()
		if !w.closed {
			t.Error("close did not mark registry closed")
		}
		assertAffinityWaiterIndexes(t, &w, nil, nil)
		if w.register(b, sets.New(q)) {
			t.Error("closed registry accepted registration")
		}
		w.remove(a.UID)
		w.close()
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q), func(pool poolName) bool {
			t.Errorf("closed registry evaluated pool %v", pool)
			return true
		}))
		assertAffinityWaiterIndexes(t, &w, nil, nil)
	})

	t.Run("close-racing-with-register-leaves-no-retained-pods", func(t *testing.T) {
		var w affinityWaiters
		start := make(chan struct{})
		var workers sync.WaitGroup
		for i := 0; i < 32; i++ {
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pod-%d", i), UID: types.UID(fmt.Sprintf("uid-%d", i))}}
			workers.Add(1)
			go func() {
				defer workers.Done()
				<-start
				w.register(pod, sets.New(p, q))
			}()
		}
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			w.close()
		}()
		close(start)
		workers.Wait()
		if !w.closed {
			t.Error("racing close did not mark registry closed")
		}
		assertAffinityWaiterIndexes(t, &w, nil, nil)
		if w.register(a, sets.New(p, q)) {
			t.Error("registration after racing close succeeded")
		}
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q), func(poolName) bool { return true }))
		assertAffinityWaiterIndexes(t, &w, nil, nil)
	})

	t.Run("one-hundred-registration-deletion-cycles-stay-bounded", func(t *testing.T) {
		var w affinityWaiters
		for i := 0; i < 100; i++ {
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "recreated", UID: types.UID(fmt.Sprintf("uid-%d", i))}}
			if !w.register(pod, sets.New(p, q)) {
				t.Errorf("cycle %d registration rejected", i)
			}
			assertAffinityWaiterIndexes(t, &w,
				map[poolName]map[types.UID]*v1.Pod{p: {pod.UID: pod}, q: {pod.UID: pod}},
				map[types.UID]sets.Set[poolName]{pod.UID: sets.New(p, q)})
			w.remove(pod.UID)
			assertAffinityWaiterIndexes(t, &w, nil, nil)
			assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q), func(poolName) bool { return true }))
		}
	})

	t.Run("registration-validates-cached-pod-lifecycle", func(t *testing.T) {
		for _, lifecycle := range []string{"deleted", "recreated", "bound", "terminating", "closed"} {
			t.Run(lifecycle, func(t *testing.T) {
				logger, _ := ktesting.NewTestContext(t)
				indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
				cached := a.DeepCopy()
				switch lifecycle {
				case "recreated":
					cached.UID = "replacement"
				case "bound":
					cached.Spec.NodeName = "node"
				case "terminating":
					now := metav1.Now()
					cached.DeletionTimestamp = &now
				}
				if lifecycle != "deleted" {
					if err := indexer.Add(cached); err != nil {
						t.Fatal(err)
					}
				}
				w := &affinityWaiters{podLister: corelisters.NewPodLister(indexer)}
				if lifecycle == "closed" {
					w.close()
				}
				pl := &DynamicResources{affinityWaiters: w}
				// No tracker or activation handle: stale registrations must
				// be removed before checking readiness or activating.
				pl.registerAffinityWaiters(logger, a, sets.New(p, q))
				assertAffinityWaiterIndexes(t, w, nil, nil)
			})
		}
	})

	t.Run("delete-between-insertion-and-lifecycle-validation", func(t *testing.T) {
		logger, _ := ktesting.NewTestContext(t)
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
		if err := indexer.Add(a); err != nil {
			t.Fatal(err)
		}
		w := &affinityWaiters{}
		gets := 0
		w.podLister = affinityWaiterPodLister{
			PodLister: corelisters.NewPodLister(indexer),
			beforeGet: func() {
				gets++
				assertAffinityWaiterIndexes(t, w,
					map[poolName]map[types.UID]*v1.Pod{p: {a.UID: a}, q: {a.UID: a}, {}: {a.UID: a}},
					map[types.UID]sets.Set[poolName]{a.UID: sets.New(p, q, poolName{})})
				// Informer stores advance before lifecycle callbacks. Hold the
				// callback until validation has observed the missing Pod.
				if err := indexer.Delete(a); err != nil {
					t.Fatal(err)
				}
			},
		}
		pl := &DynamicResources{affinityWaiters: w}
		// A nil tracker/handle also ensures the deleted Pod never reaches
		// readiness evaluation or activation.
		pl.registerAffinityWaiters(logger, a, sets.New(p, q, poolName{}))
		if gets != 1 {
			t.Fatalf("Pod lifecycle reads = %d, want exactly one", gets)
		}
		assertAffinityWaiterIndexes(t, w, nil, nil)
		w.remove(a.UID)
		assertAffinityWaiterIndexes(t, w, nil, nil)
		assertAffinityResolvedPods(t, w.takeResolved(sets.New(p, q, poolName{}), func(poolName) bool { return true }))
	})
}
