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
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

type affinityActivationHandle struct {
	fwk.Handle
	factory informers.SharedInformerFactory
	mu      sync.Mutex
	calls   []map[string]*v1.Pod
}

func (h *affinityActivationHandle) SharedInformerFactory() informers.SharedInformerFactory {
	return h.factory
}

func (h *affinityActivationHandle) Activate(_ klog.Logger, pods map[string]*v1.Pod) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.calls = append(h.calls, maps.Clone(pods))
}

func (h *affinityActivationHandle) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.calls)
}

func affinityPluginWaiter(t *testing.T, ctx context.Context, tc *testContext, pod *v1.Pod) (*DynamicResources, *affinityActivationHandle, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	handle := &affinityActivationHandle{factory: tc.informerFactory}
	pl := &DynamicResources{draManager: tc.draManager, fh: handle}
	if err := pl.initAffinityWaiters(ctx); err != nil {
		t.Fatalf("initialize profile waiters: %v", err)
	}
	if err := tc.informerFactory.Core().V1().Pods().Informer().GetIndexer().Add(pod); err != nil {
		t.Fatalf("cache waiting Pod: %v", err)
	}
	return pl, handle, cancel
}

func affinityWaitForActivation(t *testing.T, ctx context.Context, handle *affinityActivationHandle, pod *v1.Pod) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return handle.count() != 0, nil
	}); err != nil {
		t.Fatalf("waiting for Pod activation: %v", err)
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if len(handle.calls) != 1 || len(handle.calls[0]) != 1 || handle.calls[0][pod.Namespace+"/"+pod.Name] != pod {
		t.Fatalf("activation calls = %#v, want exactly one activation of %s/%s", handle.calls, pod.Namespace, pod.Name)
	}
}

func TestSharingAffinityReconciliation_PluginWaiters(t *testing.T) {
	p := poolName{accountingDriver, accountingPool}
	q := poolName{accountingDriver, "incomplete"}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "waiting", UID: "waiting"}}

	t.Run("progress-before-registration-activates-on-any-resolved-pool", func(t *testing.T) {
		slices := poolEventsSlices()
		incomplete := slices[0].DeepCopy()
		incomplete.Name, incomplete.Spec.Pool.Name = "incomplete-metadata", q.pool
		owner := sharingAffinityAccountingClaim("incomplete-owner", "blue")
		owner.Status.Allocation.Devices.Results[0].Pool = q.pool
		ctx, tc := reconcileSetup(t, append(slices, incomplete), owner)
		pl, handle, _ := affinityPluginWaiter(t, ctx, tc, pod)
		if resolved, err := pl.affinityTracker.exclusionResolved(q); err != nil || resolved {
			t.Fatalf("incomplete sibling resolved=%t, err=%v", resolved, err)
		}
		// The readiness change already happened. Registration must recheck it
		// rather than waiting for a notification that will never recur.
		claim, _, _ := filterAffinityFixture()
		cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, nil)
		state.affinityExcluded = sets.New(p, q)
		if _, status := pl.PostFilter(ctx, cycle, pod, nil); status.Code() != fwk.Unschedulable {
			t.Fatalf("PostFilter = %v, want Unschedulable", status)
		}
		affinityWaitForActivation(t, ctx, handle, pod)
		assertAffinityWaiterIndexes(t, pl.affinityWaiters, nil, nil)
		pl.activateAffinityWaiters(klog.FromContext(ctx), sets.New(p, q))
		if handle.count() != 1 {
			t.Fatal("a second notification activated an already removed waiter")
		}
		if resolved, err := pl.affinityTracker.exclusionResolved(q); err != nil || resolved {
			t.Fatalf("activation incorrectly required or changed sibling readiness: resolved=%t, err=%v", resolved, err)
		}
	})

	t.Run("owner-commit-activates-without-another-cluster-event", func(t *testing.T) {
		ctx, tc := reconcileSetup(t, poolEventsSlices())
		pl, handle, _ := affinityPluginWaiter(t, ctx, tc, pod)
		c := tc.draManager.resourceClaimTracker
		entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		t.Cleanup(unblock)
		c.allocatedDevices.observedAfterComputeHook = func() {
			close(entered)
			<-release
		}
		go func() {
			defer close(done)
			assumecache.AddTestObject(c.cache, sharingAffinityAccountingClaim("held-owner", "blue"))
		}()
		reconcileAwait(t, entered, "owner computation")
		pl.registerAffinityWaiters(klog.FromContext(ctx), pod, sets.New(p))
		if handle.count() != 0 {
			t.Fatal("pending owner activated waiter before its commit")
		}
		unblock()
		reconcileAwait(t, done, "owner commit")
		affinityWaitForActivation(t, ctx, handle, pod)
		assertAffinityWaiterIndexes(t, pl.affinityWaiters, nil, nil)
	})

	t.Run("held-publication-and-independent-profile-cleanup", func(t *testing.T) {
		slices := poolEventsSlices()
		ctx, tc := reconcileSetup(t, slices)
		first, firstHandle, cancelFirst := affinityPluginWaiter(t, ctx, tc, pod)
		second, secondHandle, _ := affinityPluginWaiter(t, ctx, tc, pod)
		c := tc.draManager.resourceClaimTracker
		c.listenersMu.Lock()
		listenersBeforeCancel := len(c.listeners)
		c.listenersMu.Unlock()
		entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
		var publications atomic.Int32
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		t.Cleanup(unblock)
		c.publishBeforeCommitHook = func(driver, pool string) {
			if driver == p.driver && pool == p.pool && publications.Add(1) == 1 {
				close(entered)
				<-release
			}
		}
		changed := slices[0].DeepCopy()
		changed.Spec.SharingAffinity[0].Expression = `"changed"`
		go func() {
			defer close(done)
			poolEventsChange(t, ctx, tc, changed, false)
		}()
		reconcileAwait(t, entered, "held metadata publication")
		for _, pl := range []*DynamicResources{first, second} {
			pl.registerAffinityWaiters(klog.FromContext(ctx), pod, sets.New(p))
			pl.activateAffinityWaiters(klog.FromContext(ctx), sets.New(p))
		}
		if firstHandle.count() != 0 || secondHandle.count() != 0 {
			t.Fatal("lister-ahead metadata activated waiters before publication")
		}
		cancelFirst()
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			first.affinityWaiters.mu.Lock()
			closed := first.affinityWaiters.closed
			first.affinityWaiters.mu.Unlock()
			c.listenersMu.Lock()
			unsubscribed := len(c.listeners) == listenersBeforeCancel-1
			c.listenersMu.Unlock()
			return closed && unsubscribed, nil
		}); err != nil {
			t.Fatalf("cancelled profile did not close its waiter registry and unsubscribe: %v", err)
		}
		assertAffinityWaiterIndexes(t, first.affinityWaiters, nil, nil)
		first.registerAffinityWaiters(klog.FromContext(ctx), pod, sets.New(p))
		unblock()
		reconcileAwait(t, done, "metadata publication")
		affinityWaitForActivation(t, ctx, secondHandle, pod)
		if firstHandle.count() != 0 {
			t.Fatal("cancelled profile received an activation")
		}
		assertAffinityWaiterIndexes(t, second.affinityWaiters, nil, nil)
	})
}

type affinityNodeAllocator struct{}

func (affinityNodeAllocator) Allocate(context.Context, *v1.Node, []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, error) {
	return nil, fmt.Errorf("Filter must use AllocateWithAffinity")
}

func (affinityNodeAllocator) AllocateWithAffinity(_ context.Context, node *v1.Node, _ []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, []structured.ClaimAffinity, error) {
	if strings.HasPrefix(node.Name, "excluded-") {
		return nil, nil, structured.ErrFailedAllocationOnNode
	}
	result := sharingAffinityTestResult("req", "device", "share")
	result.Pool = "healthy"
	return []resourceapi.AllocationResult{{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{result}}}},
		[]structured.ClaimAffinity{{Devices: []structured.DeviceAffinity{{
			ResultIndex: 0, Request: result.Request, Device: structured.MakeDeviceID(result.Driver, result.Pool, result.Device),
			Pool: sharingaffinity.PoolKey{Driver: result.Driver, Name: result.Pool, Generation: 1}, Digest: structured.EmptyExtractorDigest,
		}}}}, nil
}

func TestSharingAffinityReconciliation_ParallelFilterExclusions(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	_, ctx := ktesting.NewTestContext(t)
	claim, slice, _ := filterAffinityFixture()
	slice.Spec.AllNodes = nil
	slice.Spec.NodeSelector = &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{
		MatchExpressions: []v1.NodeSelectorRequirement{{Key: "pool", Operator: v1.NodeSelectorOpIn, Values: []string{"excluded"}}},
	}}}
	p := poolName{slice.Spec.Driver, slice.Spec.Pool.Name}
	cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, affinityNodeAllocator{})
	state.affinityExcluded = sets.New(p)
	state.affinityExcludedReasons = map[poolName]string{p: "computing"}
	state.affinitySlices = []*resourceapi.ResourceSlice{slice}
	state.affinityMetadata = map[poolName]uint64{{slice.Spec.Driver, "healthy"}: 7}
	beforeSlice := slice.DeepCopy()
	beforeExcluded := state.affinityExcluded.Clone()
	beforeReasons := maps.Clone(state.affinityExcludedReasons)
	beforeMetadata := maps.Clone(state.affinityMetadata)
	pl := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
	var workers sync.WaitGroup
	for i := 0; i < 32; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			group, want := "healthy", fwk.Success
			if i%2 == 0 {
				group, want = "excluded", fwk.Unschedulable
			}

			node := framework.NewNodeInfo()
			node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-%d", group, i), Labels: map[string]string{"pool": group}}})
			status := pl.Filter(ctx, cycle, &v1.Pod{}, node)
			if status.Code() != want {
				t.Errorf("Filter on %s = %v, want %v", node.Node().Name, status, want)
			}
		}()
	}
	workers.Wait()
	if !state.affinityExcluded.Equal(beforeExcluded) || !maps.Equal(state.affinityExcludedReasons, beforeReasons) ||
		!maps.Equal(state.affinityMetadata, beforeMetadata) || !reflect.DeepEqual(slice, beforeSlice) {
		t.Fatal("parallel Filter mutated captured exclusions, metadata, or ResourceSlices")
	}
	if len(state.nodeAllocations) != 16 {
		t.Fatalf("successful node plans = %d, want 16 healthy nodes", len(state.nodeAllocations))
	}
	for node := range state.nodeAllocations {
		if !strings.HasPrefix(node, "healthy-") {
			t.Errorf("excluded node retained a plan: %s", node)
		}
	}
}

func TestSharingAffinityReconciliation_CapturedPoolReadiness(t *testing.T) {
	t.Run("publication-notifies-after-releasing-tracker-locks", func(t *testing.T) {
		c, _ := newPoolEventsUnitTracker(t, poolEventsSlices()...)
		calls := 0
		unsubscribe := c.subscribeExclusionChanges(func(_ klog.Logger, _ sets.Set[poolName], _ bool) {
			calls++
			for name, lock := range map[string]*sync.Mutex{"publication": &c.publishMutex, "listeners": &c.listenersMu} {
				if !lock.TryLock() {
					t.Errorf("%s lock held during exclusion callback", name)
				} else {
					lock.Unlock()
				}
			}
			if !c.inFlightMutex.TryLock() {
				t.Error("in-flight lock held during exclusion callback")
			} else {
				c.inFlightMutex.Unlock()
			}
			if !c.allocatedDevices.mutex.TryLock() {
				t.Error("accounting lock held during exclusion callback")
			} else {
				c.allocatedDevices.mutex.Unlock()
			}
		})
		defer unsubscribe()
		c.publishMetadata(true, nil)
		if calls != 1 {
			t.Fatalf("publication notifications=%d, want one", calls)
		}
	})

	for _, mode := range []string{"no-affinity", "newly-added-affinity", "active-fence", "established-lock"} {
		t.Run(mode, func(t *testing.T) {
			all := poolEventsSlices()
			plain := all[1].DeepCopy()
			plain.Spec.Pool.ResourceSliceCount = 1
			key := poolName{plain.Spec.Driver, plain.Spec.Pool.Name}
			poolID := schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}
			device := structured.MakeDeviceID(key.driver, key.pool, plain.Spec.Devices[0].Name)
			c, lister := newPoolEventsUnitTracker(t, plain)
			c.publishMetadata(true, nil)
			published := publishRaceCommitted(c).pools[key]
			a := c.allocatedDevices
			a.mutex.Lock()
			a.ids.Insert(device)
			switch mode {
			case "active-fence":
				a.fenced.Insert(device)
			case "established-lock":
				a.affinityDevices[device] = &deviceAffinity{established: map[string]string{"subnet": "blue"}}
			}
			a.mutex.Unlock()
			ahead := plain.DeepCopy()
			ahead.Spec.Pool.Generation++
			listed := []*resourceapi.ResourceSlice{ahead}
			if mode == "newly-added-affinity" {
				ahead.Spec.Pool.ResourceSliceCount = 2
				metadata := all[0].DeepCopy()
				metadata.Spec.Pool.Generation = ahead.Spec.Pool.Generation
				listed = append(listed, metadata)
			}
			lister.set(listed...)
			before := make([]*resourceapi.ResourceSlice, len(listed))
			for i, slice := range listed {
				before[i] = slice.DeepCopy()
			}
			wantExcluded := mode != "no-affinity"
			check := func(snapshot affinityCycleSnapshot) {
				t.Helper()
				_, stale := snapshot.state.DeviceSharingAffinity.StalePools[poolID]
				if snapshot.excluded.Has(key) != wantExcluded || stale != wantExcluded {
					t.Errorf("excluded=%v stale=%v, want exclusion=%t", snapshot.excluded, snapshot.state.DeviceSharingAffinity.StalePools, wantExcluded)
				}
				if wantExcluded {
					if snapshot.reasons[key] != "listerAhead" {
						t.Errorf("exclusion reason=%q, want listerAhead", snapshot.reasons[key])
					}
					if _, captured := snapshot.captured[key]; captured {
						t.Error("excluded pool retained Reserve provenance")
					}
				} else if snapshot.captured[key] != published {
					t.Errorf("captured metadata=%+v, want publication %+v", snapshot.captured[key], published)
				}
				if !snapshot.state.AllocatedDevices.Has(device) {
					t.Error("pool readiness snapshot lost existing allocation")
				}
			}
			snapshot, err := c.gatherAllocatedStateForSlices(listed)
			if err != nil {
				t.Fatalf("capture pool readiness: %v", err)
			}
			check(snapshot)
			resolved, err := c.exclusionResolved(key)
			if err != nil || resolved == wantExcluded {
				t.Errorf("waiter resolution=%t err=%v, want resolved=%t", resolved, err, !wantExcluded)
			}
			snapshot.excluded.Insert(poolName{"unrelated.example", "mutated"})
			snapshot.reasons[key] = "mutated"
			snapshot.captured[key] = poolMeta{epoch: 999}
			snapshot.state.AllocatedDevices.Delete(device)
			delete(snapshot.state.DeviceSharingAffinity.StalePools, poolID)
			again, err := c.gatherAllocatedStateForSlices(listed)
			if err != nil {
				t.Fatalf("capture readiness after caller mutation: %v", err)
			}
			check(again)
			if again.excluded.Has(poolName{"unrelated.example", "mutated"}) || !reflect.DeepEqual(listed, before) {
				t.Error("caller mutation escaped into another snapshot or cached slices")
			}
		})
	}

	t.Run("startup-sentinel-requires-registration-and-owner-replay", func(t *testing.T) {
		slices := poolEventsSlices()
		c, _ := newPoolEventsUnitTracker(t, slices...)
		for _, synced := range []bool{false, true} {
			for _, ready := range []bool{false, true} {
				c.allocatedDevices.mutex.Lock()
				c.metadata.synced = synced
				c.allocatedDevices.startupReady = ready
				c.allocatedDevices.mutex.Unlock()
				resolved, err := c.exclusionResolved(poolName{})
				want := synced && ready
				if err != nil || resolved != want {
					t.Errorf("synced=%t ready=%t: startup resolved=%t err=%v, want %t", synced, ready, resolved, err, want)
				}
				_, err = c.gatherAllocatedStateForSlices(slices)
				if !want && !errors.Is(err, errAffinityMetadataNotCoherent) || want && err != nil {
					t.Errorf("synced=%t ready=%t: snapshot error=%v, want coherent=%t", synced, ready, err, want)
				}
			}
		}
	})
}

// Waiters must be removed by the real Pod informer callbacks, not only by
// direct registry calls.
func TestSharingAffinityReconciliation_InformerWaiterCleanup(t *testing.T) {
	q := poolName{accountingDriver, "incomplete"}
	for _, mode := range []string{"delete", "bind", "terminating"} {
		t.Run(mode, func(t *testing.T) {
			slices := poolEventsSlices()
			incomplete := slices[0].DeepCopy()
			incomplete.Name, incomplete.Spec.Pool.Name = "incomplete-metadata", q.pool
			owner := sharingAffinityAccountingClaim("incomplete-owner", "blue")
			owner.Status.Allocation.Devices.Results[0].Pool = q.pool
			ctx, tc := reconcileSetup(t, append(slices, incomplete), owner)
			handle := &affinityActivationHandle{factory: tc.informerFactory}
			pl := &DynamicResources{draManager: tc.draManager, fh: handle}
			if err := pl.initAffinityWaiters(ctx); err != nil {
				t.Fatalf("initialize waiters: %v", err)
			}
			tc.informerFactory.Start(ctx.Done())
			tc.informerFactory.WaitForCacheSync(ctx.Done())
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "waiting-" + mode}}
			created, err := tc.client.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("create Pod: %v", err)
			}
			if created.UID == "" {
				created.UID = "uid-" + types.UID(mode)
				if created, err = tc.client.CoreV1().Pods(pod.Namespace).Update(ctx, created, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("set Pod UID: %v", err)
				}
			}
			lister := tc.informerFactory.Core().V1().Pods().Lister()
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
				cached, err := lister.Pods(pod.Namespace).Get(pod.Name)
				return err == nil && cached.UID == created.UID, nil
			}); err != nil {
				t.Fatalf("Pod not cached: %v", err)
			}
			pl.registerAffinityWaiters(klog.FromContext(ctx), created, sets.New(q))
			assertAffinityWaiterIndexes(t, pl.affinityWaiters,
				map[poolName]map[types.UID]*v1.Pod{q: {created.UID: created}},
				map[types.UID]sets.Set[poolName]{created.UID: sets.New(q)})
			switch mode {
			case "delete":
				err = tc.client.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			case "bind":
				bound := created.DeepCopy()
				bound.Spec.NodeName = "pool-events-node"
				_, err = tc.client.CoreV1().Pods(pod.Namespace).Update(ctx, bound, metav1.UpdateOptions{})
			case "terminating":
				terminating := created.DeepCopy()
				now := metav1.Now()
				terminating.DeletionTimestamp = &now
				_, err = tc.client.CoreV1().Pods(pod.Namespace).Update(ctx, terminating, metav1.UpdateOptions{})
			}
			if err != nil {
				t.Fatalf("%s Pod: %v", mode, err)
			}
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
				pl.affinityWaiters.mu.Lock()
				defer pl.affinityWaiters.mu.Unlock()
				return len(pl.affinityWaiters.byPod) == 0 && len(pl.affinityWaiters.byPool) == 0, nil
			}); err != nil {
				t.Fatalf("informer %s callback did not remove waiter: %v", mode, err)
			}
			if handle.count() != 0 {
				t.Fatalf("removed waiter was activated: %d calls", handle.count())
			}
		})
	}
}

// A composite child cycle excludes a pool; reconciliation resolves it before
// the root PodGroupPostFilter. The root must still see the child's exclusions,
// register the member and activate it exactly once via register-then-check.
func TestSharingAffinityCompositeRecoveryBeforeRootPostFilter(t *testing.T) {
	q := poolName{accountingDriver, "incomplete"}
	slices := poolEventsSlices()
	incomplete := slices[0].DeepCopy()
	incomplete.Name, incomplete.Spec.Pool.Name = "incomplete-metadata", q.pool
	incompleteOwner := sharingAffinityAccountingClaim("incomplete-owner", "blue")
	incompleteOwner.Status.Allocation.Devices.Results[0].Pool = q.pool
	ctx, tc := reconcileSetup(t, append(slices, incomplete), incompleteOwner)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "waiting", UID: "waiting"}}
	pl, handle, _ := affinityPluginWaiter(t, ctx, tc, pod)
	if resolved, err := pl.affinityTracker.exclusionResolved(q); err != nil || resolved {
		t.Fatalf("incomplete pool resolved=%t, err=%v", resolved, err)
	}
	key := poolName{accountingDriver, accountingPool}
	if resolved, err := tc.draManager.resourceClaimTracker.exclusionResolved(key); err != nil || !resolved {
		t.Fatalf("recovered pool not ready: %v, %v", resolved, err)
	}

	rootState := framework.NewCycleState()
	placement := framework.NewCycleState()
	placement.SetPodGroupCycleState(rootState)
	childState := framework.NewCycleState()
	childState.SetPlacementCycleState(placement)
	pl.publishPodStateToAncestors(childState, pod, &stateData{affinityExcluded: sets.New(key)})

	root := &framework.PodGroupInfo{
		GenericPodGroup: fwk.NewGenericCompositePodGroup(&schedulingv1alpha3.CompositePodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "cpg", UID: "cpg"}}),
		Children: []*framework.PodGroupInfo{{
			GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "leaf", UID: "leaf"}}),
			UnscheduledPods: []*v1.Pod{pod},
		}},
	}
	if _, status := pl.PodGroupPostFilter(ctx, rootState, root, nil); status.Code() != fwk.Unschedulable {
		t.Fatalf("PodGroupPostFilter = %v, want Unschedulable", status)
	}
	affinityWaitForActivation(t, ctx, handle, pod)

	// Unresolved exclusions must not activate unconditionally.
	other := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "other", UID: "other"}}
	rootState2 := framework.NewCycleState()
	placement2 := framework.NewCycleState()
	placement2.SetPodGroupCycleState(rootState2)
	child2 := framework.NewCycleState()
	child2.SetPlacementCycleState(placement2)
	pl.publishPodStateToAncestors(child2, other, &stateData{affinityExcluded: sets.New(q)})
	if err := tc.informerFactory.Core().V1().Pods().Informer().GetIndexer().Add(other); err != nil {
		t.Fatal(err)
	}
	root.Children[0].UnscheduledPods = []*v1.Pod{other}
	if _, status := pl.PodGroupPostFilter(ctx, rootState2, root, nil); status.Code() != fwk.Unschedulable {
		t.Fatalf("PodGroupPostFilter = %v, want Unschedulable", status)
	}
	if got := handle.count(); got != 1 {
		t.Fatalf("activations = %d, want 1 (unresolved pool must not activate)", got)
	}
	if !pl.affinityWaiters.byPod[other.UID].Has(q) {
		t.Fatalf("unresolved member not registered")
	}
}

// After affinity metadata is removed, a surviving allocated owner has no
// contributions. Replacing another owner must not drain the device's fence.
func TestSharingAffinityReplacementHoldPreservesContributionlessSurvivor(t *testing.T) {
	c, lister := newInterleaveTracker(t, poolEventsSlices())
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	old.Name = "replaced"
	survivor := sharingAffinityAccountingClaim("survivor", "blue")
	for _, owner := range []*resourceapi.ResourceClaim{old, survivor} {
		assumecache.AddTestObject(c.cache, owner)
		a.onAdd(owner)
	}
	interleaveDrain(t, t.Context(), c)
	device := sharingAffinityAccountingDevice(old)
	withoutAffinity := reconcileSlices(poolEventsSlices(), 2)
	for _, slice := range withoutAffinity {
		slice.Spec.SharingAffinity = nil
	}
	lister.set(withoutAffinity...)
	c.publishMetadata(false, nil)
	interleaveDrain(t, t.Context(), c)
	assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusUnreconstructable, nil)
	replacement := sharingAffinityAccountingClaim("replacement", "red")
	replacement.Name, replacement.ResourceVersion = old.Name, "2"
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	a.onAdd(replacement)
	interleaveDrain(t, t.Context(), c)
	state := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingCapacity(t, state, device, "10")
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
}
