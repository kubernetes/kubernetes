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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func TestSharingAffinityPodGroupPostFilter_RegistersAffinityWaitersWithoutDeallocation(t *testing.T) {
	p := poolName{accountingDriver, accountingPool}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "waiting", UID: "waiting"}}
	podState := &stateData{
		claims:           newClaimStore(nil, nil, nil),
		affinityExcluded: sets.New(p),
	}

	for _, tc := range []struct {
		name           string
		features       feature.Features
		pgInfo         fwk.PodGroupInfo
		wantActivation bool
	}{
		{
			name:     "basic-workload-claim-gate-off",
			features: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true},
			pgInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "pg", UID: "pg"}}),
				UnscheduledPods: []*v1.Pod{pod},
			},
			wantActivation: true,
		},
		{
			name:     "composite-workload-claim-gate-on",
			features: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true, EnableDRAWorkloadResourceClaims: true},
			pgInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericCompositePodGroup(&schedulingv1alpha3.CompositePodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "cpg", UID: "cpg"}}),
				Children: []*framework.PodGroupInfo{{
					GenericPodGroup: fwk.NewGenericPodGroup(&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "leaf", UID: "leaf"}}),
					UnscheduledPods: []*v1.Pod{pod},
				}},
			},
			wantActivation: true,
		},
		{
			name:     "topology-aware-workload-claim-gate-on",
			features: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true, EnableDRAWorkloadResourceClaims: true, EnableTopologyAwareWorkloadScheduling: true},
			pgInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingapi.PodGroup{
					ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "topology", UID: "topology"},
					Spec: schedulingapi.PodGroupSpec{
						SchedulingConstraints: &schedulingapi.PodGroupSchedulingConstraints{
							Topology: []schedulingapi.TopologyConstraint{{Key: "rack"}},
						},
					},
				}),
				UnscheduledPods: []*v1.Pod{pod},
			},
			wantActivation: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, setup := reconcileSetup(t, poolEventsSlices())
			pl, handle, _ := affinityPluginWaiter(t, ctx, setup, pod)
			pl.fts = tc.features
			cycle := framework.NewCycleState()
			cycle.Write(stateKey, &podGroupStateData{
				podsStateData: map[types.NamespacedName]*stateData{
					{Namespace: pod.Namespace, Name: pod.Name}: podState,
				},
			})
			before := gatherSharingAffinityAccounting(t, setup.draManager.resourceClaimTracker)

			_, status := pl.PodGroupPostFilter(ctx, cycle, tc.pgInfo, func(context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
				t.Fatal("DynamicResources PodGroupPostFilter must not reschedule directly")
				return nil, nil
			})
			if status.Code() != fwk.Unschedulable {
				t.Fatalf("PodGroupPostFilter status = %v, want Unschedulable", status)
			}
			after := gatherSharingAffinityAccounting(t, setup.draManager.resourceClaimTracker)
			if before.DeviceSharingAffinity.DeviceStates[structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)].Status != after.DeviceSharingAffinity.DeviceStates[structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)].Status ||
				len(after.AllocatedSharedDeviceIDs) != len(before.AllocatedSharedDeviceIDs) || len(after.AllocatedDevices) != len(before.AllocatedDevices) {
				t.Fatalf("PodGroupPostFilter mutated allocation accounting: before=%#v after=%#v", before, after)
			}
			if tc.wantActivation {
				affinityWaitForActivation(t, ctx, handle, pod)
				assertAffinityWaiterIndexes(t, pl.affinityWaiters, nil, nil)
			}
		})
	}
}

func TestSharingAffinityInformerTombstoneAndRepeatedPostFilterCleanup(t *testing.T) {
	q := poolName{accountingDriver, "incomplete"}
	slices := poolEventsSlices()
	incomplete := slices[0].DeepCopy()
	incomplete.Name, incomplete.Spec.Pool.Name = "incomplete-metadata", q.pool
	owner := sharingAffinityAccountingClaim("incomplete-owner", "blue")
	owner.Status.Allocation.Devices.Results[0].Pool = q.pool
	var lifecycleHandler cache.ResourceEventHandler
	affinityPodLifecycleHandlerHook = func(handler cache.ResourceEventHandler) {
		lifecycleHandler = handler
	}
	t.Cleanup(func() { affinityPodLifecycleHandlerHook = nil })
	ctx, setup := reconcileSetup(t, append(slices, incomplete), owner)
	pl, handle, _ := affinityPluginWaiter(t, ctx, setup, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "seed", UID: "seed"}})
	if lifecycleHandler == nil {
		t.Fatal("Pod lifecycle handler was not registered")
	}
	informer := setup.informerFactory.Core().V1().Pods().Informer()

	for i, mode := range []string{"tombstone", "postfilter-delete", "postfilter-delete-again"} {
		t.Run(mode, func(t *testing.T) {
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: mode, UID: types.UID(mode)}}
			if err := informer.GetIndexer().Add(pod); err != nil {
				t.Fatalf("cache Pod: %v", err)
			}
			if i == 0 {
				pl.registerAffinityWaiters(klog.FromContext(ctx), pod, sets.New(q))
			} else {
				cycle, state := filterAffinityState(nil, nil)
				state.affinityExcluded = sets.New(q)
				if _, status := pl.PostFilter(ctx, cycle, pod, nil); status.Code() != fwk.Unschedulable {
					t.Fatalf("PostFilter = %v, want Unschedulable", status)
				}
			}
			assertAffinityWaiterIndexes(t, pl.affinityWaiters,
				map[poolName]map[types.UID]*v1.Pod{q: {pod.UID: pod}},
				map[types.UID]sets.Set[poolName]{pod.UID: sets.New(q)})
			if i == 0 {
				lifecycleHandler.OnDelete(cache.DeletedFinalStateUnknown{Key: pod.Namespace + "/" + pod.Name, Obj: pod})
			} else {
				if err := informer.GetStore().Delete(pod); err != nil {
					t.Fatalf("delete Pod from store: %v", err)
				}
				lifecycleHandler.OnDelete(pod)
			}
			assertAffinityWaiterIndexes(t, pl.affinityWaiters, nil, nil)
			if handle.count() != 0 {
				t.Fatalf("removed waiter was activated: %d calls", handle.count())
			}
		})
	}
}

func TestSharingAffinityReserve_RealOwnerUpdateErrorBackoffRetry(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)
	ctx, cancel := context.WithCancel(tCtx)
	defer cancel()
	owner := sharingAffinityAccountingClaim("owner", "blue")
	claim := reserveClaim("candidate", "blue")
	f := newReserveFixtureWithContext(t, tCtx, true, owner, claim)
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return f.snapshot(t).DeviceSharingAffinity.DeviceStates[structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)].Status == structured.AffinityStatusLocked, nil
	}); err != nil {
		t.Fatalf("initial owner did not settle: %v", err)
	}

	pod := reservePod("candidate-pod", claim)
	q := internalqueue.NewSchedulingQueue(nil, f.tc.informerFactory,
		internalqueue.WithMetricsRecorder(metrics.NewMetricsAsyncRecorder(10, time.Second, ctx.Done())),
		internalqueue.WithPodInitialBackoffDuration(50*time.Millisecond),
		internalqueue.WithPodMaxBackoffDuration(50*time.Millisecond))
	defer q.Close()
	q.Run(logger)
	q.Add(ctx, pod)
	popped, err := q.Pop(logger)
	if err != nil {
		t.Fatalf("Pop: %v", err)
	}
	pInfo := popped.(*framework.QueuedPodInfo)
	state := f.filter(t, pod, nil)

	c := f.tc.draManager.resourceClaimTracker
	a := c.allocatedDevices
	entered, release := make(chan struct{}), make(chan struct{})
	var once, enterOnce sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	a.mutex.Lock()
	a.observedAfterComputeHook = func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	}
	a.mutex.Unlock()
	cached, err := c.cache.Get(owner.Namespace + "/" + owner.Name)
	if err != nil {
		t.Fatal(err)
	}
	updated := cached.(*resourceapi.ResourceClaim).DeepCopy()
	updated.ResourceVersion = "99"
	updated.Status.Allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("7")}
	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		assumecache.AddTestObject(c.cache, updated)
	}()
	reserveAwait(t, entered)

	status := f.tc.p.Reserve(ctx, state, pod, workerNode.Name)
	if status.Code() != fwk.Error || !errors.Is(status.AsError(), errAffinityNotReady) {
		t.Fatalf("Reserve during held owner update = %v, want Error wrapping errAffinityNotReady", status)
	}
	if c.GetPendingAllocation(claim.UID) != nil || len(f.pendingTokens(claim.UID)) != 0 {
		t.Fatal("failed Reserve left pending allocation or tokens")
	}
	f.unreserve(state, pod)
	if err := q.AddUnschedulablePodIfNotPresent(ctx, pInfo, q.SchedulingCycle()); err != nil {
		t.Fatalf("return Reserve Error pod to queue: %v", err)
	}
	unblock()
	reserveAwait(t, handlerDone)
	reserveAssertState(t, f.snapshot(t), "blue", "7")

	retried := make(chan error, 1)
	go func() {
		retry, err := q.Pop(logger)
		if err == nil && retry.(*framework.QueuedPodInfo).Pod.UID != pod.UID {
			err = errors.New("popped unexpected pod")
		}
		retried <- err
	}()
	select {
	case err := <-retried:
		if err != nil {
			t.Fatalf("retry Pop: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Reserve Error pod was not retried from queue backoff")
	}
	retryState := f.filter(t, pod, nil)
	f.reserve(t, retryState, pod)
	f.unreserve(retryState, pod)
	reserveAssertState(t, f.snapshot(t), "blue", "7")
}
