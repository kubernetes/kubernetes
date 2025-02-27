/*
Copyright 2025 The Kubernetes Authors.

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

package devicetainteviction

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller/testutil"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// setup creates a controller which is ready to have its handle* methods called.
func setup(tb testing.TB) *testContext {
	tCtx := ktesting.Init(tb)
	fakeClientset := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	tContext := &testContext{
		TContext:        tCtx,
		Controller:      controller,
		recorder:        testutil.NewFakeRecorder(),
		client:          fakeClientset,
		informerFactory: informerFactory,
	}
	tContext.logger = tCtx.Logger()
	tContext.evictPod = func(pod object, fireAt time.Time) {
		tContext.evicting = append(tContext.evicting, evictAt{pod, fireAt})
		sort.Slice(tContext.evicting, func(i, j int) bool {
			switch strings.Compare(tContext.evicting[i].pod.Namespace, tContext.evicting[j].pod.Namespace) {
			case 1:
				return false
			case -1:
				return true
			}
			return tContext.evicting[i].pod.Name < tContext.evicting[j].pod.Name
		})
	}
	tContext.cancelEvict = func(pod object) bool {
		index := slices.IndexFunc(tContext.evicting, func(e evictAt) bool { return e.pod == pod })
		if index >= 0 {
			tContext.evicting = slices.Delete(tContext.evicting, index, index+1)
			return true
		}
		return false
	}
	tContext.Controller.recorder = tContext.recorder

	return tContext
}

type testContext struct {
	ktesting.TContext
	*Controller
	evicting        []evictAt // sorted by namespace, name
	recorder        *testutil.FakeRecorder
	client          *fake.Clientset
	informerFactory informers.SharedInformerFactory
}

type evictAt struct {
	pod    object
	fireAt time.Time
}

type state struct {
	pods            []*v1.Pod
	allocatedClaims []allocatedClaim
	slices          []*resourceapi.ResourceSlice
	evicting        []evictAt
}

func (s state) allocatedClaimsAsMap() map[types.NamespacedName]allocatedClaim {
	claims := make(map[types.NamespacedName]allocatedClaim)
	for _, claim := range s.allocatedClaims {
		claims[types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name}] = claim
	}
	return claims
}

func (s state) slicesAsMap() map[poolID]pool {
	pools := make(map[poolID]pool)
	for _, slice := range s.slices {
		id := poolID{driverName: slice.Spec.Driver, poolName: slice.Spec.Pool.Name}
		pool := pools[id]
		if pool.slices == nil {
			pool.slices = sets.New[*resourceapi.ResourceSlice]()
		}
		pool.slices.Insert(slice)
		pools[id] = pool
	}
	for id, pool := range pools {
		maxGeneration := int64(math.MinInt64)
		for slice := range pool.slices {
			if slice.Spec.Pool.Generation > maxGeneration {
				maxGeneration = slice.Spec.Pool.Generation
			}
		}
		pool.maxGeneration = maxGeneration
		pools[id] = pool
	}
	return pools
}

type testCase struct {
	initialState state

	// events contains pairs of old and new objects which will
	// be passed to handle* methods.
	// Objects can be slices, claims, and pods.
	// [add], [remove], and [update] can be used to produce
	// such pairs.
	events []any

	finalState state
}

func add[T any](obj *T) [2]*T {
	return [2]*T{nil, obj}
}

func remove[T any](obj *T) [2]*T {
	return [2]*T{obj, nil}
}

func update[T any](oldObj, newObj *T) [2]*T {
	return [2]*T{oldObj, newObj}
}

// TestHandlers exercices the event handler logic. Each test case starts with
// some known state, applies a sequence of independent events in all
// permutations of their order, then checks against the expected final
// state. The final state must be the same in all permutations. This simulates
// the random order in which informer updates can be perceived.
func TestHandlers(t *testing.T) {
	podKind := v1.SchemeGroupVersion.WithKind("Pod")
	nodeName := "worker"
	nodeName2 := "worker-2"
	driver := "some-driver"
	podName := "my-pod"
	podUID := "1234"
	className := "my-resource-class"
	resourceName := "my-resource"
	claimName := podName + "-" + resourceName
	namespace := "default"
	taintTime := metav1.Date(2025, 2, 26, 12, 05, 30, 0, time.UTC)
	slice := st.MakeResourceSlice(nodeName, driver).
		Device("instance").
		Device("instance-no-schedule", resourceapi.DeviceTaint{Effect: resourceapi.DeviceTaintEffectNoSchedule}).
		Device("instance-no-execute", resourceapi.DeviceTaint{Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime}).
		Obj()
	sliceReplaced := slice.DeepCopy()
	sliceReplaced.Name += "-updated"
	sliceReplaced.Spec.Pool.Generation++
	sliceTainted := slice.DeepCopy()
	sliceTainted.Spec.Devices[0].Basic.Taints = []resourceapi.DeviceTaint{{Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime}}
	sliceUntainted := slice.DeepCopy()
	sliceUntainted.Spec.Devices[1].Basic.Taints = nil
	sliceUntainted.Spec.Devices[2].Basic.Taints = nil
	slice2 := slice.DeepCopy()
	slice2.Spec.NodeName = nodeName2
	slice2.Spec.Pool.Name = nodeName2
	slice2.Spec.Pool.Generation++
	claim := st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Request(className).
		Obj()
	allocationResult := &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance",
				Request: "req-1",
			}},
		},
	}
	inUseClaim := st.FromResourceClaim(claim).
		OwnerReference(podName, podUID, podKind).
		Allocation(allocationResult).
		ReservedForPod(podName, types.UID(podUID)).
		Obj()
	podWithClaimName := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
		Obj()
	podWithClaimTemplate := st.MakePod().Name(podName).Namespace(namespace).
		UID(podUID).
		PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimTemplateName: &claimName}).
		Obj()
	podWithClaimTemplateInStatus := podWithClaimTemplate.DeepCopy()
	podWithClaimTemplateInStatus.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
		{
			Name:              podWithClaimTemplateInStatus.Spec.ResourceClaims[0].Name,
			ResourceClaimName: &claimName,
		},
	}

	for name, tc := range map[string]testCase{
		"empty": {},
		"populate-pools": {
			events: []any{
				add(slice),
				add(slice2),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{slice, slice2},
			},
		},
		"update-pools": {
			initialState: state{
				slices: []*resourceapi.ResourceSlice{slice, slice2},
			},
			events: []any{
				update(slice, sliceUntainted),
				remove(slice2),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceUntainted},
			},
		},
		"untainted-claim": {
			events: []any{
				add(slice),
				add(slice2),
				add(inUseClaim),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{slice, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
		"tainted-claim": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
			},
		},
		"evict-pod-resourceclaim": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
		},
		"evict-pod-resourceclaimtemplate": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplateInStatus),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
		},
		"no-evict-pod-resourceclaimtemplate-no-status": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplate),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
			},
		},
		"cancel-eviction-remove-taint": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				update(sliceTainted, slice),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{slice, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
		"cancel-eviction-remove-taint-in-new-slice": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				// This moves the in-use device from one slice to another and removes the taint at the same time.
				remove(sliceTainted),
				add(sliceReplaced),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceReplaced, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
		"cancel-eviction-remove-slice": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				remove(sliceTainted),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			numEvents := len(tc.events)
			if numEvents <= 1 {
				// No permutations.
				tContext := setup(t)
				testHandlers(tContext, tc)
				return
			}

			permutation := make([]int, numEvents)
			var permutate func(depth int)
			permutate = func(depth int) {
				if depth >= numEvents {
					// Define a sub-test which runs the current permutation of events.
					events := make([]any, numEvents)
					for i := 0; i < numEvents; i++ {
						events[i] = tc.events[permutation[i]]
					}
					tc := tc
					tc.events = events
					name := strings.Trim(fmt.Sprintf("%v", permutation), "[]")
					t.Run(name, func(t *testing.T) {
						tContext := setup(t)
						testHandlers(tContext, tc)
					})
					return
				}
				for i := 0; i < numEvents; i++ {
					if slices.Index(permutation[0:depth], i) != -1 {
						// Already taken.
						continue
					}
					// Pick it for the current position in permutation,
					// continue with next position.
					permutation[depth] = i
					permutate(depth + 1)
				}
			}
			permutate(0)
		})
	}
}

func testHandlers(tContext *testContext, tc testCase) {
	// Shallow copy of slice and maps is sufficient for now.
	tContext.evicting = slices.Clone(tc.initialState.evicting)
	tContext.allocatedClaims = tc.initialState.allocatedClaimsAsMap()
	tContext.pools = tc.initialState.slicesAsMap()

	// Pods are the only items which get retrieved from the informer cache,
	// so for those (and only those) we have to keep the store up-to-date.
	store := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()
	for _, pod := range tc.initialState.pods {
		store.Add(pod)
	}

	for _, event := range tc.events {
		switch pair := event.(type) {
		case [2]*resourceapi.ResourceSlice:
			tContext.handleSliceChange(pair[0], pair[1])
		case [2]*resourceapi.ResourceClaim:
			tContext.handleClaimChange(pair[0], pair[1])
		case [2]*v1.Pod:
			switch {
			case pair[0] != nil && pair[1] != nil:
				store.Update(pair[1])
			case pair[0] != nil:
				store.Delete(pair[0])
			default:
				store.Add(pair[1])
			}
			tContext.handlePodChange(pair[0], pair[1])
		default:
			tContext.Fatalf("unexpected event type %T", event)
		}
	}

	// Use nil instead of empty map or slice to avoid nil vs. empty differences.
	if len(tContext.evicting) == 0 {
		tContext.evicting = nil
	}
	assert.Equal(tContext, tc.finalState.evicting, tContext.evicting, "evicting pods")
	assert.Equal(tContext, tc.finalState.allocatedClaimsAsMap(), tContext.allocatedClaims, "allocated claims")
	assert.Equal(tContext, tc.finalState.slicesAsMap(), tContext.pools, "pools")
}
