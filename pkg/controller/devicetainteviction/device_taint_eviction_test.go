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
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller/testutil"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
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
	wantEvents []*v1.Event
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

var (
	podKind      = v1.SchemeGroupVersion.WithKind("Pod")
	nodeName     = "worker"
	nodeName2    = "worker-2"
	driver       = "some-driver"
	podName      = "my-pod"
	podUID       = "1234"
	className    = "my-resource-class"
	resourceName = "my-resource"
	claimName    = podName + "-" + resourceName
	namespace    = "default"
	taintTime    = metav1.Date(2025, 2, 26, 12, 05, 30, 0, time.UTC)
	taintKey     = "example.com/taint"
	slice        = st.MakeResourceSlice(nodeName, driver).
			Device("instance").
			Device("instance-no-schedule", resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoSchedule}).
			Device("instance-no-execute", resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime}).
			Obj()
	sliceReplaced = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Name += "-updated"
		slice.Spec.Pool.Generation++
		return slice
	}()
	sliceTainted = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[0].Basic.Taints = []resourceapi.DeviceTaint{{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime}}
		return slice
	}()
	sliceTaintedTwice = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[0].Basic.Taints = []resourceapi.DeviceTaint{
			{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime},
			{Key: taintKey + "-other", Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: &taintTime},
		}
		return slice
	}()
	sliceUntainted = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[1].Basic.Taints = nil
		slice.Spec.Devices[2].Basic.Taints = nil
		return slice
	}()
	slice2 = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Name += "-2"
		slice.Spec.NodeName = nodeName2
		slice.Spec.Pool.Name = nodeName2
		slice.Spec.Pool.Generation++
		return slice
	}()
	claim = st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Request(className).
		Obj()
	allocationResult = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance",
				Request: "req-1",
			}},
		},
	}
	inUseClaim = st.FromResourceClaim(claim).
			OwnerReference(podName, podUID, podKind).
			Allocation(allocationResult).
			ReservedForPod(podName, types.UID(podUID)).
			Obj()
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Node(nodeName).
				Obj()
	podWithClaimTemplate = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimTemplateName: &claimName}).
				Node(nodeName).
				Obj()
	podWithClaimTemplateInStatus = func() *v1.Pod {
		pod := podWithClaimTemplate.DeepCopy()
		pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
			{
				Name:              pod.Spec.ResourceClaims[0].Name,
				ResourceClaimName: &claimName,
			},
		}
		return pod
	}()
	cancelPodEviction = &v1.Event{
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			Namespace:  namespace,
			Name:       podName,
			APIVersion: "v1",
		},
		Reason:  "DeviceTaintManagerEviction",
		Message: "Cancelling deletion",
		Type:    v1.EventTypeNormal,
		Source: v1.EventSource{
			Component: "nodeControllerTest",
		},
		Count: 1,
	}

	matchDeletionEvent = gomega.ConsistOf(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Source":  gomega.HaveField("Component", gomega.Equal("device-taint-eviction")),
		"Reason":  gomega.Equal("DeviceTaintManagerEviction"),
		"Message": gomega.Equal("Marking for deletion"),
		"Count":   gomega.Equal(int32(1)),
		"InvolvedObject": gomega.Equal(v1.ObjectReference{
			Kind:       "Pod",
			Namespace:  namespace,
			Name:       podName,
			APIVersion: "v1",
		}),
	}))
)

func listEvents(tCtx ktesting.TContext) []v1.Event {
	events, err := tCtx.Client().CoreV1().Events("").List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list events")
	return events.Items
}

// TestHandlers exercices the event handler logic. Each test case starts with
// some known state, applies a sequence of independent events in all
// permutations of their order, then checks against the expected final
// state. The final state must be the same in all permutations. This simulates
// the random order in which informer updates can be perceived.
func TestHandlers(t *testing.T) {
	t.Parallel()

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
		"evict-pod-later": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Add(time.Minute)}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time.Add(time.Minute)}},
			},
		},
		"evict-pod-later-many": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Key:               taintKey + "-other",
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(20)),
						},
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Key:               taintKey + "-other",
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(20)),
						},
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Add(30 * time.Second)}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time.Add(30 * time.Second)}},
			},
		},
		"evict-pod-toleration-mismatch": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Key:               taintKey + "-other",
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Key:               taintKey + "-other",
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Time}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
		},
		"evict-pod-toleration-forever": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Effect: resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Effect: resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}()}},
			},
		},
		"evict-pod-toleration-forever-many": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Effect: resourceapi.DeviceTaintEffectNoExecute,
						},
					}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Effect: resourceapi.DeviceTaintEffectNoExecute,
						},
					}
					return claim
				}()}},
			},
		},
		"evict-pod-partial-toleration": {
			events: []any{
				add(sliceTaintedTwice),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Key:      taintKey,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTaintedTwice, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Key:      taintKey,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}(), evictionTime: &taintTime}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
		},
		"evict-pod-many-taints": {
			events: []any{
				add(sliceTaintedTwice),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taintKey,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taintKey + "-other",
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTaintedTwice, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taintKey,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taintKey + "-other",
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Add(30 * time.Second)}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time.Add(30 * time.Second)}},
			},
		},
		"no-evict-pod-not-scheduled": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(func() *v1.Pod {
					pod := podWithClaimName.DeepCopy()
					pod.Spec.NodeName = ""
					return pod
				}()),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
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
			wantEvents: []*v1.Event{cancelPodEviction},
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
			wantEvents: []*v1.Event{cancelPodEviction},
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
			wantEvents: []*v1.Event{cancelPodEviction},
		},
		// TODO: cancel eviction on pod deletion
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
	if len(tContext.recorder.Events) == 0 {
		tContext.recorder.Events = nil
	}
	assert.Equal(tContext, tc.finalState.evicting, tContext.evicting, "evicting pods")
	assert.Equal(tContext, tc.finalState.allocatedClaimsAsMap(), tContext.allocatedClaims, "allocated claims")
	if !assert.Equal(tContext, tc.finalState.slicesAsMap(), tContext.pools, "pools") {
		for key := range tContext.pools {
			assert.Equal(tContext, tc.finalState.slicesAsMap()[key], tContext.pools[key], "pool")
		}
	}
	if diff := cmp.Diff(tc.wantEvents, tContext.recorder.Events, cmpopts.IgnoreTypes(metav1.ObjectMeta{}, metav1.Time{})); diff != "" {
		tContext.Errorf("unexpected events (-want, +got):\n%s", diff)
	}
}

// TestEviction runs through the full flow of starting the controller and evicting one pod.
func TestEviction(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	// This scenario is the same as "evict-pod-resourceclaim" above.
	pod := podWithClaimName.DeepCopy()
	fakeClientset := fake.NewSimpleClientset(
		sliceTainted,
		slice2,
		inUseClaim,
		pod,
	)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	var mutex sync.Mutex
	var podUpdates int
	var updatedPod *v1.Pod
	var podDeletions int

	fakeClientset.PrependReactor("patch", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podUpdates++
		podName := action.(core.PatchAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of patched pod")
		return false, nil, nil
	})
	fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podDeletions++
		podName := action.(core.DeleteAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of deleted pod")
		obj, err := fakeClientset.Tracker().Get(v1.SchemeGroupVersion.WithResource("pods"), pod.Namespace, pod.Name)
		require.NoError(tCtx, err)
		updatedPod = obj.(*v1.Pod)
		return false, nil, nil
	})
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	informerFactory.Start(tCtx.Done())
	defer informerFactory.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		t.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.Run(tCtx)
	}()

	// Eventually the pod gets deleted (= evicted).
	assert.Eventually(tCtx, func() bool {
		_, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 30*time.Second, time.Millisecond, "pod evicted")

	// Now we can check the API calls.
	mutex.Lock()
	defer mutex.Unlock()
	assert.Equal(tCtx, 1, podUpdates, "number of pod update calls")
	assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
	pod.Status.Conditions = []v1.PodCondition{{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  "DeletionByDeviceTaintManager",
		Message: "Device Taint manager: deleting due to NoExecute taint",
	}}
	if diff := cmp.Diff(pod, updatedPod, cmpopts.IgnoreTypes(metav1.Time{})); diff != "" {
		tCtx.Errorf("unexpected modified pod (-want, +got):\n%s", diff)
	}

	// Shortly after deletion we should also see the event.
	ktesting.Eventually(tCtx, listEvents).WithTimeout(10 * time.Second).Should(matchDeletionEvent)

	// We also don't want any other events, in particular not a cancellation event
	// because the pod deletion was observed or another occurrence of the same event.
	ktesting.Consistently(tCtx, listEvents).WithTimeout(5 * time.Second).Should(matchDeletionEvent)
}

// TestCancelEviction deletes the pod before the controller deletes it,
// causing eviction to be cancelled.
func TestCancelEviction(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	// The claim tolerates the taint long enough for us to
	// undo the tainting, which then cancels pod deletion.
	pod := podWithClaimName.DeepCopy()
	slice := sliceTainted.DeepCopy()
	slice.Spec.Devices[0].Basic.Taints[0].TimeAdded = &metav1.Time{Time: time.Now()}
	claim := inUseClaim.DeepCopy()
	claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
		Effect:            resourceapi.DeviceTaintEffectNoExecute,
		TolerationSeconds: ptr.To(int64(60)),
	}}
	fakeClientset := fake.NewSimpleClientset(
		slice,
		claim,
		pod,
	)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	informerFactory.Start(tCtx.Done())
	defer informerFactory.Shutdown()

	var mutex sync.Mutex
	podEvicting := false
	controller.evictPod = func(podRef object, fireAt time.Time) {
		assert.Equal(tCtx, newObject(pod), podRef)
		mutex.Lock()
		defer mutex.Unlock()
		podEvicting = true
	}
	controller.cancelEvict = func(podRef object) bool {
		assert.Equal(tCtx, newObject(pod), podRef)
		mutex.Lock()
		defer mutex.Unlock()
		podEvicting = false
		return false
	}

	var wg sync.WaitGroup
	defer func() {
		t.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.Run(tCtx)
	}()

	// Eventually the pod gets scheduled for eviction.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podEvicting
	}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("pod pending eviction"))

	// Now we can delete the pod.
	tCtx.ExpectNoError(fakeClientset.CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{}))

	// Shortly after deletion we should also see the cancellation.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podEvicting
	}).WithTimeout(30 * time.Second).Should(gomega.BeFalseBecause("pod no longer pending eviction"))

	// And no events!
	ktesting.Consistently(tCtx, listEvents).WithTimeout(5 * time.Second).Should(gomega.BeEmpty())

	// TODO (here and elsewhere): check metrics
}

// TestParallelPodDeletion covers the scenario that a pod gets deleted right before
// trying to evict it.
func TestParallelPodDeletion(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	// This scenario is the same as "evict-pod-resourceclaim" above.
	pod := podWithClaimName.DeepCopy()
	fakeClientset := fake.NewSimpleClientset(
		sliceTainted,
		slice2,
		inUseClaim,
		pod,
	)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	var mutex sync.Mutex
	var podGets int
	var podDeletions int

	fakeClientset.PrependReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podGets++
		podName := action.(core.GetAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of patched pod")

		// This gets called directly before eviction. Pretend that it is deleted.
		err = fakeClientset.Tracker().Delete(v1.SchemeGroupVersion.WithResource("pods"), pod.Namespace, pod.Name)
		assert.NoError(tCtx, err, "delete pod")
		return true, nil, apierrors.NewNotFound(v1.SchemeGroupVersion.WithResource("pods").GroupResource(), pod.Name)
	})
	fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podDeletions++
		podName := action.(core.DeleteAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of deleted pod")
		return false, nil, nil
	})
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	informerFactory.Start(tCtx.Done())
	defer informerFactory.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		t.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.Run(tCtx)
	}()

	// Eventually the pod gets deleted, in this test by us.
	assert.Eventually(tCtx, func() bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podGets >= 1
	}, 30*time.Second, time.Millisecond, "pod eviction started")

	// Now we can check the API calls.
	mutex.Lock()
	defer mutex.Unlock()
	assert.Equal(tCtx, 1, podGets, "number of pod get calls")
	assert.Equal(tCtx, 0, podDeletions, "number of pod delete calls")

	// We don't want any events.
	ktesting.Consistently(tCtx, listEvents).WithTimeout(5 * time.Second).Should(gomega.BeEmpty())
}

// TestRetry covers the scenario that an eviction attempt must be retried.
func TestRetry(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	// This scenario is the same as "evict-pod-resourceclaim" above.
	pod := podWithClaimName.DeepCopy()
	fakeClientset := fake.NewSimpleClientset(
		sliceTainted,
		slice2,
		inUseClaim,
		pod,
	)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	var mutex sync.Mutex
	var podGets int
	var podDeletions int

	fakeClientset.PrependReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podGets++
		podName := action.(core.GetAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of patched pod")

		// This gets called directly before eviction. Pretend that there is an intermittent error.
		if podGets == 1 {
			return true, nil, apierrors.NewInternalError(errors.New("fake error"))
		}
		return false, nil, nil
	})
	fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podDeletions++
		podName := action.(core.DeleteAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of deleted pod")
		return false, nil, nil
	})
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	informerFactory.Start(tCtx.Done())
	defer informerFactory.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		t.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.Run(tCtx)
	}()

	// Eventually the pod gets deleted.
	assert.Eventually(tCtx, func() bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podDeletions >= 1
	}, 30*time.Second, time.Millisecond, "pod eviction done")

	// Now we can check the API calls.
	// TODO: check everything repeatedly.
	mutex.Lock()
	assert.Equal(tCtx, 2, podGets, "number of pod get calls")
	assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
	mutex.Unlock()
	ktesting.Consistently(tCtx, listEvents).WithTimeout(5 * time.Second).Should(matchDeletionEvent)
	mutex.Lock()
	assert.Equal(tCtx, 2, podGets, "number of pod get calls")
	assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
	mutex.Unlock()
}

// TestRetry covers the scenario that an eviction attempt fails.
func TestEvictionFailure(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	// This scenario is the same as "evict-pod-resourceclaim" above.
	pod := podWithClaimName.DeepCopy()
	fakeClientset := fake.NewSimpleClientset(
		sliceTainted,
		slice2,
		inUseClaim,
		pod,
	)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	var mutex sync.Mutex
	var podGets int
	var podDeletions int

	fakeClientset.PrependReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podGets++
		podName := action.(core.GetAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of patched pod")
		return false, nil, nil
	})
	fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		podDeletions++
		podName := action.(core.DeleteAction).GetName()
		assert.Equal(t, podWithClaimName.Name, podName, "name of deleted pod")
		return true, nil, apierrors.NewInternalError(errors.New("fake error"))
	})
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().ResourceSlicePatches(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	informerFactory.Start(tCtx.Done())
	defer informerFactory.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		t.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		controller.Run(tCtx)
	}()

	// Eventually deletion is attempted a few times.
	assert.Eventually(tCtx, func() bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podDeletions >= retries
	}, 30*time.Second, time.Millisecond, "pod eviction failed")

	// Now we can check the API calls.
	mutex.Lock()
	assert.Equal(tCtx, retries, podGets, "number of pod get calls")
	assert.Equal(tCtx, retries, podDeletions, "number of pod delete calls")
	mutex.Unlock()
	ktesting.Consistently(tCtx, listEvents).WithTimeout(5 * time.Second).Should(matchDeletionEvent)
	mutex.Lock()
	assert.Equal(tCtx, retries, podGets, "number of pod get calls")
	assert.Equal(tCtx, retries, podDeletions, "number of pod delete calls")
	mutex.Unlock()
}
