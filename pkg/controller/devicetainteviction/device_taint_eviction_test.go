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
	gomegatypes "github.com/onsi/gomega/types"
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
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction/metrics"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	controllertestutil "k8s.io/kubernetes/pkg/controller/testutil"
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
		informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	tContext := &testContext{
		TContext:        tCtx,
		Controller:      controller,
		recorder:        controllertestutil.NewFakeRecorder(),
		client:          fakeClientset,
		informerFactory: informerFactory,
	}
	tContext.logger = tCtx.Logger()
	// Always log, not matter what the -v value is.
	controller.eventLogger = &tContext.logger
	tContext.evictPod = func(podRef tainteviction.NamespacedObject, fireAt time.Time) {
		// Always replace an existing entry for the same pod.
		index := slices.IndexFunc(tContext.evicting, func(e evictAt) bool {
			return e.podRef == podRef
		})
		e := evictAt{podRef, fireAt}
		if index == -1 {
			tContext.evicting = append(tContext.evicting, e)
		} else {
			tContext.evicting[index] = e
		}
		sort.Slice(tContext.evicting, func(i, j int) bool {
			switch strings.Compare(tContext.evicting[i].podRef.Namespace, tContext.evicting[j].podRef.Namespace) {
			case 1:
				return false
			case -1:
				return true
			}
			return tContext.evicting[i].podRef.Name < tContext.evicting[j].podRef.Name
		})
	}
	tContext.cancelEvict = func(pod tainteviction.NamespacedObject) bool {
		index := slices.IndexFunc(tContext.evicting, func(e evictAt) bool { return e.podRef == pod })
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
	recorder        *controllertestutil.FakeRecorder
	client          *fake.Clientset
	informerFactory informers.SharedInformerFactory
}

type evictAt struct {
	podRef tainteviction.NamespacedObject
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
	//
	// Alternatively, it can also contain a list of such pairs.
	// Those will be applied in the order in which they appear
	// in each event entry.
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
	taintTime    = metav1.Now() // This cannot be a fixed value in the past, otherwise the "seconds since taint time" delta overflows.
	taintKey     = "example.com/taint"
	taintValue   = "something"
	simpleSlice  = st.MakeResourceSlice(nodeName, driver).
			Device("instance").
			Obj()
	slice = st.MakeResourceSlice(nodeName, driver).
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
	sliceUnknownDevices = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		for i := range slice.Spec.Devices {
			slice.Spec.Devices[i].Basic = nil
		}
		return slice
	}()
	sliceTainted = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[0].Basic.Taints = []resourceapi.DeviceTaint{{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, Value: taintValue, TimeAdded: &taintTime}}
		return slice
	}()
	sliceTaintedExtended = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		slice.Spec.Devices = append(slice.Spec.Devices, *slice.Spec.Devices[0].DeepCopy())
		slice.Spec.Devices[len(slice.Spec.Devices)-1].Name += "-other"
		return slice
	}()
	sliceTaintedNoSchedule = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		for i := range slice.Spec.Devices {
			for j := range slice.Spec.Devices[i].Basic.Taints {
				slice.Spec.Devices[i].Basic.Taints[j].Effect = resourceapi.DeviceTaintEffectNoSchedule
			}
		}
		return slice
	}()
	sliceTaintedValueOther = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		for i := range slice.Spec.Devices {
			for j := range slice.Spec.Devices[i].Basic.Taints {
				slice.Spec.Devices[i].Basic.Taints[j].Value += "-other"
			}
		}
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
	// A test may run for an hour without reaching the end of this period.
	tolerationDuration       = 60 * 60 * time.Second
	inUseClaimWithToleration = func() *resourceapi.ResourceClaim {
		claim := inUseClaim.DeepCopy()
		claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
			Key:               taintKey,
			Operator:          resourceapi.DeviceTolerationOpEqual,
			Value:             taintValue,
			Effect:            resourceapi.DeviceTaintEffectNoExecute,
			TolerationSeconds: ptr.To(int64(tolerationDuration.Seconds())),
		}}
		return claim
	}()
	inUseClaimOld = st.FromResourceClaim(inUseClaim).
			OwnerReference(podName, podUID+"-other", podKind).
			UID("other").
			Obj()
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Node(nodeName).
				Obj()
	podWithClaimNameOther = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID + "-other").
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
	podWithClaimTemplateNoClaimInStatus = func() *v1.Pod {
		pod := podWithClaimTemplate.DeepCopy()
		pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
			{
				Name: pod.Spec.ResourceClaims[0].Name,
				// This is valid: it was decided that the pod should run without
				// its claim generated for it. Not used in practice.
				ResourceClaimName: nil,
			},
		}
		return pod
	}()
	cancelPodEviction = &v1.Event{
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Namespace:  namespace,
			Name:       podName,
			UID:        types.UID(podUID),
		},
		Reason:  "DeviceTaintManagerEviction",
		Message: "Cancelling deletion",
		Type:    v1.EventTypeNormal,
		Source: v1.EventSource{
			Component: "nodeControllerTest",
		},
		Count: 1,
	}
)

func matchDeletionEvent() gomegatypes.GomegaMatcher {
	return matchEvent("Marking for deletion")
}

func matchCancellationEvent() gomegatypes.GomegaMatcher {
	return matchEvent("Cancelling deletion")
}

func matchEvent(message string) gomegatypes.GomegaMatcher {
	return gomega.ConsistOf(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
		"Source":  gomega.HaveField("Component", gomega.Equal("device-taint-eviction")),
		"Reason":  gomega.Equal("DeviceTaintManagerEviction"),
		"Message": gomega.Equal(message),
		"Count":   gomega.Equal(int32(1)),
		"InvolvedObject": gomega.Equal(v1.ObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Namespace:  namespace,
			Name:       podName,
			UID:        types.UID(podUID),
		}),
	}))
}

func listEvents(tCtx ktesting.TContext) []v1.Event {
	events, err := tCtx.Client().CoreV1().Events("").List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list events")
	return events.Items
}

// TestHandlers covers the event handler logic. Each test case starts with
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
		"evict-pod-resourceclaim-again": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimName},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
			events: []any{
				[]any{remove(sliceTainted), add(sliceTainted)},
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
			// It is debatable whether the controller should react
			// to slice changes (a deletion in this case)
			// quickly. On the one hand we want to cancel eviction
			// quickly in case that a taint goes away, on the other
			// hand it can also restore the previous state and emit
			// an event, as in this test case.
			//
			// At the moment, the code reliably cancels right away.
			wantEvents: []*v1.Event{cancelPodEviction},
		},
		"evict-pod-resourceclaim-unrelated-changes": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimName},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
			events: []any{
				update(sliceTainted, sliceTaintedExtended),
				update(inUseClaim, inUseClaim),             // No real change here, good enough for testing some code paths.
				update(podWithClaimName, podWithClaimName), // Same here.
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTaintedExtended, slice2},
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
				add(inUseClaimWithToleration),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaimWithToleration, evictionTime: &metav1.Time{Time: taintTime.Add(tolerationDuration)}}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time.Add(tolerationDuration)}},
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
							Operator:          resourceapi.DeviceTolerationOpEqual,
							Value:             taintValue,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(20)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
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
							Operator:          resourceapi.DeviceTolerationOpEqual,
							Value:             taintValue,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(20)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
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
						Operator:          resourceapi.DeviceTolerationOpEqual,
						Value:             taintValue,
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
						Operator:          resourceapi.DeviceTolerationOpEqual,
						Value:             taintValue,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Time}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time}},
			},
		},
		"no-evict-pod-toleration-forever": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
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
						Operator: resourceapi.DeviceTolerationOpExists,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}()}},
			},
		},
		"no-evict-pod-toleration-forever-many": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator: resourceapi.DeviceTolerationOpExists,
							Effect:   resourceapi.DeviceTaintEffectNoExecute,
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
							Operator:          resourceapi.DeviceTolerationOpExists,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator: resourceapi.DeviceTolerationOpExists,
							Effect:   resourceapi.DeviceTaintEffectNoExecute,
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
		"no-evict-no-taint": {
			events: []any{
				add(simpleSlice),
				add(inUseClaim),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{simpleSlice},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
		"no-evict-no-taint-update": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimName},
				slices:          []*resourceapi.ResourceSlice{simpleSlice},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
			events: []any{
				update(simpleSlice, simpleSlice),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{simpleSlice},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
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
		"no-evict-pod-resourceclaimtemplate-no-claim": {
			// It doesn't make sense to have a claim generated for the pod and
			// then have no claim listed for the pod, but for the sake of completeness...
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplateNoClaimInStatus),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
			},
		},
		"no-evict-unknown-device": {
			events: []any{
				add(sliceUnknownDevices),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplateInStatus),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceUnknownDevices, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
		},
		"no-evict-wrong-pod": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimNameOther),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
			},
		},
		"evict-wrong-pod-replaced": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimNameOther},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
			},
			events: []any{
				[]any{remove(podWithClaimNameOther), add(podWithClaimName)},
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimName), taintTime.Time}},
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
		"cancel-eviction-reduce-taint": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				update(sliceTainted, sliceTaintedNoSchedule),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTaintedNoSchedule, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim}},
			},
			wantEvents: []*v1.Event{cancelPodEviction},
		},
		"eviction-change-taint": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaimWithToleration, evictionTime: &metav1.Time{Time: taintTime.Add(tolerationDuration)}}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time.Add(tolerationDuration)}},
			},
			events: []any{
				// Going from a taint which is tolerated for 60 seconds to one which isn't.
				update(sliceTainted, sliceTaintedValueOther),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTaintedValueOther, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaimWithToleration, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
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
		"cancel-eviction-pod-deletion": {
			initialState: state{
				pods:   []*v1.Pod{podWithClaimName},
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator:          resourceapi.DeviceTolerationOpExists,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Add(60 * time.Second)}}},
				evicting: []evictAt{{newObject(podWithClaimName), taintTime.Time.Add(60 * time.Second)}},
			},
			events: []any{
				remove(podWithClaimName),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator:          resourceapi.DeviceTolerationOpExists,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), evictionTime: &metav1.Time{Time: taintTime.Add(60 * time.Second)}}},
			},
		},
		"no-evict-wrong-resourceclaim": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaimOld), // pod not the owner
				add(podWithClaimTemplateInStatus),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaimOld, evictionTime: &taintTime}},
			},
		},
		"evict-wrong-resourceclaim-replaced": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaimOld, evictionTime: &taintTime}},
			},
			events: []any{
				update(inUseClaimOld, inUseClaim),
			},
			finalState: state{
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
		},
		"no-evict-resourceclaim-deallocated": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				// This removes the allocation while the pod is scheduled.
				// The normal situation for this is when the pod has terminated.
				// The abnormal one is a forced status update, which is similar
				// to a forced removal.
				//
				// It is debatable whether controller should try to "fix" the
				// cluster in those abnormal situations by evicting the pod.
				// Currently it doesn't because that is not it's job and
				// could cause problems by itself if not done carefully,
				// like killing a pod that still can run.
				update(inUseClaim, claim),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
			},
			wantEvents: []*v1.Event{cancelPodEviction},
		},
		"no-evict-resourceclaim-deleted": {
			initialState: state{
				pods:            []*v1.Pod{podWithClaimTemplateInStatus},
				slices:          []*resourceapi.ResourceSlice{sliceTainted, slice2},
				allocatedClaims: []allocatedClaim{{ResourceClaim: inUseClaim, evictionTime: &taintTime}},
				evicting:        []evictAt{{newObject(podWithClaimTemplateInStatus), taintTime.Time}},
			},
			events: []any{
				// Same as for "no-evict-resourceclaim-deallocated" this can be normal
				// (pod has terminated) and abnormal (force-delete).
				remove(inUseClaim),
			},
			finalState: state{
				slices: []*resourceapi.ResourceSlice{sliceTainted, slice2},
			},
			wantEvents: []*v1.Event{cancelPodEviction},
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
		tContext.ExpectNoError(store.Add(pod))
	}

	for _, event := range tc.events {
		switch event := event.(type) {
		case []any:
			for _, event := range event {
				applyEventPair(tContext, event)
			}
		default:
			applyEventPair(tContext, event)
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

func applyEventPair(tContext *testContext, event any) {
	store := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()

	switch pair := event.(type) {
	case [2]*resourceapi.ResourceSlice:
		tContext.handleSliceChange(pair[0], pair[1])
	case [2]*resourceapi.ResourceClaim:
		tContext.handleClaimChange(pair[0], pair[1])
	case [2]*v1.Pod:
		switch {
		case pair[0] != nil && pair[1] != nil:
			tContext.ExpectNoError(store.Update(pair[1]))
		case pair[0] != nil:
			tContext.ExpectNoError(store.Delete(pair[0]))
		default:
			tContext.ExpectNoError(store.Add(pair[1]))
		}
		tContext.handlePodChange(pair[0], pair[1])
	default:
		tContext.Fatalf("unexpected event type %T", event)
	}
}

func startTestController(tCtx ktesting.TContext, informerFactory informers.SharedInformerFactory) *Controller {
	controller := New(tCtx.Client(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1beta1().ResourceClaims(),
		informerFactory.Resource().V1beta1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		informerFactory.Resource().V1beta1().DeviceClasses(),
		"device-taint-eviction",
	)
	controller.metrics = metrics.New(300 /* one large initial bucket for testing */)
	// Always log, not matter what the -v value is.
	logger := klog.FromContext(tCtx)
	controller.eventLogger = &logger
	informerFactory.Start(tCtx.Done())
	return controller
}

func testPodDeletionsMetrics(controller *Controller, total int) error {
	// We cannot predict the sum of the latencies. Therefore we leave it at zero here
	// and replace expected (>= 0) values when gathering.
	expectedMetric := fmt.Sprintf(`# HELP device_taint_eviction_controller_pod_deletion_duration_seconds [ALPHA] Latency, in seconds, between the time when a device taint effect has been activated and a Pod's deletion via DeviceTaintEvictionController.
# TYPE device_taint_eviction_controller_pod_deletion_duration_seconds histogram
device_taint_eviction_controller_pod_deletion_duration_seconds_bucket{le="300"} %[1]d
device_taint_eviction_controller_pod_deletion_duration_seconds_bucket{le="+Inf"} %[1]d
device_taint_eviction_controller_pod_deletion_duration_seconds_sum 0
device_taint_eviction_controller_pod_deletion_duration_seconds_count %[1]d
# HELP device_taint_eviction_controller_pod_deletions_total [ALPHA] Total number of Pods deleted by DeviceTaintEvictionController since its start.
# TYPE device_taint_eviction_controller_pod_deletions_total counter
device_taint_eviction_controller_pod_deletions_total %[1]d
`, total)
	names := []string{
		controller.metrics.PodDeletionsTotal.FQName(),
		controller.metrics.PodDeletionsLatency.FQName(),
	}
	gather := func() ([]*metricstestutil.MetricFamily, error) {
		got, err := controller.metrics.Gather()
		for _, mf := range got {
			for _, m := range mf.Metric {
				if m.Histogram == nil || m.Histogram.SampleSum == nil || *m.Histogram.SampleSum < 0 {
					continue
				}
				m.Histogram.SampleSum = ptr.To(float64(0))
			}
		}
		return got, err
	}

	return metricstestutil.GatherAndCompare(metricstestutil.GathererFunc(gather), strings.NewReader(expectedMetric), names...)
}

// TestEviction runs through the full flow of starting the controller and evicting one pod.
// This scenario is the same as "evict-pod-resourceclaim" above. It also covers all
// event handlers by leading to the same end state through several different combinations
// of initial objects and add/update/delete calls.
func TestEviction(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	pod := podWithClaimName.DeepCopy()
	for name, tt := range map[string]struct {
		initialObjects []runtime.Object
		afterSync      func(tCtx ktesting.TContext)
	}{
		"initial": {
			initialObjects: []runtime.Object{
				sliceTainted,
				inUseClaim,
				pod,
			},
		},
		"add": {
			afterSync: func(tCtx ktesting.TContext) {
				var err error
				_, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
				require.NoError(tCtx, err, "create pod")
				_, err = tCtx.Client().ResourceV1beta1().ResourceSlices().Create(tCtx, sliceTainted, metav1.CreateOptions{})
				require.NoError(tCtx, err, "create slice")
				_, err = tCtx.Client().ResourceV1beta1().ResourceClaims(inUseClaim.Namespace).Create(tCtx, inUseClaim, metav1.CreateOptions{})
				require.NoError(tCtx, err, "create claim")
			},
		},
		"update": {
			initialObjects: []runtime.Object{
				slice,
				claim,
				func() *v1.Pod {
					pod := pod.DeepCopy()
					pod.Spec.NodeName = ""
					return pod
				}(),
			},
			afterSync: func(tCtx ktesting.TContext) {
				var err error
				_, err = tCtx.Client().CoreV1().Pods(pod.Namespace).Update(tCtx, pod, metav1.UpdateOptions{})
				require.NoError(tCtx, err, "update pod")
				_, err = tCtx.Client().ResourceV1beta1().ResourceSlices().Update(tCtx, sliceTainted, metav1.UpdateOptions{})
				require.NoError(tCtx, err, "update slice")
				_, err = tCtx.Client().ResourceV1beta1().ResourceClaims(inUseClaim.Namespace).UpdateStatus(tCtx, inUseClaim, metav1.UpdateOptions{})
				require.NoError(tCtx, err, "update claim")
			},
		},
		"delete": {
			initialObjects: []runtime.Object{
				sliceTainted,
				func() *resourceapi.ResourceSlice {
					// This has a higher generation and thus "shadows" sliceTainted until it gets removed.
					slice := slice.DeepCopy()
					slice.Name += "-other"
					slice.Spec.Pool.Generation += 100
					return slice
				}(),
				claim,
				pod,
			},
			afterSync: func(tCtx ktesting.TContext) {
				var err error

				err = tCtx.Client().ResourceV1beta1().ResourceSlices().Delete(tCtx, slice.Name+"-other", metav1.DeleteOptions{})
				require.NoError(tCtx, err, "delete slice")
				err = tCtx.Client().ResourceV1beta1().ResourceClaims(inUseClaim.Namespace).Delete(tCtx, inUseClaim.Name, metav1.DeleteOptions{})
				require.NoError(tCtx, err, "delete claim")

				// Re-create after deletion to enabled the normal flow.
				_, err = tCtx.Client().ResourceV1beta1().ResourceClaims(inUseClaim.Namespace).Create(tCtx, inUseClaim, metav1.CreateOptions{})
				require.NoError(tCtx, err, "create claim")
			},
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			tCtx.Parallel()
			fakeClientset := fake.NewSimpleClientset(tt.initialObjects...)
			tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

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
			controller := startTestController(tCtx, informerFactory)
			defer informerFactory.Shutdown()

			var wg sync.WaitGroup
			defer func() {
				tCtx.Log("Waiting for goroutine termination...")
				tCtx.Cancel("time to stop")
				wg.Wait()
			}()
			wg.Add(1)
			go func() {
				defer wg.Done()
				assert.NoError(tCtx, controller.Run(tCtx), "eviction controller failed")
			}()

			// Eventually the controller should have synced it's informers.
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
				return controller.hasSynced.Load() > 0
			}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("controller synced"))
			if tt.afterSync != nil {
				tt.afterSync(tCtx)
			}

			// Eventually the pod gets deleted (= evicted).
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
				_, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
				return apierrors.IsNotFound(err)
			}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("pod evicted"))

			pod := pod.DeepCopy()
			pod.Status.Conditions = []v1.PodCondition{{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByDeviceTaintManager",
				Message: "Device Taint manager: deleting due to NoExecute taint",
			}}
			if diff := cmp.Diff(pod, updatedPod, cmpopts.IgnoreTypes(metav1.Time{})); diff != "" {
				tCtx.Errorf("unexpected modified pod (-want, +got):\n%s", diff)
			}

			// Shortly after deletion we should also see updated metrics.
			// This is the last thing the controller does for a pod.
			// However, actually creating the event on the server is asynchronous,
			// so we also have to wait for that.
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
				gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
				return testPodDeletionsMetrics(controller, 1)
			}).WithTimeout(30*time.Second).Should(gomega.Succeed(), "pod eviction done")

			// We also don't want any other events, in particular not a cancellation event
			// because the pod deletion was observed or another occurrence of the same event.
			ktesting.Consistently(tCtx, func(tCtx ktesting.TContext) error {
				mutex.Lock()
				defer mutex.Unlock()
				assert.Equal(tCtx, 1, podUpdates, "number of pod update calls")
				assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
				gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
				tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 1))
				return nil
			}).WithTimeout(5 * time.Second).Should(gomega.Succeed())
		})
	}
}

// TestCancelEviction deletes the pod before the controller deletes it
// or removes the slice. Either way, eviction gets cancelled.
func TestCancelEviction(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Parallel()

	tCtx.Run("pod-deleted", func(tCtx ktesting.TContext) { testCancelEviction(tCtx, true) })
	tCtx.Run("slice-deleted", func(tCtx ktesting.TContext) { testCancelEviction(tCtx, false) })
}

func testCancelEviction(tCtx ktesting.TContext, deletePod bool) {
	// The claim tolerates the taint long enough for us to
	// do something which cancels eviction.
	pod := podWithClaimName.DeepCopy()
	slice := sliceTainted.DeepCopy()
	slice.Spec.Devices[0].Basic.Taints[0].TimeAdded = &metav1.Time{Time: time.Now()}
	claim := inUseClaim.DeepCopy()
	claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
		Operator:          resourceapi.DeviceTolerationOpExists,
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
	controller := startTestController(tCtx, informerFactory)
	defer informerFactory.Shutdown()

	var mutex sync.Mutex
	podEvicting := false
	controller.evictPod = func(podRef tainteviction.NamespacedObject, fireAt time.Time) {
		assert.Equal(tCtx, newObject(pod), podRef)
		mutex.Lock()
		defer mutex.Unlock()
		podEvicting = true
	}
	controller.cancelEvict = func(podRef tainteviction.NamespacedObject) bool {
		assert.Equal(tCtx, newObject(pod), podRef)
		mutex.Lock()
		defer mutex.Unlock()
		podEvicting = false
		return false
	}

	var wg sync.WaitGroup
	defer func() {
		tCtx.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		assert.NoError(tCtx, controller.Run(tCtx), "eviction controller failed")
	}()

	// Eventually the pod gets scheduled for eviction.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podEvicting
	}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("pod pending eviction"))

	// Now we can delete the pod or slice.
	if deletePod {
		tCtx.ExpectNoError(fakeClientset.CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{}))
	} else {
		tCtx.ExpectNoError(fakeClientset.ResourceV1beta1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
	}

	// Shortly after deletion we should also see the cancellation.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podEvicting
	}).WithTimeout(30 * time.Second).Should(gomega.BeFalseBecause("pod no longer pending eviction"))

	// Whether we get an event depends on whether the pod still exists.
	ktesting.Consistently(tCtx, func(tCtx ktesting.TContext) error {
		matchEvents := matchCancellationEvent()
		if deletePod {
			matchEvents = gomega.BeEmpty()
		}
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchEvents)
		tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 0))
		return nil
	}).WithTimeout(5 * time.Second).Should(gomega.Succeed())
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
		assert.NoError(tCtx, err, "delete pod") //nolint:testifylint // Here recording an unknown error and continuing is okay.
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
	controller := startTestController(tCtx, informerFactory)
	defer informerFactory.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		tCtx.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		assert.NoError(tCtx, controller.Run(tCtx), "eviction controller failed")
	}()

	// Eventually the pod gets deleted, in this test by us.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podGets >= 1
	}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("pod eviction started"))

	// We don't want any events.
	ktesting.Consistently(tCtx, func(tCtx ktesting.TContext) error {
		mutex.Lock()
		defer mutex.Unlock()
		assert.Equal(tCtx, 1, podGets, "number of pod get calls")
		assert.Equal(tCtx, 0, podDeletions, "number of pod delete calls")
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(gomega.BeEmpty())
		tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 0))
		return nil
	}).WithTimeout(5 * time.Second).Should(gomega.Succeed())
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
	controller := startTestController(tCtx, informerFactory)
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
		assert.NoError(tCtx, controller.Run(tCtx), "eviction controller failed")
	}()

	// Eventually the pod gets deleted and the event is recorded.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
		return testPodDeletionsMetrics(controller, 1)
	}).WithTimeout(30*time.Second).Should(gomega.Succeed(), "pod eviction done")

	// Now we can check the API calls.
	ktesting.Consistently(tCtx, func(tCtx ktesting.TContext) error {
		mutex.Lock()
		defer mutex.Unlock()
		assert.Equal(tCtx, 2, podGets, "number of pod get calls")
		assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
		tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 1))
		return nil
	}).WithTimeout(5 * time.Second).Should(gomega.Succeed())
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
	controller := startTestController(tCtx, informerFactory)
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
		assert.NoError(tCtx, controller.Run(tCtx), "eviction controller failed")
	}()

	// Eventually deletion is attempted a few times.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) int {
		mutex.Lock()
		defer mutex.Unlock()
		return podDeletions
	}).WithTimeout(30*time.Second).Should(gomega.BeNumerically(">=", retries), "pod eviction failed")

	// Now we can check the API calls.
	ktesting.Consistently(tCtx, func(tCtx ktesting.TContext) error {
		mutex.Lock()
		defer mutex.Unlock()
		assert.Equal(tCtx, retries, podGets, "number of pod get calls")
		assert.Equal(tCtx, retries, podDeletions, "number of pod delete calls")
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
		tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 0))
		return nil
	}).WithTimeout(5 * time.Second).Should(gomega.Succeed())
}

// BenchTaintUntaint checks the full flow of detecting a claim as
// tainted because of a new DeviceTaintRule, starting to evict its
// consumer, and then undoing that when the DeviceTaintRule is removed.
func BenchmarkTaintUntaint(b *testing.B) {
	tContext := setup(b)
	podStore := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()
	// No output, comment out if output is desired.
	tContext.Controller.eventLogger = nil

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Add objects...
		tContext.handleSliceChange(nil, slice)
		tContext.handleClaimChange(nil, inUseClaimWithToleration)
		require.NoError(tContext, podStore.Add(podWithClaimName), "add pod")
		tContext.handlePodChange(nil, podWithClaimName)
		require.Empty(tContext, tContext.evicting)

		// Now evict.
		tContext.handleSliceChange(slice, sliceTainted)

		// Because informer event handlers are synchronous, we get the expected result immediately.
		require.NotEmpty(tContext, tContext.evicting)

		// ... and remove them again.
		tContext.handleSliceChange(sliceTainted, slice)
		require.Empty(tContext, tContext.evicting)

		tContext.handlePodChange(podWithClaimName, nil)
		require.NoError(tContext, podStore.Delete(podWithClaimName), "remove pod")
		tContext.handleClaimChange(inUseClaimWithToleration, nil)
		tContext.handleSliceChange(slice, nil)
	}
}
