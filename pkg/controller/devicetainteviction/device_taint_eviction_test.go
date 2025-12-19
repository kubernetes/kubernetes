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
	"maps"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
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
	resourceapi "k8s.io/api/resource/v1"
	resourcealpha "k8s.io/api/resource/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction/metrics"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	controllertestutil "k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// Reduce typing with some constructors.

func metav1Time(time time.Time) *metav1.Time {
	return &metav1.Time{Time: time}
}

func ac(claim *resourceapi.ResourceClaim, eviction ...*evictionAndReason) allocatedClaim {
	ac := allocatedClaim{
		ResourceClaim: claim,
	}
	switch len(eviction) {
	case 1:
		ac.eviction = eviction[0]
	case 0:
	default:
		panic("wrong number of evictions")
	}
	return ac
}

func l[T any](items ...T) []T {
	return items
}

// setup creates a controller which is ready to have its handle* methods called.
func setup(tCtx ktesting.TContext) *testContext {
	featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate,
		featuregatetesting.FeatureOverrides{
			features.DRADeviceTaints:     true,
			features.DRADeviceTaintRules: true,
		},
	)

	fakeClientset := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(fakeClientset, 0)
	controller := New(fakeClientset,
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		informerFactory.Resource().V1().DeviceClasses(),
		"device-taint-eviction",
	)
	tContext := &testContext{
		TContext:        tCtx,
		Controller:      controller,
		recorder:        controllertestutil.NewFakeRecorder(),
		client:          fakeClientset,
		informerFactory: informerFactory,
	}
	controller.workqueue = &tContext.mockQueue
	tContext.logger = tCtx.Logger()
	// Always log, not matter what the -v value is.
	controller.eventLogger = &tContext.logger
	tContext.Controller.recorder = tContext.recorder

	return tContext
}

type testContext struct {
	ktesting.TContext
	*Controller
	mockQueue       Mock[workItem]
	recorder        *controllertestutil.FakeRecorder
	client          *fake.Clientset
	informerFactory informers.SharedInformerFactory
}

type state struct {
	pods            []*v1.Pod
	allocatedClaims []allocatedClaim
	slices          []*resourceapi.ResourceSlice
	rules           []*resourcealpha.DeviceTaintRule
	ruleStats       map[types.UID]taintRuleStats

	// Pods might have been queued in the past and then not removed when removing from deletePodAt.
	// Therefore we need to describe the expected state separately for both fields.

	deletePodAt evictMap
	queued      MockState[workItem]
}

// step describes a state after handling ready work items and how much to move time forward.
type step struct {
	pods        []*v1.Pod
	rules       []*resourcealpha.DeviceTaintRule
	ruleStats   map[types.UID]taintRuleStats
	deletePodAt evictMap

	// queuedProcessed is the state after handling all ready items.
	queuedProcessed MockState[workItem]
	// Move time forward.
	advance time.Duration
	// queuedShifted is the state after reducing the delay for future items.
	queuedShifted MockState[workItem]
}

type evictMap map[tainteviction.NamespacedObject]evictionAndReason

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
			pool.slices = make(map[string]*resourceapi.ResourceSlice)
		}
		pool.slices[slice.Name] = slice
		pools[id] = pool
	}
	for id, pool := range pools {
		maxGeneration := int64(math.MinInt64)
		for _, slice := range pool.slices {
			if slice.Spec.Pool.Generation > maxGeneration {
				maxGeneration = slice.Spec.Pool.Generation
			}
		}
		pool.maxGeneration = maxGeneration
		pools[id] = pool
	}
	return pools
}

// assertEqual uses cmp.Diff instead of spew+diff like testify's Equal.
// This provides a full context for a modified field.
// Allows comparing unexported fields, empty and nil maps/slices are equal.
func assertEqual[T any](t interface {
	Helper()
	Errorf(string, ...any)
}, expected, actual T, what string, opts ...cmp.Option) bool {
	t.Helper()

	opts = append(opts, cmp.Exporter(func(reflect.Type) bool { return true }), cmpopts.EquateEmpty())

	if diff := cmp.Diff(expected, actual, opts...); diff != "" {
		t.Errorf("Unexpected diff for %s (-expected, +actual):\n%s", what, diff)
		return false
	}
	return true
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

	// finalState represents the result of feeding the event handlers.
	finalState state

	// process represents the result of processing all of the ready work queue items,
	// potentially multiple times. After each flush the time advances and pending
	// work items become ready. The default is []step{{}}.
	process []step

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
	// taintTime is the start time of a synctest bubble.
	// All tests run inside such a bubble and thus have a deterministic
	// delta between this taint time and their current clock.
	taintTime  = metav1Time(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC))
	taintKey   = "example.com/taint"
	taintValue = "something"

	// All slices use the internal format.
	// For client-go they get converted back to the v1 API.

	simpleSlice = st.MakeResourceSlice(nodeName, driver).
			Device("instance").
			Obj()
	slice = st.MakeResourceSlice(nodeName, driver).
		Device("instance").
		Device("instance-no-schedule", resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoSchedule}).
		Device("instance-no-execute", resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNone, TimeAdded: taintTime}, resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: taintTime}).
		Obj()
	sliceReplaced = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Name += "-updated"
		slice.Spec.Pool.Generation++
		return slice
	}()
	sliceOtherDevices = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		for i := range slice.Spec.Devices {
			slice.Spec.Devices[i].Name += "-other"
		}
		return slice
	}()
	taint        = &resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, Value: taintValue, TimeAdded: taintTime}
	sliceTainted = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[0].Taints = []resourceapi.DeviceTaint{*taint}
		return slice
	}()
	sliceTaintedExtended = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		slice.Spec.Devices = append(slice.Spec.Devices, *slice.Spec.Devices[0].DeepCopy())
		slice.Spec.Devices[len(slice.Spec.Devices)-1].Name += "-other"
		return slice
	}()
	sliceTaintedNone = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		for i := range slice.Spec.Devices {
			for j := range slice.Spec.Devices[i].Taints {
				slice.Spec.Devices[i].Taints[j].Effect = resourceapi.DeviceTaintEffectNone
			}
		}
		return slice
	}()
	sliceTaintedUnknown = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		for i := range slice.Spec.Devices {
			for j := range slice.Spec.Devices[i].Taints {
				slice.Spec.Devices[i].Taints[j].Effect = resourceapi.DeviceTaintEffect("unknown-effect")
			}
		}
		return slice
	}()
	sliceTaintedNoSchedule = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		for i := range slice.Spec.Devices {
			for j := range slice.Spec.Devices[i].Taints {
				slice.Spec.Devices[i].Taints[j].Effect = resourceapi.DeviceTaintEffectNoSchedule
			}
		}
		return slice
	}()
	taintOther             = &resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, Value: taintValue + "-other", TimeAdded: taintTime}
	sliceTaintedValueOther = func() *resourceapi.ResourceSlice {
		slice := sliceTainted.DeepCopy()
		slice.Spec.Devices[0].Taints[0] = *taintOther
		return slice
	}()
	taint1            = &resourceapi.DeviceTaint{Key: taintKey, Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: taintTime}
	taint2            = &resourceapi.DeviceTaint{Key: taintKey + "-other", Effect: resourceapi.DeviceTaintEffectNoExecute, TimeAdded: taintTime}
	sliceTaintedTwice = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[0].Taints = []resourceapi.DeviceTaint{*taint1, *taint2}
		return slice
	}()
	sliceUntainted = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Spec.Devices[1].Taints = nil
		slice.Spec.Devices[2].Taints = nil
		return slice
	}()
	slice2 = func() *resourceapi.ResourceSlice {
		slice := slice.DeepCopy()
		slice.Name += "-2"
		slice.Spec.NodeName = &nodeName2
		slice.Spec.Pool.Name = nodeName2
		slice.Spec.Pool.Generation++
		return slice
	}()
	ruleEvict = &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "evict",
			UID:  "1234",
		},

		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Driver: ptr.To(driver),
			},
			Taint: resourcealpha.DeviceTaint{
				Key:       taint.Key,
				Value:     taint.Value,
				Effect:    resourcealpha.DeviceTaintEffect(taint.Effect),
				TimeAdded: taint.TimeAdded,
			},
		},
	}
	ruleEvictInstance1 = &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "evict-instance",
			UID:  "1234",
		},

		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Driver: ptr.To(driver),
				Device: ptr.To("instance"),
			},
			Taint: resourcealpha.DeviceTaint{
				Key:       taint.Key,
				Value:     taint.Value,
				Effect:    resourcealpha.DeviceTaintEffect(taint.Effect),
				TimeAdded: taintTime,
			},
		},
	}
	taintTimeLater          = metav1Time(taintTime.Add(40 * time.Second))
	ruleEvictInstance2Later = &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "evict-instance-no-execute",
			UID:  "5678",
		},

		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Driver: ptr.To(driver),
				Device: ptr.To("instance-no-execute"),
			},
			Taint: resourcealpha.DeviceTaint{
				Key:       taint.Key,
				Value:     taint.Value,
				Effect:    resourcealpha.DeviceTaintEffect(taint.Effect),
				TimeAdded: taintTimeLater,
			},
		},
	}
	ruleNone = &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "evict",
			UID:  "1234",
		},

		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Driver: ptr.To(driver),
			},
			Taint: resourcealpha.DeviceTaint{
				Key:       taint.Key,
				Value:     taint.Value,
				Effect:    resourcealpha.DeviceTaintEffectNone,
				TimeAdded: taint.TimeAdded,
			},
		},
	}
	ruleEvictOther = &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "evict-other",
			UID:  "1234-other",
		},

		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Device: ptr.To("instance"),
			},
			Taint: resourcealpha.DeviceTaint{
				Key:       taint.Key,
				Value:     taint.Value,
				Effect:    resourcealpha.DeviceTaintEffect(taint.Effect),
				TimeAdded: taint.TimeAdded,
			},
		},
	}
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
	allocationResultOtherDevices = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-no-schedule",
					Request: "req-1",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-no-execute",
					Request: "req-1",
				},
			},
		},
	}
	inUseClaim = st.FromResourceClaim(claim).
			OwnerReference(podName, podUID, podKind).
			Allocation(allocationResult).
			ReservedForPod(podName, types.UID(podUID)).
			Obj()
	inUseClaimOtherNamespace = st.FromResourceClaim(claim).
					Namespace(namespace+"-other").
					OwnerReference(podName, podUID+"-2", podKind). // podWithClaimNameOtherNamespace below.
					Allocation(allocationResultOtherDevices).
					ReservedForPod(podName, types.UID(podUID+"-2")).
					Obj()
	inUseClaimOtherName = st.FromResourceClaim(claim).
				Name(claimName+"-other").
				OwnerReference(podName+"-other", podUID+"-3", podKind). // podWithClaimNameOtherName below.
				Allocation(allocationResultOtherDevices).
				ReservedForPod(podName+"-other", types.UID(podUID+"-3")).
				Obj()
	inUseClaimOtherNameShared = st.FromResourceClaim(claim).
					Name(claimName+"-other").
					OwnerReference(podName+"-other", podUID+"-3", podKind). // podWithClaimNameOtherName below.
					Allocation(allocationResultOtherDevices).
					ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}, resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName + "-other", UID: types.UID(podUID + "-3")}).
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
	unscheduledPodWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
					UID(podUID).
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
					Obj()
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Node(nodeName).
				Obj()
	podWithTwoClaimNames = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &inUseClaim.Name}).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName + "-other", ResourceClaimName: &inUseClaimOtherName.Name}).
				Node(nodeName).
				Obj()
	podWithClaimNameOtherUID = st.MakePod().Name(podName).Namespace(namespace).
					UID(podUID + "-other").
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
					Node(nodeName).
					Obj()
	podWithClaimNameOtherNamespace = st.MakePod().Name(podName).Namespace(namespace + "-other").
					UID(podUID + "-2").
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
					Node(nodeName).
					Obj()
	podWithClaimNameOtherName = st.MakePod().Name(podName + "-other").Namespace(namespace).
					UID(podUID + "-3").
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &inUseClaimOtherName.Name}).
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
	deletePodEvent = &v1.Event{
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Namespace:  namespace,
			Name:       podName,
			UID:        types.UID(podUID),
		},
		Reason:  "DeviceTaintManagerEviction",
		Message: "Marking for deletion",
		Type:    v1.EventTypeNormal,
		Source: v1.EventSource{
			Component: "nodeControllerTest",
		},
		Count: 1,
	}
	deletePodEventOtherNamespace = &v1.Event{
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Namespace:  namespace + "-other",
			Name:       podName,
			UID:        types.UID(podUID + "-2"),
		},
		Reason:  "DeviceTaintManagerEviction",
		Message: "Marking for deletion",
		Type:    v1.EventTypeNormal,
		Source: v1.EventSource{
			Component: "nodeControllerTest",
		},
		Count: 1,
	}
	deletePodEventOtherName = &v1.Event{
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Namespace:  namespace,
			Name:       podName + "-other",
			UID:        types.UID(podUID + "-3"),
		},
		Reason:  "DeviceTaintManagerEviction",
		Message: "Marking for deletion",
		Type:    v1.EventTypeNormal,
		Source: v1.EventSource{
			Component: "nodeControllerTest",
		},
		Count: 1,
	}
)

func newEvictionTime(when *metav1.Time, args ...any) *evictionAndReason {
	if when == nil {
		return nil
	}

	var reason []trackedTaint
	i := 0
	for i < len(args) {
		switch obj := args[i].(type) {
		case *resourceapi.ResourceSlice:
			reason = append(reason, trackedTaint{slice: sliceDeviceTaint{slice: obj, deviceName: args[i+1].(string), taintIndex: args[i+2].(int)}})
			i += 3
		case *resourcealpha.DeviceTaintRule:
			reason = append(reason, trackedTaint{rule: obj})
			i++
		default:
			panic(fmt.Sprintf("unsupported argument type %T", args[i]))
		}
	}

	return &evictionAndReason{
		when:   *when,
		reason: reason,
	}
}

func newWorkItems(objs ...metav1.Object) []workItem {
	var items []workItem
	for _, obj := range objs {
		items = append(items, newWorkItem(obj))
	}
	return items
}

func newWorkItem(obj metav1.Object) workItem {
	ref := newObject(obj)
	var item workItem
	switch obj.(type) {
	case *resourcealpha.DeviceTaintRule:
		item.ruleRef = ref
	case *v1.Pod:
		item.podRef = ref
	default:
		panic(fmt.Sprintf("invalid type %T", obj))
	}
	return item
}

func newDelayedWorkItems(objsAndDelay ...any) []MockDelayedItem[workItem] {
	var items []MockDelayedItem[workItem]
	for i := 0; i < len(objsAndDelay); i += 2 {
		item := newWorkItem(objsAndDelay[i].(metav1.Object))
		delay := objsAndDelay[i+1].(time.Duration)
		items = append(items, MockDelayedItem[workItem]{item, delay})
	}
	return items
}

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

func inProgress(rule *resourcealpha.DeviceTaintRule, status bool, reason, message string, when *metav1.Time) *resourcealpha.DeviceTaintRule {
	rule = rule.DeepCopy()
	condition := metav1.Condition{
		Type:               resourcealpha.DeviceTaintConditionEvictionInProgress,
		Status:             metav1.ConditionFalse,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: rule.Generation,
		LastTransitionTime: *when,
	}
	if status {
		condition.Status = metav1.ConditionTrue
	}
	rule.Status.Conditions = []metav1.Condition{condition}
	return rule
}

// TestController covers the event handler logic and handling work.
//
// It runs inside a synctest bubble without actually starting the
// controller. Each test case starts with
// some known state, applies a sequence of independent events in all
// permutations of their order, then checks against the expected final
// state. The final state must be the same in all permutations. This simulates
// the random order in which informer updates can be perceived.
//
// Then pending work gets handled, potentially multiple times after
// advancing time to reach "later" work items.
func TestController(t *testing.T) { testController(ktesting.Init(t)) }
func testController(tCtx ktesting.TContext) {
	for name, tc := range map[string]testCase{
		"empty": {},
		"populate-pools": {
			events: []any{
				add(slice),
				add(slice2),
			},
			finalState: state{
				slices: l(slice, slice2),
			},
		},
		"update-pools": {
			initialState: state{
				slices: l(slice, slice2),
			},
			events: []any{
				update(slice, sliceUntainted),
				remove(slice2),
			},
			finalState: state{
				slices: l(sliceUntainted),
			},
		},
		"untainted-claim": {
			events: []any{
				add(slice),
				add(slice2),
				add(inUseClaim),
			},
			finalState: state{
				slices:          l(slice, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
		},
		"tainted-claim-through-resourceslice": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
		},
		"rule-status": {
			events: []any{
				add(ruleEvict),
			},
			finalState: state{
				queued: MockState[workItem]{Ready: newWorkItems(ruleEvict)},
			},
			process: []step{{
				rules: l(inProgress(ruleEvict, false, "NotStarted", "", taintTime)),
			}},
		},
		"tainted-claim-through-rule": {
			events: []any{
				add(ruleEvict),
				add(inUseClaim),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvict))),
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict)},
			},
			process: []step{{
				rules: l(inProgress(ruleEvict, false, "NotStarted", "", taintTime)),
			}},
		},
		"evict-pod-resourceclaim": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-rule": {
			events: []any{
				add(inUseClaim),
				add(podWithClaimName),
				add(ruleEvict),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvict))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, ruleEvict)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict, podWithClaimName)},
			},
			process: []step{
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					// Initial update.
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					// Final update.
					rules: l(inProgress(ruleEvict, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-rule-later": {
			events: []any{
				add(inUseClaimWithToleration),
				add(podWithClaimName),
				add(ruleEvict),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaimWithToleration, newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), ruleEvict))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), ruleEvict)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict), Later: newDelayedWorkItems(podWithClaimName, tolerationDuration)},
			},
			process: []step{
				{
					// Initial update.
					deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), ruleEvict)},
					pods:        l(podWithClaimName),
					rules:       l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),

					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, tolerationDuration)},
					advance:         tolerationDuration,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
				},
				{
					// Deleted, but condition not updated yet.
					ruleStats:       map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict)},
				},
				{
					// Final update.
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					rules:     l(inProgress(ruleEvict, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(tolerationDuration+ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-many-pods-many-namespaces": {
			events: []any{
				add(inUseClaim),
				add(podWithClaimName),
				add(inUseClaimOtherNamespace),
				add(podWithClaimNameOtherNamespace),
				add(ruleEvict),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvict)), ac(inUseClaimOtherNamespace, newEvictionTime(taintTime, ruleEvict))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, ruleEvict), newObject(podWithClaimNameOtherNamespace): *newEvictionTime(taintTime, ruleEvict)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict, podWithClaimName, podWithClaimNameOtherNamespace)},
			},
			process: []step{
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 2}},
					// Initial update.
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "2 pods need to be evicted in 2 different namespaces.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 2}},
					// Final update.
					rules: l(inProgress(ruleEvict, false, "Completed", "2 pods evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent, deletePodEventOtherNamespace),
		},
		"evict-many-pods-same-namespace": {
			events: []any{
				add(inUseClaim),
				add(podWithClaimName),
				add(inUseClaimOtherName),
				add(podWithClaimNameOtherName),
				add(ruleEvict),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvict)), ac(inUseClaimOtherName, newEvictionTime(taintTime, ruleEvict))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, ruleEvict), newObject(podWithClaimNameOtherName): *newEvictionTime(taintTime, ruleEvict)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict, podWithClaimName, podWithClaimNameOtherName)},
			},
			process: []step{
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 2}},
					// Initial update.
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "2 pods need to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 2}},
					// Final update.
					rules: l(inProgress(ruleEvict, false, "Completed", "2 pods evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent, deletePodEventOtherName),
		},
		"none-pod-rule": {
			events: []any{
				add(slice),
				add(inUseClaim),
				add(podWithClaimName),
				add(ruleNone),
			},
			finalState: state{
				slices:          l(slice),
				allocatedClaims: l(ac(inUseClaim)),
				queued:          MockState[workItem]{Ready: newWorkItems(ruleNone)},
			},
			process: []step{
				{
					pods:  l(podWithClaimName),
					rules: l(inProgress(ruleNone, false, "NoEffect", "3 published devices selected. 1 allocated device selected. 1 pod would be evicted in 1 namespace if the effect was NoExecute. This information will not be updated again. Recreate the DeviceTaintRule to trigger an update.", taintTime)),
				},
			},
		},
		"none-many-pods-many-namespaces": {
			events: []any{
				add(inUseClaim),
				add(podWithClaimName),
				add(inUseClaimOtherNamespace),
				add(podWithClaimNameOtherNamespace),
				add(ruleNone),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim), ac(inUseClaimOtherNamespace)),
				queued:          MockState[workItem]{Ready: newWorkItems(ruleNone)},
			},
			process: []step{
				{
					pods:  l(podWithClaimName, podWithClaimNameOtherNamespace),
					rules: l(inProgress(ruleNone, false, "NoEffect", "0 published devices selected. 3 allocated devices selected. 2 pods would be evicted in 2 different namespaces if the effect was NoExecute. This information will not be updated again. Recreate the DeviceTaintRule to trigger an update.", taintTime)),
				},
			},
		},
		"none-many-pods-same-namespace": {
			events: []any{
				add(inUseClaim),
				add(podWithClaimName),
				add(inUseClaimOtherName),
				add(podWithClaimNameOtherName),
				add(ruleNone),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim), ac(inUseClaimOtherName)),
				queued:          MockState[workItem]{Ready: newWorkItems(ruleNone)},
			},
			process: []step{
				{
					pods:  l(podWithClaimName, podWithClaimNameOtherName),
					rules: l(inProgress(ruleNone, false, "NoEffect", "0 published devices selected. 3 allocated devices selected. 2 pods would be evicted in 1 namespace if the effect was NoExecute. This information will not be updated again. Recreate the DeviceTaintRule to trigger an update.", taintTime)),
				},
			},
		},
		"multiple-claims-same-rule": {
			events: []any{
				add(inUseClaim),
				add(inUseClaimOtherNameShared),
				add(podWithTwoClaimNames),
				add(ruleEvict),
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvict)), ac(inUseClaimOtherNameShared, newEvictionTime(taintTime, ruleEvict))),
				deletePodAt:     evictMap{newObject(podWithTwoClaimNames): *newEvictionTime(taintTime, ruleEvict)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvict, podWithTwoClaimNames)},
			},
			process: []step{
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					// Initial update.
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}},
					// Final update.
					rules: l(inProgress(ruleEvict, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent),
		},
		"multiple-claims-different-rules-order-1": {
			// Different rules cause eviction at different times because the second one gets added a bit in the future.
			// It's debatable whether eviction of the pod should then be attributed to both rules. That is
			// how it is currently implemented, based on the rationale that the second rule would have caused
			// eviction eventually if it just had been given more time.
			//
			// The order matters here: the work queue gets populate differently depending on what is observed first,
			// see next case. The `queued` state is compared without considering the order, so the actual
			// order of the queue is not visible in the tests. Typically it doesn't matter, but here it does.
			events: []any{
				[]any{
					add(inUseClaim),
					add(inUseClaimOtherNameShared),
					add(podWithTwoClaimNames),
					add(ruleEvictInstance1),
					add(ruleEvictInstance2Later),
				},
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvictInstance1)), ac(inUseClaimOtherNameShared, newEvictionTime(taintTimeLater, ruleEvictInstance2Later))),
				deletePodAt:     evictMap{newObject(podWithTwoClaimNames): *newEvictionTime(taintTime, ruleEvictInstance1, ruleEvictInstance2Later)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvictInstance1, ruleEvictInstance2Later, podWithTwoClaimNames)},
			},
			process: []step{
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvictInstance1.UID: {numEvictedPods: 1}, ruleEvictInstance2Later.UID: {numEvictedPods: 1}},
					// Initial update of ruleEvictInstance1 before eviction, then update of ruleEvictInstance2Later.
					rules:           l(inProgress(ruleEvictInstance1, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime), inProgress(ruleEvictInstance2Later, false, "Completed", "1 pod evicted since starting the controller.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvictInstance1, ruleStatusPeriod, ruleEvictInstance2Later, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvictInstance1, ruleEvictInstance2Later)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}, ruleEvictInstance2Later.UID: {numEvictedPods: 1}},
					// Final update.
					rules: l(inProgress(ruleEvictInstance1, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod))), inProgress(ruleEvictInstance2Later, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent),
		},
		"multiple-claims-different-rules-order-2": {
			events: []any{
				[]any{
					add(inUseClaim),
					add(inUseClaimOtherNameShared),
					add(podWithTwoClaimNames),
					add(ruleEvictInstance2Later), // Reversed, so now the pod gets scheduled for delayed eviction, which cannot get canceled.
					add(ruleEvictInstance1),
				},
			},
			finalState: state{
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, ruleEvictInstance1)), ac(inUseClaimOtherNameShared, newEvictionTime(taintTimeLater, ruleEvictInstance2Later))),
				deletePodAt:     evictMap{newObject(podWithTwoClaimNames): *newEvictionTime(taintTime, ruleEvictInstance1, ruleEvictInstance2Later)},
				queued:          MockState[workItem]{Ready: newWorkItems(ruleEvictInstance1, ruleEvictInstance2Later, podWithTwoClaimNames), Later: newDelayedWorkItems(podWithTwoClaimNames, ruleEvictInstance2Later.Spec.Taint.TimeAdded.Sub(taintTime.Time))},
			},
			process: []step{
				// The pod is scheduled for much later and time needs to advance a few times before it gets processed.
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvictInstance1.UID: {numEvictedPods: 1}, ruleEvictInstance2Later.UID: {numEvictedPods: 1}},
					// Initial update of both rules before eviction.
					rules:           l(inProgress(ruleEvictInstance1, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime), inProgress(ruleEvictInstance2Later, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithTwoClaimNames, ruleEvictInstance2Later.Spec.Taint.TimeAdded.Sub(taintTime.Time), ruleEvictInstance1, ruleStatusPeriod, ruleEvictInstance2Later, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvictInstance1, ruleEvictInstance2Later), Later: newDelayedWorkItems(podWithTwoClaimNames, ruleEvictInstance2Later.Spec.Taint.TimeAdded.Sub(taintTime.Time)-ruleStatusPeriod)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}, ruleEvictInstance2Later.UID: {numEvictedPods: 1}},
					// Final update.
					rules:           l(inProgress(ruleEvictInstance1, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod))), inProgress(ruleEvictInstance2Later, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithTwoClaimNames, ruleEvictInstance2Later.Spec.Taint.TimeAdded.Sub(taintTime.Time)-ruleStatusPeriod)},
					advance:         ruleEvictInstance2Later.Spec.Taint.TimeAdded.Sub(taintTime.Time) - ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(podWithTwoClaimNames)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}, ruleEvictInstance2Later.UID: {numEvictedPods: 1}},
					rules:     l(inProgress(ruleEvictInstance1, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod))), inProgress(ruleEvictInstance2Later, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(ruleStatusPeriod)))),
					// The pod gets removed from the work queue without doing anything.
				},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-resourceclaim-again": {
			initialState: state{
				pods:            l(podWithClaimName),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				[]any{remove(sliceTainted), add(sliceTainted)},
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			// It is debatable whether the controller should react
			// to slice changes (a deletion in this case)
			// quickly. On the one hand we want to cancel eviction
			// quickly in case that a taint goes away, on the other
			// hand it can also restore the previous state and emit
			// an event, as in this test case.
			//
			// At the moment, the code reliably cancels right away.
			wantEvents: l(cancelPodEviction, deletePodEvent),
		},
		"evict-pod-after-scheduling": {
			initialState: state{
				pods:            l(unscheduledPodWithClaimName),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
			events: []any{
				// Normally the scheduler shouldn't schedule when there is a taint,
				// but perhaps it didn't know yet.
				update(unscheduledPodWithClaimName, podWithClaimName),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-resourceclaim-unrelated-changes": {
			initialState: state{
				pods:            l(podWithClaimName),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				update(sliceTainted, sliceTaintedExtended),
				update(inUseClaim, inUseClaim),             // No real change here, good enough for testing some code paths.
				update(podWithClaimName, podWithClaimName), // Same here.
			},
			finalState: state{
				slices:          l(sliceTaintedExtended, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-resourceclaimtemplate": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplateInStatus),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimTemplateInStatus)},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-later": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaimWithToleration),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaimWithToleration, newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, tolerationDuration)},
			},
			process: []step{
				// First advance time, then delete.
				{
					deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
					pods:            l(podWithClaimName),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, tolerationDuration)},
					advance:         tolerationDuration,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
				},
				{},
			},
			wantEvents: l(deletePodEvent),
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
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
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
				}(), newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:      MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, 30*time.Second)},
			},
			process: []step{
				// First advance time, then delete.
				{
					deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
					pods:            l(podWithClaimName),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, 30*time.Second)},
					advance:         30 * time.Second,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
				},
				{},
			},
			wantEvents: l(deletePodEvent),
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
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Key:               taintKey + "-other",
						Operator:          resourceapi.DeviceTolerationOpEqual,
						Value:             taintValue,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:      MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
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
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}())),
			},
			process: []step{{pods: l(podWithClaimName)}},
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
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
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
				}())),
			},
			process: []step{{pods: l(podWithClaimName)}},
		},
		"evict-pod-partial-toleration": {
			events: []any{
				add(sliceTaintedTwice),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Key:      taint1.Key,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: l(sliceTaintedTwice, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator: resourceapi.DeviceTolerationOpExists,
						Key:      taint1.Key,
						Effect:   resourceapi.DeviceTaintEffectNoExecute,
					}}
					return claim
				}(), newEvictionTime(taintTime, sliceTaintedTwice, sliceTainted.Spec.Devices[0].Name, 1))),
				deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTaintedTwice, sliceTainted.Spec.Devices[0].Name, 1)},
				queued:      MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"evict-pod-many-taints": {
			events: []any{
				add(ruleEvict),
				add(ruleEvictOther),
				add(sliceTaintedTwice),
				add(slice2),
				add(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taint1.Key,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taint2.Key,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}()),
				add(podWithClaimName),
			},
			finalState: state{
				slices: l(sliceTaintedTwice, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taint1.Key,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(60)),
						},
						{
							Operator:          resourceapi.DeviceTolerationOpExists,
							Key:               taint2.Key,
							Effect:            resourceapi.DeviceTaintEffectNoExecute,
							TolerationSeconds: ptr.To(int64(30)),
						},
					}
					return claim
				}(), newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), ruleEvict, ruleEvictOther, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 0, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 1))),
				deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), ruleEvict, ruleEvictOther, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 0, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 1)},
				queued:      MockState[workItem]{Ready: newWorkItems(ruleEvict, ruleEvictOther), Later: newDelayedWorkItems(podWithClaimName, 30*time.Second)},
			},
			process: []step{
				// First advance time, then delete.
				{
					deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(30*time.Second)), ruleEvict, ruleEvictOther, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 0, sliceTaintedTwice, sliceTaintedTwice.Spec.Devices[0].Name, 1)},
					pods:            l(podWithClaimName),
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime), inProgress(ruleEvictOther, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(podWithClaimName, 30*time.Second)},
					advance:         30 * time.Second,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}, ruleEvictOther.UID: {numEvictedPods: 1}},
					// Not updated yet.
					rules:           l(inProgress(ruleEvict, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime), inProgress(ruleEvictOther, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", taintTime)),
					queuedProcessed: MockState[workItem]{Later: newDelayedWorkItems(ruleEvict, ruleStatusPeriod, ruleEvictOther, ruleStatusPeriod)},
					advance:         ruleStatusPeriod,
					queuedShifted:   MockState[workItem]{Ready: newWorkItems(ruleEvict, ruleEvictOther)},
				},
				{
					ruleStats: map[types.UID]taintRuleStats{ruleEvict.UID: {numEvictedPods: 1}, ruleEvictOther.UID: {numEvictedPods: 1}},

					// Final update.
					rules: l(inProgress(ruleEvict, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(30*time.Second+ruleStatusPeriod))), inProgress(ruleEvictOther, false, "Completed", "1 pod evicted since starting the controller.", metav1Time(taintTime.Add(30*time.Second+ruleStatusPeriod)))),
				},
			},
			wantEvents: l(deletePodEvent),
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
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
		},
		"no-evict-no-taint": {
			events: []any{
				add(simpleSlice),
				add(inUseClaim),
				add(podWithClaimName),
			},
			finalState: state{
				slices:          l(simpleSlice),
				allocatedClaims: l(ac(inUseClaim)),
			},
		},
		"no-evict-no-taint-update": {
			initialState: state{
				pods:            l(podWithClaimName),
				slices:          l(simpleSlice),
				allocatedClaims: l(ac(inUseClaim)),
			},
			events: []any{
				update(simpleSlice, simpleSlice),
			},
			finalState: state{
				slices:          l(simpleSlice),
				allocatedClaims: l(ac(inUseClaim)),
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
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
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
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
		},
		"no-evict-other-device": {
			events: []any{
				add(sliceOtherDevices),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimTemplateInStatus),
			},
			finalState: state{
				slices:          l(sliceOtherDevices, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
		},
		"no-evict-wrong-pod": {
			events: []any{
				add(sliceTainted),
				add(slice2),
				add(inUseClaim),
				add(podWithClaimNameOtherUID),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
		},
		"evict-wrong-pod-replaced": {
			initialState: state{
				pods:            l(podWithClaimNameOtherUID),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
			events: []any{
				[]any{remove(podWithClaimNameOtherUID), add(podWithClaimName)},
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimName): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"cancel-eviction-remove-taint": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				update(sliceTainted, slice),
			},
			finalState: state{
				slices:          l(slice, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			wantEvents: l(cancelPodEviction),
		},
		"cancel-eviction-reduce-taint": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				update(sliceTainted, sliceTaintedNoSchedule),
			},
			finalState: state{
				slices:          l(sliceTaintedNoSchedule, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			wantEvents: l(cancelPodEviction),
		},
		"ignore-none-effect": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTaintedNone, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			finalState: state{
				slices:          l(sliceTaintedNone, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
		},
		"ignore-unknown-effect": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTaintedUnknown, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			finalState: state{
				slices:          l(sliceTaintedUnknown, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
		},
		"eviction-change-taint": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaimWithToleration, newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(metav1Time(taintTime.Add(tolerationDuration)), sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				// Going from a taint which is tolerated for 60 seconds to one which isn't.
				update(sliceTainted, sliceTaintedValueOther),
			},
			finalState: state{
				slices:          l(sliceTaintedValueOther, slice2),
				allocatedClaims: l(ac(inUseClaimWithToleration, newEvictionTime(taintTime, sliceTaintedValueOther, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTaintedValueOther, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimName)},
			},
			wantEvents: l(deletePodEvent),
		},
		"cancel-eviction-remove-taint-in-new-slice": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				// This moves the in-use device from one slice to another and removes the taint at the same time.
				remove(sliceTainted),
				add(sliceReplaced),
			},
			finalState: state{
				slices:          l(sliceReplaced, slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			wantEvents: l(cancelPodEviction),
		},
		"cancel-eviction-remove-slice": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				remove(sliceTainted),
			},
			finalState: state{
				slices:          l(slice2),
				allocatedClaims: l(ac(inUseClaim)),
			},
			wantEvents: l(cancelPodEviction),
		},
		"cancel-eviction-pod-deletion": {
			initialState: state{
				pods:   l(podWithClaimName),
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator:          resourceapi.DeviceTolerationOpExists,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), newEvictionTime(metav1Time(taintTime.Add(60*time.Second))))),
				deletePodAt: evictMap{newObject(podWithClaimName): *newEvictionTime(metav1Time(taintTime.Add(60 * time.Second)))},
			},
			events: []any{
				remove(podWithClaimName),
			},
			finalState: state{
				slices: l(sliceTainted, slice2),
				allocatedClaims: l(ac(func() *resourceapi.ResourceClaim {
					claim := inUseClaim.DeepCopy()
					claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
						Operator:          resourceapi.DeviceTolerationOpExists,
						Effect:            resourceapi.DeviceTaintEffectNoExecute,
						TolerationSeconds: ptr.To(int64(60)),
					}}
					return claim
				}(), newEvictionTime(metav1Time(taintTime.Add(60*time.Second))))),
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
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaimOld, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
		},
		"evict-wrong-resourceclaim-replaced": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaimOld, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
			},
			events: []any{
				update(inUseClaimOld, inUseClaim),
			},
			finalState: state{
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
				queued:          MockState[workItem]{Ready: newWorkItems(podWithClaimTemplateInStatus)},
			},
			wantEvents: l(deletePodEvent),
		},
		"no-evict-resourceclaim-deallocated": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
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
				slices: l(sliceTainted, slice2),
			},
			wantEvents: l(cancelPodEviction),
		},
		"no-evict-resourceclaim-deleted": {
			initialState: state{
				pods:            l(podWithClaimTemplateInStatus),
				slices:          l(sliceTainted, slice2),
				allocatedClaims: l(ac(inUseClaim, newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0))),
				deletePodAt:     evictMap{newObject(podWithClaimTemplateInStatus): *newEvictionTime(taintTime, sliceTainted, sliceTainted.Spec.Devices[0].Name, 0)},
			},
			events: []any{
				// Same as for "no-evict-resourceclaim-deallocated" this can be normal
				// (pod has terminated) and abnormal (force-delete).
				remove(inUseClaim),
			},
			finalState: state{
				slices: l(sliceTainted, slice2),
			},
			wantEvents: l(cancelPodEviction),
		},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			numEvents := len(tc.events)
			if numEvents <= 1 {
				// No permutations.
				tCtx.SyncTest("", func(tCtx ktesting.TContext) {
					tContext := setup(tCtx)
					testHandlers(tContext, tc)
				})
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
					tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
						tContext := setup(tCtx)
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
	assertEqual(tContext, MockState[workItem]{}, tc.initialState.queued, "initial work queue state")
	tContext.taintRuleStats = tc.initialState.ruleStats
	if tContext.taintRuleStats == nil {
		tContext.taintRuleStats = make(map[types.UID]taintRuleStats)
	}

	// Shallow copy of slice and maps is sufficient for now.
	if len(tc.initialState.deletePodAt) > 0 {
		tContext.deletePodAt = maps.Clone(tc.initialState.deletePodAt)
	}
	tContext.allocatedClaims = tc.initialState.allocatedClaimsAsMap()
	tContext.pools = tc.initialState.slicesAsMap()

	// Pods and DeviceTaintRules are the only items which get retrieved from the informer cache,
	// so for those (and only those) we have to keep the podStore up-to-date.
	// Same for the fake store. Because the informers are not running, API calls
	// must directly get mirrored in the caches.
	podStore := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()
	for _, pod := range tc.initialState.pods {
		tContext.ExpectNoError(podStore.Add(pod))
		tContext.ExpectNoError(tContext.client.Tracker().Add(pod))
	}
	tContext.client.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if err := podStore.Delete(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: action.GetNamespace(), Name: action.(core.DeleteAction).GetName()}}); err != nil {
			return false, nil, fmt.Errorf("delete pod in informer cache: %w", err)
		}
		return false, nil, nil
	})
	ruleStore := tContext.informerFactory.Resource().V1alpha3().DeviceTaintRules().Informer().GetStore()
	for _, rule := range tc.initialState.rules {
		tContext.ExpectNoError(ruleStore.Add(rule))
		tContext.ExpectNoError(tContext.client.Tracker().Add(rule))
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

	assertEqual(tContext, tc.finalState.ruleStats, tContext.taintRuleStats, "taintRuleStats")
	assertEqual(tContext, tc.finalState.deletePodAt, tContext.deletePodAt, "deletePodAt")
	assertEqual(tContext, tc.finalState.allocatedClaimsAsMap(), tContext.allocatedClaims, "allocated claims")
	if !assertEqual(tContext, tc.finalState.slicesAsMap(), tContext.pools, "pools") {
		for key := range tContext.pools {
			assert.Equal(tContext, tc.finalState.slicesAsMap()[key], tContext.pools[key], "pool")
		}
	}
	assertEqual(tContext, tc.finalState.queued, tContext.mockQueue.State(), "work queue after event handlers", cmpopts.SortSlices(compareWorkItems))
	assert.Empty(tContext, tc.finalState.pods, "pods not checked for final state")
	assert.Empty(tContext, tc.finalState.rules, "rules not checked for final state")

	process := tc.process
	if process == nil && len(tc.finalState.queued.Ready) > 0 {
		// Expect to clear workqueue and delete pods.
		process = []step{{}}
	}
	for i, state := range process {
		prefix := fmt.Sprintf("process #%d: ", i)

		// This runs until the "Ready" queue is empty.
		// Some state may have changed (e.g. rule status), some must remain the same (allocated claims).
		tContext.Log(prefix + "handling ready work items")
		tContext.Controller.worker(tContext)

		assertEqual(tContext, state.ruleStats, tContext.taintRuleStats, prefix+"taintRuleStats")
		assertEqual(tContext, state.deletePodAt, tContext.deletePodAt, prefix+"deletePodAt")
		assertEqual(tContext, tc.finalState.allocatedClaimsAsMap(), tContext.allocatedClaims, prefix+"allocated claims")
		pods, err := tContext.client.CoreV1().Pods("").List(tContext, metav1.ListOptions{})
		tContext.ExpectNoError(err, prefix+"list pods")
		assertEqual(tContext, state.pods, trimPods(pods.Items), prefix+"pods after flushing work queue")
		rules, err := tContext.client.ResourceV1alpha3().DeviceTaintRules().List(tContext, metav1.ListOptions{})
		tContext.ExpectNoError(err, prefix+"list rules")
		actualRules := trimRules(rules.Items)
		assertEqual(tContext, state.rules, actualRules, prefix+"rules after flushing work queue")

		// Advance time and potentially make pending work items ready.
		assertEqual(tContext, state.queuedProcessed, tContext.mockQueue.State(), prefix+"work queue after processing", cmpopts.SortSlices(compareWorkItems))
		time.Sleep(state.advance)
		for _, item := range tContext.mockQueue.State().Later {
			tContext.mockQueue.CancelAfter(item.Item)
			tContext.mockQueue.AddAfter(item.Item, item.Duration-state.advance)
		}
		assertEqual(tContext, state.queuedShifted, tContext.mockQueue.State(), prefix+"work queue after moving time forward", cmpopts.SortSlices(compareWorkItems))
	}

	assertEqual(tContext, tc.wantEvents, tContext.recorder.Events, "overall events",
		cmpopts.IgnoreTypes(metav1.ObjectMeta{}, metav1.Time{}),
		cmpopts.SortSlices(compareEvents))
}

// More fields might be needed for these compare functions in the future, but for now this is enough.

func compareEvents(a, b *v1.Event) int {
	if cmp := strings.Compare(string(a.InvolvedObject.UID), string(b.InvolvedObject.UID)); cmp != 0 {
		return cmp
	}
	return strings.Compare(a.Kind, b.Kind)
}

func compareWorkItems(a, b workItem) int {
	if cmp := strings.Compare(string(a.podRef.UID), string(b.podRef.UID)); cmp != 0 {
		return cmp
	}
	return strings.Compare(string(a.ruleRef.UID), string(b.ruleRef.UID))
}

func applyEventPair(tContext *testContext, event any) {
	switch pair := event.(type) {
	case [2]*resourceapi.ResourceSlice:
		tContext.handleSliceChange(pair[0], pair[1])
	case [2]*resourceapi.ResourceClaim:
		tContext.handleClaimChange(pair[0], pair[1])
	case [2]*v1.Pod:
		store := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()
		switch {
		case pair[0] != nil && pair[1] != nil:
			tContext.ExpectNoError(store.Update(pair[1]))
			tContext.ExpectNoError(tContext.client.Tracker().Update(v1.SchemeGroupVersion.WithResource("pods"), pair[1], pair[1].Namespace))
		case pair[0] != nil:
			tContext.ExpectNoError(store.Delete(pair[0]))
			tContext.ExpectNoError(tContext.client.Tracker().Delete(v1.SchemeGroupVersion.WithResource("pods"), pair[0].Namespace, pair[0].Name))
		default:
			tContext.ExpectNoError(store.Add(pair[1]))
			tContext.ExpectNoError(tContext.client.Tracker().Add(pair[1]))
		}
		tContext.handlePodChange(pair[0], pair[1])
	case [2]*resourcealpha.DeviceTaintRule:
		store := tContext.informerFactory.Resource().V1alpha3().DeviceTaintRules().Informer().GetStore()
		switch {
		case pair[0] != nil && pair[1] != nil:
			tContext.ExpectNoError(store.Update(pair[1]))
			tContext.ExpectNoError(tContext.client.Tracker().Update(resourcealpha.SchemeGroupVersion.WithResource("devicetaintrules"), pair[1], pair[1].Namespace))
		case pair[0] != nil:
			tContext.ExpectNoError(store.Delete(pair[0]))
			tContext.ExpectNoError(tContext.client.Tracker().Delete(resourcealpha.SchemeGroupVersion.WithResource("devicetaintrules"), pair[0].Namespace, pair[0].Name))
		default:
			tContext.ExpectNoError(store.Add(pair[1]))
			tContext.ExpectNoError(tContext.client.Tracker().Add(pair[1]))
		}
		tContext.handleRuleChange(pair[0], pair[1])
	default:
		tContext.Fatalf("unexpected event type %T", event)
	}
}

func trimPods(objs []v1.Pod) (trimmed []*v1.Pod) {
	for _, in := range objs {
		out := in.DeepCopy()
		out.ManagedFields = nil
		trimmed = append(trimmed, out)
	}
	return trimmed
}

func trimRules(objs []resourcealpha.DeviceTaintRule) (trimmed []*resourcealpha.DeviceTaintRule) {
	for _, in := range objs {
		out := in.DeepCopy()
		out.ManagedFields = nil
		out.Kind = ""
		out.APIVersion = ""
		trimmed = append(trimmed, out)
	}
	return trimmed
}

func newTestController(tCtx ktesting.TContext, clientSet *fake.Clientset) *Controller {
	// fake.Clientset suffers from a race condition related to informers:
	// it does not implement resource version support in its Watch
	// implementation and instead assumes that watches are set up
	// before further changes are made.
	//
	// If a test waits for caches to be synced and then immediately
	// adds an object, that new object will never be seen by event handlers
	// if the race goes wrong and the Watch call hadn't completed yet
	// (can be triggered by adding a sleep before https://github.com/kubernetes/kubernetes/blob/b53b9fb5573323484af9a19cf3f5bfe80760abba/staging/src/k8s.io/client-go/tools/cache/reflector.go#L431).
	//
	// To work around this, we count all watches and only proceed when
	// all of them are in place. This replaces the normal watch reactor
	// (https://github.com/kubernetes/kubernetes/blob/b53b9fb5573323484af9a19cf3f5bfe80760abba/staging/src/k8s.io/client-go/kubernetes/fake/clientset_generated.go#L161-L173).
	var numWatches atomic.Int32
	clientSet.PrependWatchReactor("*", func(action core.Action) (handled bool, ret watch.Interface, err error) {
		var opts metav1.ListOptions
		if watchActcion, ok := action.(core.WatchActionImpl); ok {
			opts = watchActcion.ListOptions
		}
		gvr := action.GetResource()
		ns := action.GetNamespace()
		watch, err := clientSet.Tracker().Watch(gvr, ns, opts)
		if err != nil {
			return false, nil, err
		}
		numWatches.Add(1)
		return true, watch, nil
	})

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	featuregatetesting.SetFeatureGatesDuringTest(tCtx, utilfeature.DefaultFeatureGate,
		featuregatetesting.FeatureOverrides{
			features.DRADeviceTaints:     true,
			features.DRADeviceTaintRules: true,
		},
	)
	controller := New(tCtx.Client(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().ResourceSlices(),
		informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		informerFactory.Resource().V1().DeviceClasses(),
		"device-taint-eviction",
	)
	controller.metrics = metrics.New(300 /* one large initial bucket for testing */) // TODO: inside a synctest bubble we should have deterministic delays and shouldn't need this trick. The remaining uncertainty comes from polling for informer cache sync.
	// Always log, not matter what the -v value is.
	logger := klog.FromContext(tCtx)
	controller.eventLogger = &logger

	informerFactory.Start(tCtx.Done())
	tCtx.Cleanup(informerFactory.Shutdown)

	tCtx.Log("starting to wait for watches")
	if tCtx.IsSyncTest() {
		tCtx.Wait()
		require.Equal(tCtx, int32(5), numWatches.Load(), "All watches should be registered.")
	} else {
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) int32 {
			return numWatches.Load()
		}).WithTimeout(5*time.Second).Should(gomega.Equal(int32(5)), "All watches should be registered.")
	}
	tCtx.Log("done waiting for watches")

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
//
// This runs in a bubble (https://pkg.go.dev/testing/synctest), so we can wait for goroutine
// activity to settle down and then check the state.
func TestEviction(t *testing.T) { testEviction(ktesting.Init(t)) }
func testEviction(tCtx ktesting.TContext) {
	do := func(tCtx ktesting.TContext, what string, action func(tCtx ktesting.TContext) error) {
		tCtx.Log(what)
		err := action(tCtx)
		require.NoError(tCtx, err, what)
	}

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
				do(tCtx, "create pod", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
					return err
				})
				do(tCtx, "create slice", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, sliceTainted, metav1.CreateOptions{})
					return err
				})
				do(tCtx, "create claim", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().ResourceV1().ResourceClaims(inUseClaim.Namespace).Create(tCtx, inUseClaim, metav1.CreateOptions{})
					return err
				})
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
				do(tCtx, "update pod", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Update(tCtx, pod, metav1.UpdateOptions{})
					return err
				})
				do(tCtx, "update slice", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().ResourceV1().ResourceSlices().Update(tCtx, sliceTainted, metav1.UpdateOptions{})
					return err
				})
				do(tCtx, "update claim", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().ResourceV1().ResourceClaims(inUseClaim.Namespace).UpdateStatus(tCtx, inUseClaim, metav1.UpdateOptions{})
					return err
				})
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
				do(tCtx, "delete slice", func(tCtx ktesting.TContext) error {
					return tCtx.Client().ResourceV1().ResourceSlices().Delete(tCtx, slice.Name+"-other", metav1.DeleteOptions{})
				})
				do(tCtx, "delete claim", func(tCtx ktesting.TContext) error {
					return tCtx.Client().ResourceV1().ResourceClaims(inUseClaim.Namespace).Delete(tCtx, inUseClaim.Name, metav1.DeleteOptions{})
				})

				// Re-create after deletion to enabled the normal flow.
				do(tCtx, "create claim", func(tCtx ktesting.TContext) error {
					_, err := tCtx.Client().ResourceV1().ResourceClaims(inUseClaim.Namespace).Create(tCtx, inUseClaim, metav1.CreateOptions{})
					return err
				})
			},
		},
	} {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			start := time.Now()
			fakeClientset := fake.NewClientset(tt.initialObjects...)
			tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)

			var podGets int
			var podUpdates int
			var updatedPod *v1.Pod
			var podDeletions int

			fakeClientset.PrependReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
				podGets++
				podName := action.(core.GetAction).GetName()
				assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to patch")
				return false, nil, nil
			})
			fakeClientset.PrependReactor("patch", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
				podUpdates++
				podName := action.(core.PatchAction).GetName()
				assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to get")
				return false, nil, nil
			})
			fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
				podDeletions++
				podName := action.(core.DeleteAction).GetName()
				assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to delete")
				obj, err := fakeClientset.Tracker().Get(v1.SchemeGroupVersion.WithResource("pods"), pod.Namespace, pod.Name)
				require.NoError(tCtx, err)
				updatedPod = obj.(*v1.Pod)
				return false, nil, nil
			})
			controller := newTestController(tCtx, fakeClientset)

			var wg sync.WaitGroup
			defer func() {
				tCtx.Log("Waiting for goroutine termination...")
				tCtx.Cancel("time to stop")
				wg.Wait()
			}()
			wg.Add(1)
			go func() {
				defer wg.Done()
				assert.NoError(tCtx, controller.Run(tCtx, 10 /* workers */), "eviction controller failed")
			}()

			// Eventually the controller should have synced it's informers.
			if false {
				// This feels like it should work (controller should run until it has started up, then block durably), but it doesn't.
				// Time progresses while the controller is blocked in cache.WaitForNamedCacheSyncWithContext, so this is
				// probably a good place to start looking.
				// TODO: make "wait for cache sync" block on a channel. Alternatively, use a context and let `context.Cause`
				// report success or failure (might be too hacky).
				tCtx.Wait()
				if controller.hasSynced.Load() <= 0 {
					tCtx.Fatal("controller should have synced")
				}
			} else {
				ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
					return controller.hasSynced.Load() > 0
				}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("controller synced"))
				if tt.afterSync != nil {
					tt.afterSync(tCtx)
				}
			}

			// We can wait for the controller to be idle.
			tCtx.Wait()

			// The number of API calls is deterministic.
			assert.Equal(tCtx, 1, podGets, "get pod once")
			assert.Equal(tCtx, 1, podUpdates, "update pod once")
			assert.Equal(tCtx, 1, podDeletions, "delete pod once")

			_, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
			switch {
			case err == nil:
				tCtx.Fatalf("Pod should have been deleted, it still exists")
			case apierrors.IsNotFound(err):
				// Okay.
			default:
				tCtx.Fatalf("Retrieving pod failed: %v", err)
			}

			pod := pod.DeepCopy()
			pod.Status.Conditions = []v1.PodCondition{{
				Type:    v1.DisruptionTarget,
				Status:  v1.ConditionTrue,
				Reason:  "DeletionByDeviceTaintManager",
				Message: "Device Taint manager: deleting due to NoExecute taint",
			}}
			if diff := cmp.Diff(pod, updatedPod, cmpopts.IgnoreTypes(metav1.Time{}, metav1.TypeMeta{}), cmpopts.IgnoreFields(metav1.ObjectMeta{}, "ManagedFields")); diff != "" {
				tCtx.Errorf("unexpected modified pod (-want, +got):\n%s", diff)
			}

			// Shortly after deletion we should also see updated metrics.
			// This is the last thing the controller does for a pod.
			// Because of Wait we know that all goroutines are durably blocked and won't
			// wake up again to change the metrics => no need for a "Consistently"!
			gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
			tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 1), "pod eviction done")

			// Depending on timing, some of the "wait for cache synced" polling sleep a bit more or less,
			// so there is a certain delta of uncertainty about the overall duration. Without that polling
			// we probably could assert zero runtime here.
			tCtx.Logf("eviction duration: %s", time.Since(start))
			delta := time.Second
			require.WithinRange(tCtx, time.Now(), start, start.Add(delta), "time to evict pod")
		})
	}
}

// TestDeviceTaintRule runs through the full flow of simulating eviction with the None effect,
// updating the rule with NoExecute, and then evicting the pod.
//
// This runs in a bubble (https://pkg.go.dev/testing/synctest), so we can wait for goroutine
// activity to settle down and then check the state.
func TestDeviceTaintRule(t *testing.T) { ktesting.Init(t).SyncTest("", synctestDeviceTaintRule) }
func synctestDeviceTaintRule(tCtx ktesting.TContext) {
	rule := ruleNone.DeepCopy()
	fakeClientset := fake.NewClientset(podWithClaimName, inUseClaim, rule)
	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)
	controller := newTestController(tCtx, fakeClientset)

	var wg sync.WaitGroup
	defer func() {
		tCtx.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Go(func() {
		assert.NoError(tCtx, controller.Run(tCtx, 10 /* workers */), "eviction controller failed")
	})

	// Eventually the controller should have synced it's informers.
	if false {
		// This feels like it should work (controller should run until it has started up, then block durably), but it doesn't.
		// Time progresses while the controller is blocked in cache.WaitForNamedCacheSyncWithContext, so this is
		// probably a good place to start looking.
		// TODO: make "wait for cache sync" block on a channel. Alternatively, use a context and let `context.Cause`
		// report success or failure (might be too hacky).
		tCtx.Wait()
		if controller.hasSynced.Load() <= 0 {
			tCtx.Fatal("controller should have synced")
		}
	} else {
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
			return controller.hasSynced.Load() > 0
		}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("controller synced"))
	}

	// We can wait for the controller to be idle.
	tCtx.Wait()
	start := metav1.Now()
	tCtx.Logf("TIME: start at %s", start)
	check(tCtx, "initial processing: ", l(inProgress(ruleNone, false, "NoEffect", "0 published devices selected. 1 allocated device selected. 1 pod would be evicted in 1 namespace if the effect was NoExecute. This information will not be updated again. Recreate the DeviceTaintRule to trigger an update.", &start /* processed before waiting for cache sync completion */)), l(podWithClaimName))

	// Move time forward to ensure that we get different time stamps.
	time.Sleep(20 * time.Second)
	rule.Spec.Taint.Effect = resourcealpha.DeviceTaintEffectNoExecute
	rule, err := tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Update(tCtx, rule, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "update rule")

	// Wait for eviction. The rule gets updated with another delay.
	tCtx.Wait()
	evicted := metav1.Now()
	tCtx.Logf("TIME: eviction done at %s", evicted)
	check(tCtx, "evict: ", l(inProgress(rule, true, "PodsPendingEviction", "1 pod needs to be evicted in 1 namespace.", &evicted)), nil)

	// AddAfter does not move time forward. Do it ourselves...
	time.Sleep(ruleStatusPeriod)
	slept := metav1.Now()
	tCtx.Logf("TIME: slept till %s", slept)
	tCtx.Wait()
	done := metav1.Now()
	tCtx.Logf("TIME: done at %s", done)
	check(tCtx, "done: ", l(inProgress(rule, false, "Completed", "1 pod evicted since starting the controller.", &slept)), nil)
	assertEqual(tCtx, map[types.UID]taintRuleStats{rule.UID: {numEvictedPods: 1}}, controller.taintRuleStats, "taint rule statistics should have counted the pod")

	// Delete the rule and verify that we don't leak memory by still tracking it.
	err = tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Delete(tCtx, rule.Name, metav1.DeleteOptions{})
	tCtx.ExpectNoError(err, "delete rule")
	tCtx.Wait()
	deleted := metav1.Now()
	tCtx.Logf("TIME: deleted at %s", deleted)
	assert.Empty(tCtx, controller.taintRuleStats, "taint rule statistics should have dropped the deleted rule")
}

func check(tCtx ktesting.TContext, prefix string, expectRules []*resourcealpha.DeviceTaintRule, expectPods []*v1.Pod) {
	tCtx.Helper()

	opts := []cmp.Option{
		// Expected objects don't have managed fields, API objects do.
		cmpopts.IgnoreFields(metav1.ObjectMeta{}, "ManagedFields"),
		// metav1.Time gets rounded to seconds during serialization.
		cmpopts.AcyclicTransformer("RoundTime", func(t metav1.Time) metav1.Time {
			return metav1.Time{Time: t.Round(time.Second)}
		}),
	}

	actualPods, err := tCtx.Client().CoreV1().Pods("").List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, prefix+"list pods")
	assertEqual(tCtx, expectPods, trimPods(actualPods.Items), prefix+"pods", opts...)
	rules, err := tCtx.Client().ResourceV1alpha3().DeviceTaintRules().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, prefix+"list rules")
	assertEqual(tCtx, expectRules, trimRules(rules.Items), prefix+"rules", opts...)
}

// TestCancelEviction deletes the pod before the controller deletes it
// or removes the slice. Either way, eviction gets cancelled.
func TestCancelEviction(t *testing.T) { testCancelEviction(ktesting.Init(t)) }
func testCancelEviction(tCtx ktesting.TContext) {
	tCtx.SyncTest("pod-deleted", func(tCtx ktesting.TContext) { doCancelEviction(tCtx, true) })
	tCtx.SyncTest("slice-deleted", func(tCtx ktesting.TContext) { doCancelEviction(tCtx, false) })
}

func doCancelEviction(tCtx ktesting.TContext, deletePod bool) {
	// The claim tolerates the taint long enough for us to
	// do something which cancels eviction.
	pod := podWithClaimName.DeepCopy()
	slice := sliceTainted.DeepCopy()
	slice.Spec.Devices[0].Taints[0].TimeAdded = metav1Time(time.Now())
	claim := inUseClaim.DeepCopy()
	tolerationSeconds := int64(60)
	claim.Status.Allocation.Devices.Results[0].Tolerations = []resourceapi.DeviceToleration{{
		Operator:          resourceapi.DeviceTolerationOpExists,
		Effect:            resourceapi.DeviceTaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	fakeClientset := fake.NewClientset(
		slice,
		claim,
		pod,
	)
	pod, err := fakeClientset.CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	require.NoError(tCtx, err, "get pod before eviction")
	assert.Equal(tCtx, podWithClaimName, pod, "test pod")

	var podGets int
	var podUpdates int
	var podDeletions int

	fakeClientset.PrependReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		podGets++
		podName := action.(core.GetAction).GetName()
		assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to patch")
		return false, nil, nil
	})
	fakeClientset.PrependReactor("patch", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		podUpdates++
		podName := action.(core.PatchAction).GetName()
		assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to get")
		return false, nil, nil
	})
	fakeClientset.PrependReactor("delete", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		podDeletions++
		podName := action.(core.DeleteAction).GetName()
		assert.Equal(tCtx, podWithClaimName.Name, podName, "name of pod to delete")
		return false, nil, nil
	})

	tCtx = ktesting.WithClients(tCtx, nil, nil, fakeClientset, nil, nil)
	controller := newTestController(tCtx, fakeClientset)

	var mutex sync.Mutex
	podEvicting := false
	controller.evictPodHook = func(podRef tainteviction.NamespacedObject, eviction evictionAndReason) {
		assert.Equal(tCtx, newObject(pod), podRef)
		mutex.Lock()
		defer mutex.Unlock()
		podEvicting = true
	}
	controller.cancelEvictHook = func(podRef tainteviction.NamespacedObject) bool {
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
		assert.NoError(tCtx, controller.Run(tCtx, 10 /* workers */), "eviction controller failed")
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
		tCtx.ExpectNoError(fakeClientset.ResourceV1().ResourceSlices().Delete(tCtx, slice.Name, metav1.DeleteOptions{}))
	}

	// Shortly after deletion we should also see the cancellation.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
		mutex.Lock()
		defer mutex.Unlock()
		return podEvicting
	}).WithTimeout(30 * time.Second).Should(gomega.BeFalseBecause("pod no longer pending eviction"))

	// Whether we get an event depends on whether the pod still exists.
	// If we expect an event, we need to wait for it.
	if !deletePod {
		ktesting.Eventually(tCtx, listEvents).WithTimeout(30 * time.Second).Should(matchCancellationEvent())
	}
	tCtx.Wait()

	matchEvents := matchCancellationEvent()
	if deletePod {
		matchEvents = gomega.BeEmpty()
		assert.Equal(tCtx, 1, podDeletions, "Pod should have been deleted exactly once by test.")
	} else {
		assert.Equal(tCtx, 0, podDeletions, "Pod should not have been deleted.")
	}

	// Naively (?) one could expect synctest.Wait to have blocked until the work item added via AddAfter
	// got processed because before that the overall state isn't stable yet. But the workqueue package
	// seems to implement AddAfter in a way which is not detected as "blocking on time to pass" by
	// by synctest and therefore it returns without advancing time enough.
	//
	// Here we trigger that manually as a workaround (?). The factor doesn't really matter.
	// Commenting this out causes the controller.maybeDeletePodCount check to fail.
	time.Sleep(10 * time.Duration(tolerationSeconds) * time.Second)
	tCtx.Wait()

	assert.Equal(tCtx, 0, podGets, "Worker should not have needed to get the pod.")
	assert.Equal(tCtx, 0, podUpdates, "Worker should not have needed to update the pod.")
	assert.Equal(tCtx, 0, controller.workqueue.Len(), "Work queue should be empty now.")
	assert.Equal(tCtx, int64(1), controller.maybeDeletePodCount, "Work queue should have processed pod.")
	gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchEvents)
	tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 0))
}

// TestParallelPodDeletion covers the scenario that a pod gets deleted right before
// trying to evict it.
func TestParallelPodDeletion(t *testing.T) { testParallelPodDeletion(ktesting.Init(t)) }
func testParallelPodDeletion(tCtx ktesting.TContext) {
	tCtx.Parallel()

	tCtx.SyncTest("", func(tCtx ktesting.TContext) {
		// This scenario is the same as "evict-pod-resourceclaim" above.
		pod := podWithClaimName.DeepCopy()
		fakeClientset := fake.NewClientset(
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
			assert.Equal(tCtx, podWithClaimName.Name, podName, "name of patched pod")

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
			assert.Equal(tCtx, podWithClaimName.Name, podName, "name of deleted pod")
			return false, nil, nil
		})
		controller := newTestController(tCtx, fakeClientset)

		var wg sync.WaitGroup
		defer func() {
			tCtx.Log("Waiting for goroutine termination...")
			tCtx.Cancel("time to stop")
			wg.Wait()
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			assert.NoError(tCtx, controller.Run(tCtx, 10 /* workers */), "eviction controller failed")
		}()

		// Eventually the pod gets deleted, in this test by us.
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) bool {
			mutex.Lock()
			defer mutex.Unlock()
			return podGets >= 1
		}).WithTimeout(30 * time.Second).Should(gomega.BeTrueBecause("pod eviction started"))

		// We don't want any events.
		tCtx.Wait()
		assert.Equal(tCtx, 1, podGets, "number of pod get calls")
		assert.Equal(tCtx, 0, podDeletions, "number of pod delete calls")
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(gomega.BeEmpty())
		tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 0))
	})
}

// TestRetry covers the scenario that an eviction attempt must be retried.
func TestRetry(t *testing.T) { ktesting.Init(t).SyncTest("", synctestRetry) }
func synctestRetry(tCtx ktesting.TContext) {
	// This scenario is the same as "evict-pod-resourceclaim" above.
	pod := podWithClaimName.DeepCopy()
	fakeClientset := fake.NewClientset(
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
		assert.Equal(tCtx, podWithClaimName.Name, podName, "name of patched pod")

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
		assert.Equal(tCtx, podWithClaimName.Name, podName, "name of deleted pod")
		return false, nil, nil
	})
	controller := newTestController(tCtx, fakeClientset)

	var wg sync.WaitGroup
	defer func() {
		tCtx.Log("Waiting for goroutine termination...")
		tCtx.Cancel("time to stop")
		wg.Wait()
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		assert.NoError(tCtx, controller.Run(tCtx, 10 /* workers */), "eviction controller failed")
	}()

	// Eventually the pod gets deleted and the event is recorded.
	ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
		gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
		return testPodDeletionsMetrics(controller, 1)
	}).WithTimeout(30*time.Second).Should(gomega.Succeed(), "pod eviction done")

	// Now we can check the API calls.
	tCtx.Wait()
	assert.Equal(tCtx, 2, podGets, "number of pod get calls")
	assert.Equal(tCtx, 1, podDeletions, "number of pod delete calls")
	gomega.NewWithT(tCtx).Expect(listEvents(tCtx)).Should(matchDeletionEvent())
	tCtx.ExpectNoError(testPodDeletionsMetrics(controller, 1))
}

// BenchTaintUntaint checks the full flow of detecting a claim as
// tainted because of a new DeviceTaintRule, starting to evict its
// consumer, and then undoing that when the DeviceTaintRule is removed.
func BenchmarkTaintUntaint(b *testing.B) {
	tCtx := ktesting.Init(b)
	tContext := setup(tCtx)
	podStore := tContext.informerFactory.Core().V1().Pods().Informer().GetStore()
	// No output, comment out if output is desired.
	tContext.Controller.eventLogger = nil

	// Condition must be exactly "b.Loop" to ensure that the special support
	// in the compiler for benchmarks is active.
	for b.Loop() {
		// Add objects...
		tContext.handleSliceChange(nil, slice)
		tContext.handleClaimChange(nil, inUseClaimWithToleration)
		require.NoError(tContext, podStore.Add(podWithClaimName), "add pod")
		tContext.handlePodChange(nil, podWithClaimName)
		require.Empty(tContext, tContext.deletePodAt)

		// Now evict.
		tContext.handleSliceChange(slice, sliceTainted)

		// Because informer event handlers are synchronous, we get the expected result immediately.
		require.NotEmpty(tContext, tContext.deletePodAt)

		// ... and remove them again.
		tContext.handleSliceChange(sliceTainted, slice)
		require.Empty(tContext, tContext.deletePodAt)

		tContext.handlePodChange(podWithClaimName, nil)
		require.NoError(tContext, podStore.Delete(podWithClaimName), "remove pod")
		tContext.handleClaimChange(inUseClaimWithToleration, nil)
		tContext.handleSliceChange(slice, nil)
	}
}
