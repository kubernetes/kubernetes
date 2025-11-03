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
	"context"
	"fmt"
	"iter"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealpha "k8s.io/api/resource/v1alpha3"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	resourceac "k8s.io/client-go/applyconfigurations/resource/v1alpha3"
	coreinformers "k8s.io/client-go/informers/core/v1"
	resourceinformers "k8s.io/client-go/informers/resource/v1"
	resourcealphainformers "k8s.io/client-go/informers/resource/v1alpha3"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	resourcealphalisters "k8s.io/client-go/listers/resource/v1alpha3"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction/metrics"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	"k8s.io/kubernetes/pkg/features"
	utilpod "k8s.io/kubernetes/pkg/util/pod"
)

const (
	// ruleStatusPeriod is the shortest time between DeviceTaintRule status
	// updates while eviction is in progress. Once it is done, it no longer gets
	// updated until in progress again.
	ruleStatusPeriod = 10 * time.Second
)

// Controller listens to Taint changes of DRA devices and Toleration changes of ResourceClaims,
// then deletes Pods which use ResourceClaims that don't tolerate a NoExecute taint.
// Pods which have already reached a final state (aka terminated) don't need to be deleted.
//
// All of the logic which identifies pods which need to be evicted runs in the
// handle* event handlers. They don't call any blocking method. All the blocking
// calls happen in a [tainteviction.TimedWorkerQueue], using the context passed to Run.
//
// The [resourceslicetracker] takes care of applying taints defined in DeviceTaintRules
// to ResourceSlices. This controller here receives modified ResourceSlices with all
// applicable taints from that tracker and doesn't need to care about where a
// taint came from, the DRA driver or a DeviceTaintRule.
type Controller struct {
	name string

	// logger is the general-purpose logger to be used for background activities.
	logger klog.Logger

	// handlerLogger is specifically for logging during event handling. It may be nil
	// if no such logging is desired.
	eventLogger *klog.Logger

	client        clientset.Interface
	recorder      record.EventRecorder
	podInformer   coreinformers.PodInformer
	podLister     corelisters.PodLister
	claimInformer resourceinformers.ResourceClaimInformer
	sliceInformer resourceinformers.ResourceSliceInformer
	ruleInformer  resourcealphainformers.DeviceTaintRuleInformer
	classInformer resourceinformers.DeviceClassInformer
	ruleLister    resourcealphalisters.DeviceTaintRuleLister
	haveSynced    []cache.InformerSynced
	hasSynced     atomic.Int32
	metrics       metrics.Metrics
	workqueue     workqueue.TypedRateLimitingInterface[workItem]

	evictPodHook    func(pod tainteviction.NamespacedObject, eviction evictionAndReason)
	cancelEvictHook func(pod tainteviction.NamespacedObject) bool

	// mutex protects the following shared data structures.
	mutex sync.Mutex

	// deletePodAt maps a pod to the time when it is meant to be evicted.
	//
	// The entry for pod gets deleted when eviction is no longer necessary
	// and updated when the time changes.
	deletePodAt map[tainteviction.NamespacedObject]evictionAndReason

	// maybeDeletePodCount counts how often a worker checked a pod.
	// This is useful for unit testing, but probably not a good public metric.
	maybeDeletePodCount int64

	// allocatedClaims holds all currently known allocated claims.
	allocatedClaims map[types.NamespacedName]allocatedClaim // A value is slightly more efficient in BenchmarkTaintUntaint (less allocations!).

	// pools indexes all slices by driver and pool name.
	pools map[poolID]pool

	// taintRuleStats tracks information about work that was done for a specific DeviceTaintRule instance.
	taintRuleStats map[types.UID]taintRuleStats

	// simulateRule is set only during simulation of a None effect.
	//
	// During such a simulation the corresponding rule from ruleLister
	// has EffectNone and this one here has EffectNoExecute.
	simulateRule *resourcealpha.DeviceTaintRule
}

type taintRuleStats struct {
	// numEvictedPods is the number of pods evicted because of this rule since starting the controller.
	numEvictedPods int64
}

type poolID struct {
	driverName, poolName string
}

type pool struct {
	// slices maps the global name to the current instance under that name.
	slices        map[string]*resourceapi.ResourceSlice
	maxGeneration int64
}

// addSlice adds one slice to the pool.
func (p *pool) addSlice(slice *resourceapi.ResourceSlice) {
	if slice == nil {
		return
	}
	if p.slices == nil {
		p.slices = make(map[string]*resourceapi.ResourceSlice)
		p.maxGeneration = math.MinInt64
	}
	p.slices[slice.Name] = slice

	// Adding a slice can only increase the generation.
	if slice.Spec.Pool.Generation > p.maxGeneration {
		p.maxGeneration = slice.Spec.Pool.Generation
	}
}

// removeSlice removes a slice. It must have been added before.
func (p *pool) removeSlice(slice *resourceapi.ResourceSlice) {
	if slice == nil {
		return
	}
	delete(p.slices, slice.Name)

	// Removing a slice might have decreased the generation to
	// that of some other slice.
	if slice.Spec.Pool.Generation == p.maxGeneration {
		maxGeneration := int64(math.MinInt64)
		for _, slice := range p.slices {
			if slice.Spec.Pool.Generation > maxGeneration {
				maxGeneration = slice.Spec.Pool.Generation
			}
		}
		p.maxGeneration = maxGeneration
	}
}

// getTaintedDevices appends all device taints with NoExecute effect.
// The result is sorted by device name.
func (p pool) getTaintedDevices() []taintedDevice {
	var buffer []taintedDevice
	for _, slice := range p.slices {
		if slice.Spec.Pool.Generation != p.maxGeneration {
			continue
		}
		for _, device := range slice.Spec.Devices {
			for _, taint := range device.Taints {
				if taint.Effect != resourceapi.DeviceTaintEffectNoExecute {
					continue
				}
				buffer = append(buffer, taintedDevice{deviceName: device.Name, taint: taint})
			}
		}
	}

	// slices.SortFunc is more efficient than sort.Slice here.
	slices.SortFunc(buffer, func(a, b taintedDevice) int {
		return strings.Compare(a.deviceName, b.deviceName)
	})
	return buffer
}

// getDevice looks up one device by name. Out-dated slices are ignored.
func (p pool) getDevice(deviceName string) (*resourceapi.ResourceSlice, *resourceapi.Device) {
	for _, slice := range p.slices {
		if slice.Spec.Pool.Generation != p.maxGeneration {
			continue
		}
		for i := range slice.Spec.Devices {
			if slice.Spec.Devices[i].Name == deviceName {
				return slice, &slice.Spec.Devices[i]
			}
		}
	}

	return nil, nil
}

type taintedDevice struct {
	deviceName string
	taint      resourceapi.DeviceTaint
}

// allocatedClaim is a ResourceClaim which has an allocation result. It
// may or may not be tainted such that pods need to be evicted.
type allocatedClaim struct {
	*resourceapi.ResourceClaim

	// eviction, if non-nil, is the time at which pods using this claim need to be evicted.
	// This is the smallest value of all such per-device values.
	// For each device, the value is calculated as `<time of setting the taint> +
	// <toleration seconds, 0 if not set>`.
	eviction *evictionAndReason
}

// evictionAndReason combines the time when eviction needs to start with all reasons
// why eviction needs to start.
type evictionAndReason struct {
	when   metav1.Time
	reason evictionReason
}

func (et evictionAndReason) String() string {
	return fmt.Sprintf("%s (%s)", et.when, et.reason)
}

func (et *evictionAndReason) equal(other *evictionAndReason) bool {
	if (et == nil) != (other == nil) ||
		et == nil {
		return false
	}
	return et.when.Equal(&other.when) &&
		slices.Equal(et.reason, other.reason)
}

// evictionReason collects all taints which caused eviction.
// It supports pretty-printing for logging and inclusion in
// user-facing descriptions
type evictionReason []trackedTaint

func (er evictionReason) String() string {
	var parts []string
	for _, taint := range er {
		parts = append(parts, taint.String())
	}
	return strings.Join(parts, ", ")
}

// trackedTaint augments a DeviceTaint with a pointer to its origin.
// rule and slice are mutually exclusive. Exactly one of them is always set.
type trackedTaint struct {
	rule  *resourcealpha.DeviceTaintRule
	slice sliceDeviceTaint
}

func (tt trackedTaint) deviceTaint() *resourceapi.DeviceTaint {
	if tt.rule != nil {
		// TODO when GA: directly point to rule.Spec.Taint.
		return &resourceapi.DeviceTaint{
			Key:       tt.rule.Spec.Taint.Key,
			Value:     tt.rule.Spec.Taint.Value,
			Effect:    resourceapi.DeviceTaintEffect(tt.rule.Spec.Taint.Effect),
			TimeAdded: tt.rule.Spec.Taint.TimeAdded,
		}
	}
	if tt.slice.slice != nil {
		index := slices.IndexFunc(tt.slice.slice.Spec.Devices, func(d resourceapi.Device) bool { return d.Name == tt.slice.deviceName })
		return &tt.slice.slice.Spec.Devices[index].Taints[tt.slice.taintIndex]
	}

	// Huh?
	return nil
}

func (tt trackedTaint) String() string {
	if tt.rule != nil {
		return fmt.Sprintf("DeviceTaintRule %s", newObject(tt.rule))
	}
	return fmt.Sprintf("ResourceSlice %s %s/%s/%s taint #%d", newObject(tt.slice.slice), tt.slice.slice.Spec.Driver, tt.slice.slice.Spec.Pool.Name, tt.slice.deviceName, tt.slice.taintIndex)
}

func (tt trackedTaint) Compare(b trackedTaint) int {
	// Rules first...
	if tt.rule != nil && b.rule == nil {
		return -1
	}
	if tt.rule == nil && b.rule != nil {
		return 1
	}
	if tt.rule != nil {
		// By rule name.
		return strings.Compare(tt.rule.Name, b.rule.Name)
	}

	if tt.slice.slice != nil && b.slice.slice != nil {
		// Sort by driver/pool/device, then taint index.
		if cmp := strings.Compare(tt.slice.slice.Spec.Driver, b.slice.slice.Spec.Driver); cmp != 0 {
			return cmp
		}
		if cmp := strings.Compare(tt.slice.slice.Spec.Pool.Name, b.slice.slice.Spec.Pool.Name); cmp != 0 {
			return cmp
		}
		if cmp := strings.Compare(tt.slice.deviceName, b.slice.deviceName); cmp != 0 {
			return cmp
		}
		return tt.slice.taintIndex - b.slice.taintIndex
	}

	// Both empty? Either way, we cannot compare further.
	return 0
}

// sliceDeviceTaint references one taint entry in a ResourceSlice device.
type sliceDeviceTaint struct {
	slice      *resourceapi.ResourceSlice
	deviceName string
	taintIndex int
}

// workItem is stored in a workqueue and describes some piece of work which
// needs to be done.
type workItem struct {
	// podRef, if not empty, references a pod which may need to be deleted.
	//
	// Controller.deletePodAt is the source of truth for if and when the pod really needs to be removed.
	podRef tainteviction.NamespacedObject

	// ruleRef, if not empty, is a DeviceTaintRule whose status may have to be updated.
	//
	// The initial update is done as quickly as possible to give immediate feedback,
	// then following updates are done at regular intervals (see ruleStatusPeriod).
	ruleRef tainteviction.NamespacedObject
}

func workItemForRule(rule *resourcealpha.DeviceTaintRule) workItem {
	return workItem{ruleRef: tainteviction.NamespacedObject{NamespacedName: types.NamespacedName{Name: rule.Name}, UID: rule.UID}}
}

// maybeDeletePod checks whether the pod needs to be deleted now and if so, does it.
// Three results are possible:
// - an error if anything goes wrong and the operation needs to be repeated
// - a positive delay if the operation needs to be repeated in the future
// - a zero delay if the deletion is done or no longer necessary
func (tc *Controller) maybeDeletePod(ctx context.Context, podRef tainteviction.NamespacedObject) (againAfter time.Duration, finalErr error) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "pod", podRef)

	// We must not hold this mutex while doing blocking API calls.
	tc.mutex.Lock()
	tc.maybeDeletePodCount++
	eviction, ok := tc.deletePodAt[podRef]
	tc.mutex.Unlock()
	logger.V(5).Info("Processing pod deletion work item", "active", ok, "eviction", eviction)

	if !ok {
		logger.V(5).Info("Work item for pod deletion obsolete, nothing to do")
		return 0, nil
	}

	now := time.Now()
	againAfter = eviction.when.Sub(now)
	if againAfter > 0 {
		// Not yet. Maybe the fireAt time got updated.
		return againAfter, nil
	}

	defer func() {
		if finalErr == nil {
			// Forget the deletion time, we are done.
			tc.mutex.Lock()
			delete(tc.deletePodAt, podRef)
			tc.mutex.Unlock()
		}
	}()

	err := tc.addConditionAndDeletePod(ctx, podRef)
	if apierrors.IsNotFound(err) {
		// Not a problem, the work is done.
		// But we didn't do it, so don't
		// bump the metric.
		return 0, nil
	}
	if err != nil {
		return 0, err
	}

	podDeletionLatency := time.Since(eviction.when.Time)
	logger.V(2).Info("Evicted pod by deleting it", "latency", podDeletionLatency, "reason", eviction.reason)
	tc.metrics.PodDeletionsTotal.Inc()
	tc.metrics.PodDeletionsLatency.Observe(float64(podDeletionLatency.Seconds()))
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	for _, reason := range eviction.reason {
		if reason.rule != nil {
			stats := tc.taintRuleStats[reason.rule.UID]
			stats.numEvictedPods++
			tc.taintRuleStats[reason.rule.UID] = stats

			// Ensure that the status gets updated eventually.
			// Doing this immediately is not useful because
			// it would just race with the informers update
			// (rule status reads from cache!).
			tc.workqueue.AddAfter(workItemForRule(reason.rule), ruleStatusPeriod)
		}
	}

	return 0, nil
}

func (tc *Controller) addConditionAndDeletePod(ctx context.Context, podRef tainteviction.NamespacedObject) (err error) {
	pod, err := tc.client.CoreV1().Pods(podRef.Namespace).Get(ctx, podRef.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	if pod.UID != podRef.UID {
		// This special error suppresses event logging in our caller and prevents further retries.
		// We can stop because the pod we were meant to evict is already gone and happens to
		// be replaced by some other pod which reuses the same name.
		return apierrors.NewNotFound(v1.SchemeGroupVersion.WithResource("pods").GroupResource(), pod.Name)
	}

	if pod.DeletionTimestamp != nil {
		// Already deleted, no need to evict.
		return nil
	}

	// Emit the event only if we are actually doing something.
	tc.emitPodDeletionEvent(podRef)

	newStatus := pod.Status.DeepCopy()
	updated := apipod.UpdatePodCondition(newStatus, &v1.PodCondition{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  "DeletionByDeviceTaintManager",
		Message: "Device Taint manager: deleting due to NoExecute taint",
	})
	if updated {
		if _, _, _, err := utilpod.PatchPodStatus(ctx, tc.client, pod.Namespace, pod.Name, pod.UID, pod.Status, *newStatus); err != nil {
			return err
		}
	}
	// Unlikely, but it could happen that the pod we got above got replaced with
	// another pod using the same name in the meantime. Include a precondition
	// to prevent that race. This delete attempt then fails and the next one detects
	// the new pod and stops retrying.
	return tc.client.CoreV1().Pods(podRef.Namespace).Delete(ctx, podRef.Name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &podRef.UID,
		},
	})
}

func (tc *Controller) maybeUpdateRuleStatus(ctx context.Context, ruleRef tainteviction.NamespacedObject) (time.Duration, error) {
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "deviceTaintRule", ruleRef)
	logger.V(5).Info("Processing DeviceTaintRule status work item")

	tc.mutex.Lock()
	defer tc.mutex.Unlock()

	rule, err := tc.ruleLister.Get(ruleRef.Name)
	if apierrors.IsNotFound(err) {
		logger.V(5).Info("DeviceTaintRule got deleted, removing from work queue")
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("get DeviceTaintRule %s: %w", ruleRef.Name, err)
	}
	if rule.UID != ruleRef.UID {
		logger.V(5).Info("DeviceTaintRule got replaced, removing old one from work queue")
		return 0, nil
	}

	// Already set?
	index := slices.IndexFunc(rule.Status.Conditions, func(condition metav1.Condition) bool {
		return condition.Type == resourcealpha.DeviceTaintConditionEvictionInProgress
	})

	// LastTransitionTime gets bumped each time we make any change to the condition,
	// even if it is only a change of the message. We use this to track when it is
	// time for another update.
	//
	// This is intentionally checked before counting pending pods because that might
	// be expensive. The effect is that "eviction in progress" gets set to false
	// only with a certain delay instead of immediately after deleting the last
	// pod.
	var existingCondition metav1.Condition
	now := metav1.Now()
	if index >= 0 {
		existingCondition = rule.Status.Conditions[index]
		since := now.Time.Sub(existingCondition.LastTransitionTime.Time)
		if existingCondition.ObservedGeneration == rule.Generation &&
			since < ruleStatusPeriod {
			// Don't update quite yet.
			return ruleStatusPeriod - since, nil
		}
	}

	// Checking all pods might be expensive. Only do it if really needed.
	var numTaintedSliceDevices, numTaintedAllocatedDevices, numPendingPods, numPendingNamespaces int64
	switch rule.Spec.Taint.Effect {
	case resourcealpha.DeviceTaintEffectNone:
		// Temporarily change the effect from None to NoExecute to simulate.
		// We pretend to do that through informer events. We hold the lock,
		// so there is no race with real informer events or other goroutines.
		//
		// To avoid having a lasting impact on the real controller instance
		// we make a temporary copy.
		ruleEvict := rule.DeepCopy()
		ruleEvict.Spec.Taint.Effect = resourcealpha.DeviceTaintEffectNoExecute
		tc := &Controller{
			logger:          klog.LoggerWithName(logger, "simulation"),
			podLister:       tc.podLister,
			ruleLister:      nil, // Replaced by simulateRule.
			deletePodAt:     make(map[tainteviction.NamespacedObject]evictionAndReason),
			allocatedClaims: maps.Clone(tc.allocatedClaims),
			pools:           tc.pools,
			simulateRule:    ruleEvict,
			// TODO: stub implementation
			workqueue: workqueue.NewTypedRateLimitingQueueWithConfig(workqueue.DefaultTypedControllerRateLimiter[workItem](), workqueue.TypedRateLimitingQueueConfig[workItem]{}),
		}
		defer tc.workqueue.ShutDown()

		tc.handleRuleChange(rule, ruleEvict)
		numPendingPods, numPendingNamespaces, err = tc.countPendingPods(rule)
		numTaintedSliceDevices, numTaintedAllocatedDevices = tc.countTaintedDevices(rule)
	case resourcealpha.DeviceTaintEffectNoExecute:
		numPendingPods, numPendingNamespaces, err = tc.countPendingPods(rule)
	default:
		err = nil
	}
	if err != nil {
		return 0, fmt.Errorf("determine pending pods: %w", err)
	}

	// Some fields are tentative and get updated below.
	newCondition := metav1.Condition{
		Type:               resourcealpha.DeviceTaintConditionEvictionInProgress,
		Status:             metav1.ConditionFalse,
		Reason:             string("Effect" + rule.Spec.Taint.Effect),
		ObservedGeneration: rule.Generation,
		LastTransitionTime: existingCondition.LastTransitionTime, // To avoid a false "is different" in the comparison, gets updated later.
	}
	switch rule.Spec.Taint.Effect {
	case resourcealpha.DeviceTaintEffectNoExecute:
		switch {
		case numPendingPods > 0:
			newCondition.Reason = "PodsPendingEviction"
			newCondition.Status = metav1.ConditionTrue
			if numPendingPods == 1 {
				newCondition.Message = "1 pod needs to be evicted in "
			} else {
				newCondition.Message = fmt.Sprintf("%d pods need to be evicted in ", numPendingPods)
			}
			if numPendingNamespaces == 1 {
				newCondition.Message += "1 namespace."
			} else {
				newCondition.Message += fmt.Sprintf("%d different namespaces.", numPendingNamespaces)
			}
		case tc.taintRuleStats[rule.UID].numEvictedPods > 0:
			newCondition.Reason = "Completed"
		default:
			newCondition.Reason = "NotStarted"
		}
	case resourcealpha.DeviceTaintEffectNone:
		newCondition.Reason = "NoEffect"
		if numTaintedSliceDevices == 1 {
			newCondition.Message += "1 published device selected. "
		} else {
			newCondition.Message += fmt.Sprintf("%d published devices selected. ", numTaintedSliceDevices)
		}
		if numTaintedAllocatedDevices == 1 {
			newCondition.Message += "1 allocated device selected. "
		} else {
			newCondition.Message += fmt.Sprintf("%d allocated devices selected. ", numTaintedAllocatedDevices)
		}
		if numPendingPods == 1 {
			newCondition.Message += "1 pod would be evicted in "
		} else {
			newCondition.Message += fmt.Sprintf("%d pods would be evicted in ", numPendingPods)
		}
		if numPendingNamespaces == 1 {
			newCondition.Message += "1 namespace "
		} else {
			newCondition.Message += fmt.Sprintf("%d different namespaces ", numPendingNamespaces)
		}
		newCondition.Message += "if the effect was NoExecute. This information will not be updated again. Recreate the DeviceTaintRule to trigger an update."
	default:
		newCondition.Reason = "OtherEffect"
		newCondition.Message = "Eviction only happens for the NoExecute effect."
	}
	if numEvictedPods := tc.taintRuleStats[rule.UID].numEvictedPods; numEvictedPods > 0 {
		if newCondition.Message != "" {
			newCondition.Message += " "
		}
		if numEvictedPods == 1 {
			newCondition.Message += "1 pod "
		} else {
			newCondition.Message += fmt.Sprintf("%d pods ", numEvictedPods)
		}
		newCondition.Message += "evicted since starting the controller."
	}

	if newCondition != existingCondition {
		newCondition.LastTransitionTime = now
		logger.V(4).Info("Calculated new condition", "condition", newCondition)

		// Apply the new condition, but only if the UID matches.
		ruleAC := resourceac.DeviceTaintRule(rule.Name).WithUID(rule.UID).WithStatus(resourceac.DeviceTaintRuleStatus().WithConditions(&metav1ac.ConditionApplyConfiguration{
			Type:               &newCondition.Type,
			Status:             &newCondition.Status,
			Reason:             &newCondition.Reason,
			Message:            &newCondition.Message,
			ObservedGeneration: &newCondition.ObservedGeneration,
			LastTransitionTime: &newCondition.LastTransitionTime,
		}))
		if _, err := tc.client.ResourceV1alpha3().DeviceTaintRules().ApplyStatus(ctx, ruleAC, metav1.ApplyOptions{FieldManager: tc.name, Force: true}); err != nil {
			return 0, fmt.Errorf("add condition to DeviceTaintRule status: %w", err)
		}
	}

	// No further updates needed until some more pods get evicted or ready for eviction.
	return 0, nil
}

func (tc *Controller) countPendingPods(rule *resourcealpha.DeviceTaintRule) (int64, int64, error) {
	pods, err := tc.podLister.List(labels.Everything())
	if err != nil {
		return -1, -1, fmt.Errorf("list pod: %w", err)
	}

	namespaces := sets.New[string]()
	var numPendingPods int64
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		eviction := tc.podEvictionTime(pod)
		if eviction == nil {
			continue
		}
		for _, reason := range eviction.reason {
			if reason.rule != nil &&
				reason.rule.UID == rule.UID {
				numPendingPods++
				namespaces.Insert(pod.Namespace)
			}
		}
	}

	return numPendingPods, int64(namespaces.Len()), nil
}

// countTaintedDevices determines the number of devices in slices matching the rule and
// the number of allocated devices matching the rule.
func (tc *Controller) countTaintedDevices(rule *resourcealpha.DeviceTaintRule) (numTaintedSliceDevices int64, numTaintedAllocatedDevices int64) {
	for poolID, pool := range tc.pools {
		for _, slice := range pool.slices {
			if slice.Spec.Pool.Generation != pool.maxGeneration {
				continue
			}
			for _, device := range slice.Spec.Devices {
				if ruleMatchesDevice(rule, poolID.driverName, poolID.poolName, device.Name) {
					numTaintedSliceDevices++
				}
			}
		}
	}

	for _, claim := range tc.allocatedClaims {
		for _, allocatedDevice := range claim.Status.Allocation.Devices.Results {
			if ruleMatchesDevice(rule, allocatedDevice.Driver, allocatedDevice.Pool, allocatedDevice.Device) {
				numTaintedAllocatedDevices++
			}
		}
	}

	return
}

// New creates a new Controller that will use passed clientset to communicate with the API server.
// Spawns no goroutines. That happens in Run.
func New(c clientset.Interface, podInformer coreinformers.PodInformer, claimInformer resourceinformers.ResourceClaimInformer, sliceInformer resourceinformers.ResourceSliceInformer, ruleInformer resourcealphainformers.DeviceTaintRuleInformer, classInformer resourceinformers.DeviceClassInformer, controllerName string) *Controller {
	metrics.Register() // It would be nicer to pass the controller name here, but that probably would break generating https://kubernetes.io/docs/reference/instrumentation/metrics.

	tc := &Controller{
		name: controllerName,

		client:          c,
		podInformer:     podInformer,
		podLister:       podInformer.Lister(),
		claimInformer:   claimInformer,
		sliceInformer:   sliceInformer,
		classInformer:   classInformer,
		deletePodAt:     make(map[tainteviction.NamespacedObject]evictionAndReason),
		allocatedClaims: make(map[types.NamespacedName]allocatedClaim),
		pools:           make(map[poolID]pool),
		taintRuleStats:  make(map[types.UID]taintRuleStats),
		// Instantiate all informers now to ensure that they get started.
		haveSynced: []cache.InformerSynced{
			podInformer.Informer().HasSynced,
			claimInformer.Informer().HasSynced,
			sliceInformer.Informer().HasSynced,
			classInformer.Informer().HasSynced,
		},
		metrics: metrics.Global,
	}

	// The informer for DeviceTaintRules only gets instantiated if the corresponding
	// feature is enabled. If disabled, nothings is done with (eviction) or for (status)
	// any DeviceTaintRule.
	if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules) {
		tc.ruleInformer = ruleInformer
		tc.ruleLister = ruleInformer.Lister()
		tc.haveSynced = append(tc.haveSynced, ruleInformer.Informer().HasSynced)
	}

	return tc
}

// Run starts the controller which will run until the context is done.
// An error is returned for startup problems.
func (tc *Controller) Run(ctx context.Context, numWorkers int) error {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", tc.name, "numWorkers", numWorkers)
	defer logger.Info("Shut down controller", "controller", tc.name, "reason", context.Cause(ctx))
	tc.logger = logger

	var wg sync.WaitGroup
	defer wg.Wait()

	// Doing debug logging?
	if loggerV := logger.V(6); loggerV.Enabled() {
		tc.eventLogger = &loggerV
	}

	// Delayed construction of broadcaster because it spawns goroutines.
	// tc.recorder.Eventf is a local in-memory operation which never
	// blocks, so it is safe to call from an event handler. The
	// actual API calls then happen in those spawned goroutines.
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	tc.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: tc.name}).WithLogger(logger)
	defer eventBroadcaster.Shutdown()

	queueLogger := klog.LoggerWithName(logger, "workqueue")
	delayingQueue := workqueue.NewTypedDelayingQueueWithConfig(workqueue.TypedDelayingQueueConfig[workItem]{
		Logger: &queueLogger,
		Name:   tc.name,
	})
	tc.workqueue = workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[workItem](),
		workqueue.TypedRateLimitingQueueConfig[workItem]{
			Name:          tc.name,
			DelayingQueue: delayingQueue,
		},
	)
	defer func() {
		logger.V(3).Info("Shutting down work queue")
		tc.workqueue.ShutDown()
	}()

	// Start events processing pipeline.
	eventBroadcaster.StartStructuredLogging(3)
	if tc.client != nil {
		logger.Info("Sending events to api server")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: tc.client.CoreV1().Events("")})
	} else {
		logger.Error(nil, "kubeClient is nil", "controller", tc.name)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	defer eventBroadcaster.Shutdown()

	claimHandler, err := tc.claimInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			claim, ok := obj.(*resourceapi.ResourceClaim)
			if !ok {
				logger.Error(nil, "Expected ResourceClaim", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleClaimChange(nil, claim)
		},
		UpdateFunc: func(oldObj, newObj any) {
			oldClaim, ok := oldObj.(*resourceapi.ResourceClaim)
			if !ok {
				logger.Error(nil, "Expected ResourceClaim", "actual", fmt.Sprintf("%T", oldObj))
				return
			}
			newClaim, ok := newObj.(*resourceapi.ResourceClaim)
			if !ok {
				logger.Error(nil, "Expected ResourceClaim", "actual", fmt.Sprintf("%T", newObj))
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleClaimChange(oldClaim, newClaim)
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			claim, ok := obj.(*resourceapi.ResourceClaim)
			if !ok {
				logger.Error(nil, "Expected ResourceClaim", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleClaimChange(claim, nil)
		},
	})
	if err != nil {
		return fmt.Errorf("adding claim event handler:%w", err)
	}
	defer func() {
		_ = tc.claimInformer.Informer().RemoveEventHandler(claimHandler)
	}()
	tc.haveSynced = append(tc.haveSynced, claimHandler.HasSynced)

	podHandler, err := tc.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				logger.Error(nil, "Expected Pod", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handlePodChange(nil, pod)
		},
		UpdateFunc: func(oldObj, newObj any) {
			oldPod, ok := oldObj.(*v1.Pod)
			if !ok {
				logger.Error(nil, "Expected Pod", "actual", fmt.Sprintf("%T", oldObj))
				return
			}
			newPod, ok := newObj.(*v1.Pod)
			if !ok {
				logger.Error(nil, "Expected Pod", "actual", fmt.Sprintf("%T", newObj))
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handlePodChange(oldPod, newPod)
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			pod, ok := obj.(*v1.Pod)
			if !ok {
				logger.Error(nil, "Expected Pod", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handlePodChange(pod, nil)
		},
	})
	if err != nil {
		return fmt.Errorf("adding pod event handler: %w", err)
	}
	defer func() {
		_ = tc.podInformer.Informer().RemoveEventHandler(podHandler)
	}()
	tc.haveSynced = append(tc.haveSynced, podHandler.HasSynced)

	if tc.ruleInformer != nil {
		ruleHandler, err := tc.ruleInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj any) {
				rule, ok := obj.(*resourcealpha.DeviceTaintRule)
				if !ok {
					logger.Error(nil, "Expected DeviceTaintRule", "actual", fmt.Sprintf("%T", obj))
					return
				}
				tc.mutex.Lock()
				defer tc.mutex.Unlock()
				tc.handleRuleChange(nil, rule)
			},
			UpdateFunc: func(oldObj, newObj any) {
				oldRule, ok := oldObj.(*resourcealpha.DeviceTaintRule)
				if !ok {
					logger.Error(nil, "Expected DeviceTaintRule", "actual", fmt.Sprintf("%T", oldObj))
					return
				}
				newRule, ok := newObj.(*resourcealpha.DeviceTaintRule)
				if !ok {
					logger.Error(nil, "Expected DeviceTaintRule", "actual", fmt.Sprintf("%T", newObj))
				}
				tc.mutex.Lock()
				defer tc.mutex.Unlock()
				tc.handleRuleChange(oldRule, newRule)
			},
			DeleteFunc: func(obj any) {
				if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
					obj = tombstone.Obj
				}
				rule, ok := obj.(*resourcealpha.DeviceTaintRule)
				if !ok {
					logger.Error(nil, "Expected DeviceTaintRule", "actual", fmt.Sprintf("%T", obj))
					return
				}
				tc.mutex.Lock()
				defer tc.mutex.Unlock()
				tc.handleRuleChange(rule, nil)
			},
		})
		if err != nil {
			return fmt.Errorf("adding DeviceTaintRule event handler: %w", err)
		}
		defer func() {
			_ = tc.ruleInformer.Informer().RemoveEventHandler(ruleHandler)
		}()
		tc.haveSynced = append(tc.haveSynced, ruleHandler.HasSynced)
	}

	sliceHandler, err := tc.sliceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleSliceChange(nil, slice)
		},
		UpdateFunc: func(oldObj, newObj any) {
			oldSlice, ok := oldObj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", oldObj))
				return
			}
			newSlice, ok := newObj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", newObj))
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleSliceChange(oldSlice, newSlice)
		},
		DeleteFunc: func(obj any) {
			// No need to check for DeletedFinalStateUnknown here, the resourceslicetracker doesn't use that.
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", obj))
				return
			}
			tc.mutex.Lock()
			defer tc.mutex.Unlock()
			tc.handleSliceChange(slice, nil)
		},
	})
	if err != nil {
		return fmt.Errorf("adding slice event handler: %w", err)
	}
	defer func() {
		_ = tc.sliceInformer.Informer().RemoveEventHandler(sliceHandler)
	}()
	tc.haveSynced = append(tc.haveSynced, sliceHandler.HasSynced)

	if !cache.WaitForNamedCacheSyncWithContext(ctx, tc.haveSynced...) {
		// If we get here, the caller canceled the context. This is not an error.
		return nil
	}
	logger.V(1).Info("Underlying informers have synced")
	tc.hasSynced.Store(1)

	for i := range numWorkers {
		wg.Go(func() {
			tc.worker(klog.NewContext(ctx, klog.LoggerWithName(queueLogger, fmt.Sprintf("worker-%d", i))))
		})
	}

	<-ctx.Done()
	return nil
}

// evictPod ensures that the pod gets evicted at the specified time.
// It doesn't block.
func (tc *Controller) evictPod(podRef tainteviction.NamespacedObject, eviction evictionAndReason) {
	tc.deletePodAt[podRef] = eviction
	now := time.Now()
	tc.workqueue.AddAfter(workItem{podRef: podRef}, eviction.when.Sub(now))

	if tc.evictPodHook != nil {
		tc.evictPodHook(podRef, eviction)
	}
}

// cancelEvict cancels eviction set up with evictPod earlier.
// Idempotent, returns false if there was nothing to cancel.
func (tc *Controller) cancelEvict(podRef tainteviction.NamespacedObject) bool {
	_, ok := tc.deletePodAt[podRef]
	if !ok {
		// Nothing to cancel.
		return false
	}
	delete(tc.deletePodAt, podRef)

	if tc.cancelEvictHook != nil {
		tc.cancelEvictHook(podRef)
	}

	// Cannot remove from a work queue. The worker will detect that the entry is obsolete by checking deletePodAt.
	return true
}

// worker blocks until the workqueue is shut down.
// Cancellation of the context only aborts on-going work.
func (tc *Controller) worker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	defer utilruntime.HandleCrashWithLogger(logger)

	for {
		item, shutdown := tc.workqueue.Get()
		if shutdown {
			return
		}

		func() {
			defer tc.workqueue.Done(item)

			againAfter, err := tc.handleWork(ctx, item)
			switch {
			case err != nil:
				logger.V(3).Info("Processing work item failed, will retry", "err", err)
				tc.workqueue.AddRateLimited(item)
			case againAfter > 0:
				logger.V(5).Info("Checking work item again later", "delay", againAfter)
				tc.workqueue.AddAfter(item, againAfter)
			default:
				tc.workqueue.Forget(item)
			}
		}()
	}
}

func (tc *Controller) handleWork(ctx context.Context, item workItem) (time.Duration, error) {
	if item.podRef.Name != "" {
		return tc.maybeDeletePod(ctx, item.podRef)
	}
	return tc.maybeUpdateRuleStatus(ctx, item.ruleRef)
}

func (tc *Controller) handleClaimChange(oldClaim, newClaim *resourceapi.ResourceClaim) {
	claim := newClaim
	if claim == nil {
		claim = oldClaim
	}
	name := newNamespacedName(claim)
	if tc.eventLogger != nil {
		// This is intentionally very verbose for debugging.
		tc.eventLogger.Info("ResourceClaim changed", "claimObject", name, "oldClaim", klog.Format(oldClaim), "newClaim", klog.Format(newClaim), "diff", diff.Diff(oldClaim, newClaim))
	}

	// Deleted?
	if newClaim == nil {
		delete(tc.allocatedClaims, name)
		tc.handlePods(claim)
		return
	}

	// Added?
	if oldClaim == nil {
		if claim.Status.Allocation == nil {
			return
		}
		tc.allocatedClaims[name] = allocatedClaim{
			ResourceClaim: claim,
			eviction:      tc.claimEvictionTime(claim),
		}
		tc.handlePods(claim)
		return
	}

	// If we have two claims, the UID might still be different. Unlikely, but not impossible...
	// Treat this like a remove + add.
	if oldClaim.UID != newClaim.UID {
		tc.handleClaimChange(oldClaim, nil)
		tc.handleClaimChange(nil, newClaim)
		return
	}

	syncBothClaims := func() {
		// ReservedFor may have changed. If it did, sync both old and new lists,
		// otherwise only once (same list).
		if !slices.Equal(oldClaim.Status.ReservedFor, newClaim.Status.ReservedFor) {
			tc.handlePods(oldClaim)
			tc.handlePods(newClaim)
		} else {
			tc.handlePods(claim)
		}
	}

	// Allocation added?
	if oldClaim.Status.Allocation == nil && newClaim.Status.Allocation != nil {
		tc.allocatedClaims[name] = allocatedClaim{
			ResourceClaim: claim,
			eviction:      tc.claimEvictionTime(claim),
		}
		syncBothClaims()
		return
	}

	// Allocation removed?
	if oldClaim.Status.Allocation != nil && newClaim.Status.Allocation == nil {
		delete(tc.allocatedClaims, name)
		syncBothClaims()
		return
	}

	// Allocated before and after?
	if claim.Status.Allocation != nil {
		// The Allocation is immutable, so we don't need to recompute the eviction
		// time. Storing the newer claim is enough.
		tc.allocatedClaims[name] = allocatedClaim{
			ResourceClaim: claim,
			eviction:      tc.allocatedClaims[name].eviction,
		}
		syncBothClaims()
		return
	}

	// If we get here, nothing changed.
}

// claimEvictionTime returns the earliest TimeAdded of any NoExecute taint in any allocated device
// unless that taint is tolerated, nil if none. May only be called for allocated claims.
func (tc *Controller) claimEvictionTime(claim *resourceapi.ResourceClaim) *evictionAndReason {
	var when *metav1.Time
	var taints sets.Set[trackedTaint]

	allocation := claim.Status.Allocation
	for _, allocatedDevice := range allocation.Devices.Results {
		id := poolID{driverName: allocatedDevice.Driver, poolName: allocatedDevice.Pool}
		slice, device := tc.pools[id].getDevice(allocatedDevice.Device)

	nextTaint:
		for taint := range tc.allEvictingDeviceTaints(allocatedDevice, slice, device) {
			newEvictionTime := taint.deviceTaint().TimeAdded
			haveToleration := false
			tolerationSeconds := int64(math.MaxInt64)
			for _, toleration := range allocatedDevice.Tolerations {
				if toleration.Effect == resourceapi.DeviceTaintEffectNoExecute &&
					resourceclaim.ToleratesTaint(toleration, *taint.deviceTaint()) {
					if toleration.TolerationSeconds == nil {
						// Tolerate forever -> ignore taint.
						continue nextTaint
					}
					newTolerationSeconds := *toleration.TolerationSeconds
					if newTolerationSeconds < 0 {
						newTolerationSeconds = 0
					}
					if newTolerationSeconds < tolerationSeconds {
						tolerationSeconds = newTolerationSeconds
					}
					haveToleration = true
				}
			}
			if haveToleration {
				newEvictionTime = &metav1.Time{Time: newEvictionTime.Add(time.Duration(tolerationSeconds) * time.Second)}
			}
			if taints == nil {
				taints = sets.New[trackedTaint]()
			}
			taints.Insert(taint)
			if when == nil {
				when = newEvictionTime
				tc.logger.V(5).Info("Claim is affected by device taint", "claim", klog.KObj(claim), "device", allocatedDevice, "taint", taint, "evictionTime", when)
				continue
			}
			if newEvictionTime != nil && newEvictionTime.Before(when) {
				when = newEvictionTime
				tc.logger.V(5).Info("Claim is affected by device taint", "claim", klog.KObj(claim), "device", allocatedDevice, "taint", taint, "evictionTime", when)
			}
		}
	}

	if when == nil {
		return nil
	}
	eviction := &evictionAndReason{when: *when, reason: taints.UnsortedList()}
	slices.SortFunc(eviction.reason, func(a, b trackedTaint) int { return a.Compare(b) })
	return eviction
}

// allEvictingDeviceTaints allows iterating over all DeviceTaintRules with NoExecute effect which affect the allocated device.
// A taint may come from either the ResourceSlice informer (not the tracker!) or from a DeviceTaintRule, but not both.
func (tc *Controller) allEvictingDeviceTaints(allocatedDevice resourceapi.DeviceRequestAllocationResult, slice *resourceapi.ResourceSlice, device *resourceapi.Device) iter.Seq[trackedTaint] {
	var rules []*resourcealpha.DeviceTaintRule
	var err error
	if tc.ruleLister != nil {
		rules, err = tc.ruleLister.List(labels.Everything())
		if err != nil {
			// TODO: instead of listing and handling an error, keep track of rules in the informer event handler?
			utilruntime.HandleErrorWithLogger(tc.logger, err, "Local cache failed to list DeviceTaintRules")
			return func(yield func(trackedTaint) bool) {}
		}
	}

	if tc.simulateRule != nil {
		// Mix the rule for which we simulate EffectNoExecute into the set of
		// rules which will be evaluated. Typically this is the only rule
		// during simulation.
		rules = append(rules, tc.simulateRule)
	}

	return func(yield func(trackedTaint) bool) {
		if device != nil {
			for i := range device.Taints {
				taint := &device.Taints[i]
				if taint.Effect != resourceapi.DeviceTaintEffectNoExecute {
					continue
				}
				if !yield(trackedTaint{slice: sliceDeviceTaint{slice: slice, deviceName: device.Name, taintIndex: i}}) {
					return
				}
			}
		}

		for _, rule := range rules {
			if rule.Spec.Taint.Effect != resourcealpha.DeviceTaintEffectNoExecute ||
				!ruleMatchesDevice(rule, allocatedDevice.Driver, allocatedDevice.Pool, allocatedDevice.Device) {
				continue
			}

			if !yield(trackedTaint{rule: rule}) {
				return
			}
		}
	}
}

func ruleMatchesDevice(rule *resourcealpha.DeviceTaintRule, driverName, poolName, deviceName string) bool {
	selector := rule.Spec.DeviceSelector
	if selector == nil {
		return false
	}
	if selector.Driver != nil && *selector.Driver != driverName ||
		selector.Pool != nil && *selector.Pool != poolName ||
		selector.Device != nil && *selector.Device != deviceName {
		return false
	}
	return true
}

func (tc *Controller) handleRuleChange(oldRule, newRule *resourcealpha.DeviceTaintRule) {
	rule := newRule
	if rule == nil {
		rule = oldRule
	}
	name := newNamespacedName(rule)
	if tc.eventLogger != nil {
		// This is intentionally very verbose for debugging.
		tc.eventLogger.Info("DeviceTaintRule changed", "ruleObject", name, "oldRule", klog.Format(oldRule), "newRule", klog.Format(newRule), "diff", diff.Diff(oldRule, newRule))
	}

	if oldRule == nil {
		// Update the status at least once.
		tc.workqueue.Add(workItemForRule(newRule))
	}

	if newRule == nil {
		// Clean up to avoid memory leak.
		delete(tc.taintRuleStats, oldRule.UID)
		// Removal from the work queue is handled when a worker handles the work item.
		// A work queue does not support canceling work.
	}

	if oldRule != nil &&
		newRule != nil &&
		oldRule.UID == newRule.UID &&
		apiequality.Semantic.DeepEqual(&oldRule.Spec, &newRule.Spec) {
		return
	}

	// Rule spec changes should be rare. Simply do a brute-force re-evaluation of all allocated claims.
	for name, oldAllocatedClaim := range tc.allocatedClaims {
		newAllocatedClaim := allocatedClaim{
			ResourceClaim: oldAllocatedClaim.ResourceClaim,
			eviction:      tc.claimEvictionTime(oldAllocatedClaim.ResourceClaim),
		}
		tc.allocatedClaims[name] = newAllocatedClaim
		if !newAllocatedClaim.eviction.equal(oldAllocatedClaim.eviction) {
			tc.handlePods(newAllocatedClaim.ResourceClaim)
		}
	}
}

func (tc *Controller) handleSliceChange(oldSlice, newSlice *resourceapi.ResourceSlice) {
	slice := newSlice
	if slice == nil {
		slice = oldSlice
	}
	poolID := poolID{
		driverName: slice.Spec.Driver,
		poolName:   slice.Spec.Pool.Name,
	}
	if tc.eventLogger != nil {
		// This is intentionally very verbose for debugging.
		tc.eventLogger.Info("ResourceSlice changed", "pool", poolID, "oldSlice", klog.Format(oldSlice), "newSlice", klog.Format(newSlice), "diff", diff.Diff(oldSlice, newSlice))
	}

	// Determine old and new device taints. Only devices
	// where something changes trigger additional checks for claims
	// using them.
	p := tc.pools[poolID]
	oldDeviceTaints := p.getTaintedDevices()
	p.removeSlice(oldSlice)
	p.addSlice(newSlice)
	if len(p.slices) == 0 {
		delete(tc.pools, poolID)
	} else {
		tc.pools[poolID] = p
	}
	newDeviceTaints := p.getTaintedDevices()

	// Now determine differences. This depends on both slices having been sorted
	// by device name.
	if len(oldDeviceTaints) == 0 && len(newDeviceTaints) == 0 {
		// Both empty, no changes.
		return
	}
	modifiedDevices := sets.New[string]()
	o, n := 0, 0
	for o < len(oldDeviceTaints) || n < len(newDeviceTaints) {
		// Iterate over devices in both slices with the same name.
		for o < len(oldDeviceTaints) && n < len(newDeviceTaints) && oldDeviceTaints[o].deviceName == newDeviceTaints[n].deviceName {
			if !apiequality.Semantic.DeepEqual(oldDeviceTaints[o].taint, newDeviceTaints[n].taint) { // TODO: hard-code the comparison?
				modifiedDevices.Insert(oldDeviceTaints[o].deviceName)
			}
			o++
			n++
		}

		// Step over old devices which were removed.
		newDeviceName := ""
		if n < len(newDeviceTaints) {
			newDeviceName = newDeviceTaints[n].deviceName
		}
		for o < len(oldDeviceTaints) && oldDeviceTaints[o].deviceName != newDeviceName {
			modifiedDevices.Insert(oldDeviceTaints[o].deviceName)
			o++
		}

		// Step over new devices which were added.
		oldDeviceName := ""
		if o < len(oldDeviceTaints) {
			oldDeviceName = oldDeviceTaints[o].deviceName
		}
		if n < len(newDeviceTaints) && newDeviceTaints[n].deviceName != oldDeviceName {
			modifiedDevices.Insert(newDeviceTaints[n].deviceName)
			n++
		}
	}

	// Now find all claims using at least one modified device,
	// update their eviction time, and handle their consuming pods.
	for name, claim := range tc.allocatedClaims {
		if !usesDevice(claim.Status.Allocation, poolID, modifiedDevices) {
			continue
		}
		newEvictionTime := tc.claimEvictionTime(claim.ResourceClaim)
		if newEvictionTime.equal(claim.eviction) {
			// No change.
			continue
		}
		claim.eviction = newEvictionTime
		tc.allocatedClaims[name] = claim
		// We could collect pods which depend on claims with changes.
		// In practice, most pods probably depend on one claim, so
		// it is probably more efficient to avoid building such a map
		// to make the common case simple.
		tc.handlePods(claim.ResourceClaim)
	}
}

func usesDevice(allocation *resourceapi.AllocationResult, pool poolID, modifiedDevices sets.Set[string]) bool {
	for _, device := range allocation.Devices.Results {
		if device.Driver == pool.driverName &&
			device.Pool == pool.poolName &&
			modifiedDevices.Has(device.Device) {
			return true
		}
	}
	return false
}

func (tc *Controller) handlePodChange(oldPod, newPod *v1.Pod) {
	pod := newPod
	if pod == nil {
		pod = oldPod
	}
	if tc.eventLogger != nil {
		// This is intentionally very verbose for debugging.
		tc.eventLogger.Info("Pod changed", "pod", klog.KObj(pod), "oldPod", klog.Format(oldPod), "newPod", klog.Format(newPod), "diff", diff.Diff(oldPod, newPod))
	}
	if newPod == nil {
		// Nothing left to do for it. No need to emit an event here, it's gone.
		tc.cancelEvict(newObject(oldPod))
		return
	}

	// Pods get updated quite frequently. There's no need
	// to check them again unless something changed regarding
	// their claims or they got scheduled.
	//
	// In particular this prevents adding the pod again
	// directly after the eviction condition got added
	// to it.
	if oldPod != nil &&
		oldPod.Spec.NodeName == newPod.Spec.NodeName &&
		apiequality.Semantic.DeepEqual(oldPod.Status.ResourceClaimStatuses, newPod.Status.ResourceClaimStatuses) {
		return
	}

	tc.handlePod(newPod)
}

func (tc *Controller) handlePods(claim *resourceapi.ResourceClaim) {
	for _, consumer := range claim.Status.ReservedFor {
		if consumer.APIGroup == "" && consumer.Resource == "pods" {
			pod, err := tc.podLister.Pods(claim.Namespace).Get(consumer.Name)
			if err != nil {
				if apierrors.IsNotFound(err) {
					return
				}
				// Should not happen.
				utilruntime.HandleErrorWithLogger(tc.logger, err, "retrieve pod from cache")
				return
			}
			if pod.UID != consumer.UID {
				// Not the pod we were looking for.
				return
			}
			tc.handlePod(pod)
		}
	}
}

func (tc *Controller) handlePod(pod *v1.Pod) {
	eviction := tc.podEvictionTime(pod)
	podRef := newObject(pod)
	if eviction == nil {
		if tc.cancelWorkWithEvent(podRef) {
			tc.logger.V(3).Info("Canceled pod eviction", "pod", podRef)
		}
		return
	}

	tc.logger.V(3).Info("Going to evict pod", "pod", podRef, "eviction", eviction)
	tc.evictPod(podRef, *eviction)

	// If any reason is because of a taint, then eviction is in progress and the status may need to be updated.
	for _, reason := range eviction.reason {
		if reason.rule != nil {
			tc.workqueue.Add(workItemForRule(reason.rule))
		}
	}
}

func (tc *Controller) podEvictionTime(pod *v1.Pod) *evictionAndReason {
	// Not scheduled yet? No need to evict.
	if pod.Spec.NodeName == "" {
		return nil
	}

	// If any claim in use by the pod is tainted such that the taint is not tolerated,
	// the pod needs to be evicted.
	var eviction *evictionAndReason
	for i := range pod.Spec.ResourceClaims {
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			// Not created yet or unsupported. Definitely not tainted.
			continue
		}
		if claimName == nil {
			// Claim not needed.
			continue
		}
		allocatedClaim, ok := tc.allocatedClaims[types.NamespacedName{Namespace: pod.Namespace, Name: *claimName}]
		if !ok {
			// Referenced, but not found or not allocated. Also not tainted.
			continue
		}
		if mustCheckOwner && resourceclaim.IsForPod(pod, allocatedClaim.ResourceClaim) != nil {
			// Claim and pod don't match. Ignore the claim.
			continue
		}
		if !resourceclaim.IsReservedForPod(pod, allocatedClaim.ResourceClaim) {
			// The pod isn't the one which is allowed and/or supposed to use the claim.
			// Perhaps that pod instance already got deleted and we are looking at its
			// replacement under the same name. Either way, ignore.
			continue
		}
		if allocatedClaim.eviction == nil {
			continue
		}
		if eviction == nil {
			// Use the new eviction time as-is.
			eviction = allocatedClaim.eviction
		} else {
			// Join reasons and figure out the new time.
			// Might not actually lead to any change.
			newEvictionTime := &evictionAndReason{
				when:   allocatedClaim.eviction.when,
				reason: slices.Clone(allocatedClaim.eviction.reason),
			}
			// Multiple reasons affecting the same pod should be so rare,
			// a simple insertion sort is fine.
			for _, reason := range eviction.reason {
				index, found := slices.BinarySearchFunc(newEvictionTime.reason, reason, func(a, b trackedTaint) int { return a.Compare(b) })
				if !found {
					newEvictionTime.reason = slices.Insert(newEvictionTime.reason, index, reason)
				}
			}
			if eviction.when.Before(&newEvictionTime.when) {
				newEvictionTime.when = eviction.when
			}
			if !eviction.equal(newEvictionTime) {
				eviction = newEvictionTime
			}
		}
	}

	return eviction
}

func (tc *Controller) cancelWorkWithEvent(podRef tainteviction.NamespacedObject) bool {
	if tc.cancelEvict(podRef) {
		tc.emitCancelPodDeletionEvent(podRef)
		return true
	}
	return false
}

func (tc *Controller) emitPodDeletionEvent(podRef tainteviction.NamespacedObject) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       podRef.Name,
		Namespace:  podRef.Namespace,
		UID:        podRef.UID,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "DeviceTaintManagerEviction", "Marking for deletion")
}

func (tc *Controller) emitCancelPodDeletionEvent(podRef tainteviction.NamespacedObject) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       podRef.Name,
		Namespace:  podRef.Namespace,
		UID:        podRef.UID,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "DeviceTaintManagerEviction", "Cancelling deletion")
}

func newNamespacedName(obj metav1.Object) types.NamespacedName {
	return types.NamespacedName{
		Namespace: obj.GetNamespace(),
		Name:      obj.GetName(),
	}
}

// TODO: replace with klog.ObjectInstance (https://github.com/kubernetes/klog/issues/422#issuecomment-3454948091).
func newObject(obj metav1.Object) tainteviction.NamespacedObject {
	if obj == nil {
		return tainteviction.NamespacedObject{}
	}
	return tainteviction.NamespacedObject{
		NamespacedName: newNamespacedName(obj),
		UID:            obj.GetUID(),
	}
}
