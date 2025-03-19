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
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/go-cmp/cmp" //nolint:depguard // Discouraged for production use (https://github.com/kubernetes/kubernetes/issues/104821) but has no good alternative for logging.

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	coreinformers "k8s.io/client-go/informers/core/v1"
	resourcealphainformers "k8s.io/client-go/informers/resource/v1alpha3"
	resourceinformers "k8s.io/client-go/informers/resource/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/klog/v2"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction/metrics"
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	utilpod "k8s.io/kubernetes/pkg/util/pod"
)

const (
	// retries is the number of times that the controller tries to delete a pod
	// that needs to be evicted.
	retries = 5
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
	taintInformer resourcealphainformers.DeviceTaintRuleInformer
	classInformer resourceinformers.DeviceClassInformer
	haveSynced    []cache.InformerSynced
	metrics       metrics.Metrics

	// evictPod ensures that the pod gets evicted at the specified time.
	// It doesn't block.
	evictPod func(pod tainteviction.NamespacedObject, fireAt time.Time)

	// cancelEvict cancels eviction set up with evictPod earlier.
	// Idempotent, returns false if there was nothing to cancel.
	cancelEvict func(pod tainteviction.NamespacedObject) bool

	// allocatedClaims holds all currently known allocated claims.
	allocatedClaims map[types.NamespacedName]allocatedClaim // A value is slightly more efficient in BenchmarkTaintUntaint (less allocations!).

	// pools indexes all slices by driver and pool name.
	pools map[poolID]pool

	hasSynced atomic.Int32
}

type poolID struct {
	driverName, poolName string
}

type pool struct {
	slices        sets.Set[*resourceapi.ResourceSlice]
	maxGeneration int64
}

// addSlice adds one slice to the pool.
func (p *pool) addSlice(slice *resourceapi.ResourceSlice) {
	if slice == nil {
		return
	}
	if p.slices == nil {
		p.slices = sets.New[*resourceapi.ResourceSlice]()
		p.maxGeneration = math.MinInt64
	}
	p.slices.Insert(slice)

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
	p.slices.Delete(slice)

	// Removing a slice might have decreased the generation to
	// that of some other slice.
	if slice.Spec.Pool.Generation == p.maxGeneration {
		maxGeneration := int64(math.MinInt64)
		for slice := range p.slices {
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
	for slice := range p.slices {
		if slice.Spec.Pool.Generation != p.maxGeneration {
			continue
		}
		for _, device := range slice.Spec.Devices {
			if device.Basic == nil {
				// Unknown device type, not supported.
				continue
			}
			for _, taint := range device.Basic.Taints {
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
func (p pool) getDevice(deviceName string) *resourceapi.BasicDevice {
	for slice := range p.slices {
		if slice.Spec.Pool.Generation != p.maxGeneration {
			continue
		}
		for _, device := range slice.Spec.Devices {
			if device.Basic == nil {
				// Unknown device type, not supported.
				continue
			}
			if device.Name == deviceName {
				return device.Basic
			}
		}
	}

	return nil
}

type taintedDevice struct {
	deviceName string
	taint      resourceapi.DeviceTaint
}

// allocatedClaim is a ResourceClaim which has an allocation result. It
// may or may not be tainted such that pods need to be evicted.
type allocatedClaim struct {
	*resourceapi.ResourceClaim

	// evictionTime, if non-nil, is the time at which pods using this claim need to be evicted.
	// This is the smallest value of all such per-device values.
	// For each device, the value is calculated as `<time of setting the taint> +
	// <toleration seconds, 0 if not set>`.
	evictionTime *metav1.Time
}

func (tc *Controller) deletePodHandler(c clientset.Interface, emitEventFunc func(tainteviction.NamespacedObject)) func(ctx context.Context, fireAt time.Time, args *tainteviction.WorkArgs) error {
	return func(ctx context.Context, fireAt time.Time, args *tainteviction.WorkArgs) error {
		klog.FromContext(ctx).Info("Deleting pod", "pod", args.Object)
		var err error
		for i := 0; i < retries; i++ {
			err = addConditionAndDeletePod(ctx, c, args.Object, &emitEventFunc)
			if apierrors.IsNotFound(err) {
				// Not a problem, the work is done.
				// But we didn't do it, so don't
				// bump the metric.
				return nil
			}
			if err == nil {
				tc.metrics.PodDeletionsTotal.Inc()
				tc.metrics.PodDeletionsLatency.Observe(float64(time.Since(fireAt).Seconds()))
				return nil
			}
			time.Sleep(10 * time.Millisecond)
		}
		return err
	}
}

func addConditionAndDeletePod(ctx context.Context, c clientset.Interface, podRef tainteviction.NamespacedObject, emitEventFunc *func(tainteviction.NamespacedObject)) (err error) {
	pod, err := c.CoreV1().Pods(podRef.Namespace).Get(ctx, podRef.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	if pod.UID != podRef.UID {
		// This special error suppresses event logging in our caller and prevents further retries.
		// We can stop because the pod we were meant to evict is already gone and happens to
		// be replaced by some other pod which reuses the same name.
		return apierrors.NewNotFound(v1.SchemeGroupVersion.WithResource("pods").GroupResource(), pod.Name)
	}

	// Emit the event only once, and only if we are actually doing something.
	if *emitEventFunc != nil {
		(*emitEventFunc)(podRef)
		*emitEventFunc = nil
	}

	newStatus := pod.Status.DeepCopy()
	updated := apipod.UpdatePodCondition(newStatus, &v1.PodCondition{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  "DeletionByDeviceTaintManager",
		Message: "Device Taint manager: deleting due to NoExecute taint",
	})
	if updated {
		if _, _, _, err := utilpod.PatchPodStatus(ctx, c, pod.Namespace, pod.Name, pod.UID, pod.Status, *newStatus); err != nil {
			return err
		}
	}
	// Unlikely, but it could happen that the pod we got above got replaced with
	// another pod using the same name in the meantime. Include a precondition
	// to prevent that race. This delete attempt then fails and the next one detects
	// the new pod and stops retrying.
	return c.CoreV1().Pods(podRef.Namespace).Delete(ctx, podRef.Name, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{
			UID: &podRef.UID,
		},
	})
}

// New creates a new Controller that will use passed clientset to communicate with the API server.
// Spawns no goroutines. That happens in Run.
func New(c clientset.Interface, podInformer coreinformers.PodInformer, claimInformer resourceinformers.ResourceClaimInformer, sliceInformer resourceinformers.ResourceSliceInformer, taintInformer resourcealphainformers.DeviceTaintRuleInformer, classInformer resourceinformers.DeviceClassInformer, controllerName string) *Controller {
	metrics.Register() // It would be nicer to pass the controller name here, but that probably would break generating https://kubernetes.io/docs/reference/instrumentation/metrics.

	tc := &Controller{
		name: controllerName,

		client:          c,
		podInformer:     podInformer,
		podLister:       podInformer.Lister(),
		claimInformer:   claimInformer,
		sliceInformer:   sliceInformer,
		taintInformer:   taintInformer,
		classInformer:   classInformer,
		allocatedClaims: make(map[types.NamespacedName]allocatedClaim),
		pools:           make(map[poolID]pool),
		// Instantiate all informers now to ensure that they get started.
		haveSynced: []cache.InformerSynced{
			podInformer.Informer().HasSynced,
			claimInformer.Informer().HasSynced,
			sliceInformer.Informer().HasSynced,
			taintInformer.Informer().HasSynced,
			classInformer.Informer().HasSynced,
		},
		metrics: metrics.Global,
	}

	return tc
}

// Run starts the controller which will run until the context is done.
func (tc *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()
	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", tc.name)
	defer logger.Info("Shutting down controller", "controller", tc.name)
	tc.logger = logger

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

	taintEvictionQueue := tainteviction.CreateWorkerQueue(tc.deletePodHandler(tc.client, tc.emitPodDeletionEvent))
	evictPod := tc.evictPod
	tc.evictPod = func(podRef tainteviction.NamespacedObject, fireAt time.Time) {
		// Only relevant for testing.
		if evictPod != nil {
			evictPod(podRef, fireAt)
		}
		taintEvictionQueue.UpdateWork(ctx, &tainteviction.WorkArgs{Object: podRef}, time.Now(), fireAt)
	}
	cancelEvict := tc.cancelEvict
	tc.cancelEvict = func(podRef tainteviction.NamespacedObject) bool {
		if cancelEvict != nil {
			cancelEvict(podRef)
		}
		return taintEvictionQueue.CancelWork(logger, podRef.NamespacedName.String())
	}

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

	// mutex serializes event processing.
	var mutex sync.Mutex

	claimHandler, _ := tc.claimInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			claim, ok := obj.(*resourceapi.ResourceClaim)
			if !ok {
				logger.Error(nil, "Expected ResourceClaim", "actual", fmt.Sprintf("%T", obj))
				return
			}
			mutex.Lock()
			defer mutex.Unlock()
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
			mutex.Lock()
			defer mutex.Unlock()
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
			mutex.Lock()
			defer mutex.Unlock()
			tc.handleClaimChange(claim, nil)
		},
	})
	defer func() {
		_ = tc.claimInformer.Informer().RemoveEventHandler(claimHandler)
	}()
	tc.haveSynced = append(tc.haveSynced, claimHandler.HasSynced)

	podHandler, _ := tc.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				logger.Error(nil, "Expected ResourcePod", "actual", fmt.Sprintf("%T", obj))
				return
			}
			mutex.Lock()
			defer mutex.Unlock()
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
			mutex.Lock()
			defer mutex.Unlock()
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
			mutex.Lock()
			defer mutex.Unlock()
			tc.handlePodChange(pod, nil)
		},
	})
	defer func() {
		_ = tc.podInformer.Informer().RemoveEventHandler(podHandler)
	}()
	tc.haveSynced = append(tc.haveSynced, podHandler.HasSynced)

	opts := resourceslicetracker.Options{
		EnableDeviceTaints: true,
		SliceInformer:      tc.sliceInformer,
		TaintInformer:      tc.taintInformer,
		ClassInformer:      tc.classInformer,
		KubeClient:         tc.client,
	}
	sliceTracker, err := resourceslicetracker.StartTracker(ctx, opts)
	if err != nil {
		logger.Info("Failed to initialize ResourceSlice tracker; device taint processing leading to Pod eviction is now paused", "err", err)
		return
	}
	tc.haveSynced = append(tc.haveSynced, sliceTracker.HasSynced)
	defer sliceTracker.Stop()

	// Wait for tracker to sync before we react to events.
	// This doesn't have to be perfect, it merely avoids unnecessary
	// work which might be done as events get emitted for intermediate
	// state.
	if !cache.WaitForNamedCacheSyncWithContext(ctx, tc.haveSynced...) {
		return
	}
	logger.V(1).Info("Underlying informers have synced")

	_, _ = sliceTracker.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", obj))
				return
			}
			mutex.Lock()
			defer mutex.Unlock()
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
			mutex.Lock()
			defer mutex.Unlock()
			tc.handleSliceChange(oldSlice, newSlice)
		},
		DeleteFunc: func(obj any) {
			// No need to check for DeletedFinalStateUnknown here, the resourceslicetracker doesn't use that.
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				logger.Error(nil, "Expected ResourceSlice", "actual", fmt.Sprintf("%T", obj))
				return
			}
			mutex.Lock()
			defer mutex.Unlock()
			tc.handleSliceChange(slice, nil)
		},
	})

	// sliceTracker.AddEventHandler blocked while delivering events for all known
	// ResourceSlices. Therefore our own state is up-to-date once we get here.
	tc.hasSynced.Store(1)

	<-ctx.Done()
}

func (tc *Controller) handleClaimChange(oldClaim, newClaim *resourceapi.ResourceClaim) {
	claim := newClaim
	if claim == nil {
		claim = oldClaim
	}
	name := newNamespacedName(claim)
	if tc.eventLogger != nil {
		// This is intentionally very verbose for debugging.
		tc.eventLogger.Info("ResourceClaim changed", "claimObject", name, "oldClaim", klog.Format(oldClaim), "newClaim", klog.Format(newClaim), "diff", cmp.Diff(oldClaim, newClaim))
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
			evictionTime:  tc.evictionTime(claim.Status.Allocation),
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
			evictionTime:  tc.evictionTime(claim.Status.Allocation),
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
			evictionTime:  tc.allocatedClaims[name].evictionTime,
		}
		syncBothClaims()
		return
	}

	// If we get here, nothing changed.
}

// evictionTime returns the earliest TimeAdded of any NoExecute taint in any allocated device
// unless that taint is tolerated, nil if none.
func (tc *Controller) evictionTime(allocation *resourceapi.AllocationResult) *metav1.Time {
	var evictionTime *metav1.Time

	for _, allocatedDevice := range allocation.Devices.Results {
		device := tc.pools[poolID{driverName: allocatedDevice.Driver, poolName: allocatedDevice.Pool}].getDevice(allocatedDevice.Device)
		if device == nil {
			// Unknown device? Can't be tainted...
			continue
		}

	nextTaint:
		for _, taint := range device.Taints {
			if taint.Effect != resourceapi.DeviceTaintEffectNoExecute {
				continue
			}

			newEvictionTime := taint.TimeAdded
			haveToleration := false
			tolerationSeconds := int64(math.MaxInt64)
			for _, toleration := range allocatedDevice.Tolerations {
				if toleration.Effect == resourceapi.DeviceTaintEffectNoExecute &&
					resourceclaim.ToleratesTaint(toleration, taint) {
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

			if evictionTime == nil {
				evictionTime = newEvictionTime
				continue
			}
			if newEvictionTime != nil && newEvictionTime.Before(evictionTime) {
				evictionTime = newEvictionTime
			}
		}
	}

	return evictionTime
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
		tc.eventLogger.Info("ResourceSlice changed", "pool", poolID, "oldSlice", klog.Format(oldSlice), "newSlice", klog.Format(newSlice), "diff", cmp.Diff(oldSlice, newSlice))
	}

	// Determine old and new device taints. Only devices
	// where something changes trigger additional checks for claims
	// using them.
	//
	// The pre-allocated slices are small enough to be allocated on
	// the stack (https://stackoverflow.com/a/69187698/222305).
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
		newEvictionTime := tc.evictionTime(claim.ResourceClaim.Status.Allocation)
		if newEvictionTime.Equal(claim.evictionTime) {
			// No change.
			continue
		}
		claim.evictionTime = newEvictionTime
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
		tc.eventLogger.Info("Pod changed", "pod", klog.KObj(pod), "oldPod", klog.Format(oldPod), "newPod", klog.Format(newPod), "diff", cmp.Diff(oldPod, newPod))
	}
	if newPod == nil {
		// Nothing left to do for it. No need to emit an event here, it's gone.
		tc.cancelEvict(newObject(oldPod))
		return
	}

	// Pods get updated quite frequently. There's no need
	// to check them again unless something changed regarding
	// their claims.
	//
	// In particular this prevents adding the pod again
	// directly after the eviction condition got added
	// to it.
	if oldPod != nil &&
		apiequality.Semantic.DeepEqual(oldPod.Status.ResourceClaimStatuses, newPod.Status.ResourceClaimStatuses) {
		return
	}

	tc.handlePod(newPod)
}

func (tc *Controller) handlePods(claim *resourceapi.ResourceClaim) {
	for _, consumer := range claim.Status.ReservedFor {
		if consumer.APIGroup == "" && consumer.Resource == "pods" {
			pod, err := tc.podInformer.Lister().Pods(claim.Namespace).Get(consumer.Name)
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
	// Not scheduled yet? No need to evict.
	if pod.Spec.NodeName == "" {
		return
	}

	// If any claim in use by the pod is tainted such that the taint is not tolerated,
	// the pod needs to be evicted.
	var evictionTime *metav1.Time
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
		if allocatedClaim.evictionTime == nil {
			continue
		}
		if evictionTime == nil || allocatedClaim.evictionTime.Before(evictionTime) {
			evictionTime = allocatedClaim.evictionTime
		}
	}

	podRef := newObject(pod)
	if evictionTime != nil {
		tc.evictPod(podRef, evictionTime.Time)
	} else {
		tc.cancelWorkWithEvent(podRef)
	}
}

func (tc *Controller) cancelWorkWithEvent(podRef tainteviction.NamespacedObject) {
	if tc.cancelEvict(podRef) {
		tc.emitCancelPodDeletionEvent(podRef)
	}
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

func newObject(obj metav1.Object) tainteviction.NamespacedObject {
	return tainteviction.NamespacedObject{
		NamespacedName: newNamespacedName(obj),
		UID:            obj.GetUID(),
	}
}
