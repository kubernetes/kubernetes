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
	"sort"
	"sync"
	"time"

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
	"k8s.io/kubernetes/pkg/controller/tainteviction"
	"k8s.io/kubernetes/pkg/controller/tainteviction/metrics"
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
type Controller struct {
	name   string
	logger klog.Logger

	client             clientset.Interface
	broadcaster        record.EventBroadcaster
	recorder           record.EventRecorder
	podInformer        coreinformers.PodInformer
	podLister          corelisters.PodLister
	claimInformer      resourceinformers.ResourceClaimInformer
	sliceInformer      resourceinformers.ResourceSliceInformer
	slicePatchInformer resourcealphainformers.ResourceSlicePatchInformer
	classInformer      resourceinformers.DeviceClassInformer
	haveSynced         []cache.InformerSynced

	// evictPod ensures that the pod gets evicted at the specified time.
	// It doesn't block.
	evictPod func(pod object, fireAt time.Time)

	// cancelEvict cancels eviction set up with evictPod earlier.
	// Idempotent, returns false if there was nothing to cancel.
	cancelEvict func(pod object) bool

	// allocatedClaims holds all currently known allocated claims.
	allocatedClaims map[types.NamespacedName]allocatedClaim // TODO: is value or pointer more efficient?

	// pools indexes all slices by driver and pool name.
	pools map[poolID]pool
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
//
// It takes a buffer from the caller to allow avoiding memory allocations in
// the common case of only a few taints. A flat list is used for the same
// reason. The caller should pre-allocate the buffer with a small, fixed
// capacity.
func (p pool) getTaintedDevices(buffer []taintedDevice) []taintedDevice {
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

	// TODO: is slices.SortFunc faster?
	sort.Slice(buffer, func(i, j int) bool {
		return buffer[i].deviceName < buffer[j].deviceName
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

// object combines the namespace+name of an object with its UID.
type object struct {
	types.NamespacedName
	UID types.UID
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

// deviceID references one particular device by its driver/pool/device tuple.
type deviceID struct {
	driverName, poolName, deviceName string
}

// deviceGeneration tracks the latest device that is referenced by a deviceID.
type deviceGeneration struct {
	generation int64
	*resourceapi.BasicDevice
}

// TODO: this should consider the pod UID. Otherwise the decision to evict pod "foo" might have been
// for the pod with UID 1234 but the eviction then hits the pod with UID 5678 if it happens to have
// replaced the old pod under the same name.
func deletePodHandler(c clientset.Interface, emitEventFunc func(types.NamespacedName), controllerName string) func(ctx context.Context, fireAt time.Time, args *tainteviction.WorkArgs) error {
	return func(ctx context.Context, fireAt time.Time, args *tainteviction.WorkArgs) error {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		klog.FromContext(ctx).Info("Deleting pod", "pod", args.NamespacedName)
		if emitEventFunc != nil {
			emitEventFunc(args.NamespacedName)
		}
		var err error
		for i := 0; i < retries; i++ {
			err = addConditionAndDeletePod(ctx, c, name, ns)
			if err == nil {
				metrics.PodDeletionsTotal.Inc()
				metrics.PodDeletionsLatency.Observe(float64(time.Since(fireAt) * time.Second))
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		return err
	}
}

func addConditionAndDeletePod(ctx context.Context, c clientset.Interface, name, ns string) (err error) {
	pod, err := c.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
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
	return c.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
}

// New creates a new Controller that will use passed clientset to communicate with the API server.
// Spawns no goroutines. That happens in Run.
func New(c clientset.Interface, podInformer coreinformers.PodInformer, claimInformer resourceinformers.ResourceClaimInformer, sliceInformer resourceinformers.ResourceSliceInformer, slicePatchInformer resourcealphainformers.ResourceSlicePatchInformer, classInformer resourceinformers.DeviceClassInformer, controllerName string) *Controller {
	metrics.Register() // It would be nicer to pass the controller name here, but that probably would break generating https://kubernetes.io/docs/reference/instrumentation/metrics.

	tc := &Controller{
		name: controllerName,

		client:             c,
		podInformer:        podInformer,
		podLister:          podInformer.Lister(),
		claimInformer:      claimInformer,
		sliceInformer:      sliceInformer,
		slicePatchInformer: slicePatchInformer,
		classInformer:      classInformer,
		allocatedClaims:    make(map[types.NamespacedName]allocatedClaim),
		pools:              make(map[poolID]pool),
		// Instantiate all informers now to ensure that they get started.
		haveSynced: []cache.InformerSynced{
			podInformer.Informer().HasSynced,
			claimInformer.Informer().HasSynced,
			sliceInformer.Informer().HasSynced,
			slicePatchInformer.Informer().HasSynced,
			classInformer.Informer().HasSynced,
		},
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

	// Delayed construction of broadcaster because it spawns goroutines.
	// tc.recorder.Eventf is a local in-memory operation which never
	// blocks, so it is safe to call from an event handler. The
	// actual API calls then happen in those spawned goroutines.
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	tc.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: tc.name}).WithLogger(logger)
	defer eventBroadcaster.Shutdown()

	taintEvictionQueue := tainteviction.CreateWorkerQueue(deletePodHandler(tc.client, tc.emitPodDeletionEvent, tc.name))
	tc.evictPod = func(pod object, fireAt time.Time) {
		taintEvictionQueue.AddWork(ctx, tainteviction.NewWorkArgs(pod.Name, pod.Namespace), time.Now(), fireAt)
	}
	tc.cancelEvict = func(pod object) bool {
		// TODO: clean up key handling: here we use types.NamespacedName.String, elsewhere WorkArgs.KeyFromWorkArgs.
		// Both happen to be the same, but it's still a bit dirty.
		return taintEvictionQueue.CancelWork(logger, pod.NamespacedName.String())
	}

	// Start events processing pipeline.
	tc.broadcaster.StartStructuredLogging(3)
	if tc.client != nil {
		logger.Info("Sending events to api server")
		tc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: tc.client.CoreV1().Events("")})
	} else {
		logger.Error(nil, "kubeClient is nil", "controller", tc.name)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	defer tc.broadcaster.Shutdown()

	// mutex serializes event processing.
	var mutex sync.Mutex

	claimHandler, err := tc.claimInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
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
	defer tc.claimInformer.Informer().RemoveEventHandler(claimHandler)

	podHandler, err := tc.podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
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
	defer tc.podInformer.Informer().RemoveEventHandler(podHandler)

	// Wait for the caches to be synced.
	if !cache.WaitForNamedCacheSyncWithContext(ctx, tc.haveSynced...) {
		return
	}

	// At this point, all claims and pods are assumed to be unaffected by
	// taints. This will be reconsidered in response to observing tainted
	// devices, which gets kicked of here.
	sliceTracker, err := resourceslicetracker.StartTracker(ctx, tc.client, tc.sliceInformer, tc.slicePatchInformer, tc.classInformer)
	// TODO: defer sliceTracker.Stop()
	if err != nil {
		logger.Info("Failed to initialize ResourceSlice patch tracker, stopping to evict pods", "err", err)
		return
	}

	// TODO: wait for tracker to sync before we react to events. Otherwise
	// some unnecessary work might be done as events get emitted for
	// intermediate state.

	sliceTracker.AddEventHandler(cache.ResourceEventHandlerFuncs{
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

	<-ctx.Done()
}

func (tc *Controller) handleClaimChange(oldClaim, newClaim *resourceapi.ResourceClaim) {
	claim := newClaim
	if claim == nil {
		claim = oldClaim
	}
	name := newNamespacedName(claim)

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
			evictionTime:  tc.evictionTime(claim),
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
			evictionTime:  tc.evictionTime(claim),
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

// evictionTime returns the smallest TimeAdded of any NoExecute taint in any allocated device
// unless that taint is tolerated, nil if none.
func (tc *Controller) evictionTime(claim *resourceapi.ResourceClaim) *metav1.Time {
	var evictionTime *metav1.Time

	if claim.Status.Allocation == nil {
		return nil
	}

	for _, allocatedDevice := range claim.Status.Allocation.Devices.Results {
		device := tc.pools[poolID{driverName: allocatedDevice.Driver, poolName: allocatedDevice.Pool}].getDevice(allocatedDevice.Device)
		if device == nil {
			// Unknown device? Can't be tainted...
			continue
		}

		for _, taint := range device.Taints {
			if taint.Effect != resourceapi.DeviceTaintEffectNoExecute {
				continue
			}

			// TODO: tolerations, including delay.

			if evictionTime == nil {
				evictionTime = taint.TimeAdded
				continue
			}

			if taint.TimeAdded != nil && taint.TimeAdded.Before(evictionTime) {
				evictionTime = taint.TimeAdded
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

	// Determine old and new device taints. Only devices
	// where something changes trigger additional checks for claims
	// using them.
	//
	// The pre-allocated slices are small enough to be allocated on
	// the stack (https://stackoverflow.com/a/69187698/222305).
	p := tc.pools[poolID]
	oldDeviceTaints := make([]taintedDevice, 0, 10*resourceapi.ResourceSliceMaxDevices)
	oldDeviceTaints = p.getTaintedDevices(oldDeviceTaints)
	p.removeSlice(oldSlice)
	p.addSlice(newSlice)
	if len(p.slices) == 0 {
		delete(tc.pools, poolID)
	} else {
		tc.pools[poolID] = p
	}
	newDeviceTaints := make([]taintedDevice, 0, 10*resourceapi.ResourceSliceMaxDevices)
	newDeviceTaints = p.getTaintedDevices(newDeviceTaints)

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
		newEvictionTime := tc.evictionTime(claim.ResourceClaim)
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
	if newPod == nil {
		// Nothing left to do for it.
		tc.cancelEvict(newObject(oldPod))
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
		if allocatedClaim.evictionTime == nil {
			continue
		}
		if evictionTime == nil || allocatedClaim.evictionTime.Before(evictionTime) {
			evictionTime = allocatedClaim.evictionTime
		}
	}

	podRef := newObject(pod)
	tc.cancelWorkWithEvent(podRef)
	if evictionTime != nil {
		tc.evictPod(podRef, evictionTime.Time)
	}
}

func (tc *Controller) cancelWorkWithEvent(pod object) {
	if tc.cancelEvict(pod) {
		tc.emitCancelPodDeletionEvent(pod.NamespacedName)
	}
}

func (tc *Controller) emitPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "DeviceTaintManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (tc *Controller) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "DeviceTaintManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}

// getMinTolerationTime returns minimal toleration time from the given slice, or -1 if it's infinite.
// It returns 0 if there are no tolerations.
func getMinTolerationTime(tolerations []resourceapi.DeviceToleration) time.Duration {
	minTolerationTime := int64(math.MaxInt64)
	if len(tolerations) == 0 {
		return 0
	}

	for i := range tolerations {
		if tolerations[i].TolerationSeconds != nil {
			tolerationSeconds := *(tolerations[i].TolerationSeconds)
			if tolerationSeconds <= 0 {
				return 0
			} else if tolerationSeconds < minTolerationTime {
				minTolerationTime = tolerationSeconds
			}
		}
	}

	if minTolerationTime == int64(math.MaxInt64) {
		return -1
	}
	return time.Duration(minTolerationTime) * time.Second
}

func newNamespacedName(obj metav1.Object) types.NamespacedName {
	return types.NamespacedName{
		Namespace: obj.GetNamespace(),
		Name:      obj.GetName(),
	}
}

func newObject(obj metav1.Object) object {
	return object{
		NamespacedName: newNamespacedName(obj),
		UID:            obj.GetUID(),
	}
}
