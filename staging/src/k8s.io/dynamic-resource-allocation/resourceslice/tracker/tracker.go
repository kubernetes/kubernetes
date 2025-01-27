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

package tracker

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/google/go-cmp/cmp" //nolint:depguard

	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	resourcealphainformers "k8s.io/client-go/informers/resource/v1alpha3"
	resourceinformers "k8s.io/client-go/informers/resource/v1beta1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/internal/queue"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

const (
	driverPoolDeviceIndexName = "driverPoolDevice"

	anyDriver = "*"
	anyPool   = "*"
	anyDevice = "*"
)

// Tracker maintains a view of ResourceSlice objects with matching
// DeviceTaints applied. It is backed by informers to process
// potential changes to resolved ResourceSlices asynchronously.
type Tracker struct {
	enableDeviceTaints bool

	resourceSlices        cache.SharedIndexInformer
	deviceTaints          cache.SharedIndexInformer
	deviceClasses         cache.SharedIndexInformer
	celCache              *cel.Cache
	patchedResourceSlices cache.Store
	recorder              record.EventRecorder
	// handleError usually refers to [utilruntime.HandleErrorWithContext] but
	// may be overridden in tests.
	handleError func(context.Context, error, string, ...any)

	// Synchronizes updates to these fields related to event handlers.
	rwMutex sync.RWMutex
	// All registered event handlers.
	eventHandlers       []cache.ResourceEventHandler
	handlerRegistration cache.ResourceEventHandlerRegistration
	// The eventQueue contains functions which deliver an event to one
	// event handler.
	//
	// These functions must be invoked while *not locking* rwMutex because
	// the event handlers are allowed to access the cache. Holding rwMutex
	// then would cause a deadlock.
	//
	// New functions get added as part of processing a cache update while
	// the rwMutex is locked. Each function which adds something to the queue
	// also drains the queue before returning, therefore it is guaranteed
	// that all event handlers get notified immediately (useful for unit
	// testing).
	//
	// A channel cannot be used here because it cannot have an unbounded
	// capacity. This could lead to a deadlock (writer holds rwMutex,
	// gets blocked because capacity is exhausted, reader is in a handler
	// which tries to lock the rwMutex). Writing into such a channel
	// while not holding the rwMutex doesn't work because in-order delivery
	// of events would no longer be guaranteed.
	eventQueue queue.FIFO[func()]
}

// Options configure a [Tracker].
type Options struct {
	// EnableDeviceTaints controls whether device taints
	// will be reflected in patched ResourceSlices.
	EnableDeviceTaints bool
	// KubeClient is used to generate Events when CEL expressions
	// encounter runtime errors.
	KubeClient kubernetes.Interface
}

// StartTracker creates and initializes informers for a new [Tracker].
//
// TODO: make taintInformer optional and pass through to the sliceInformer in that case (feature disabled).
func StartTracker(ctx context.Context, sliceInformer resourceinformers.ResourceSliceInformer, taintInformer resourcealphainformers.DeviceTaintInformer, classInformer resourceinformers.DeviceClassInformer, opts Options) (*Tracker, error) {
	t, err := newTracker(ctx, sliceInformer, taintInformer, classInformer, opts)
	if err != nil {
		return nil, err
	}
	err = t.initInformers(ctx)
	if err != nil {
		return nil, err
	}
	return t, nil
}

func newTracker(ctx context.Context, sliceInformer resourceinformers.ResourceSliceInformer, taintInformer resourcealphainformers.DeviceTaintInformer, classInformer resourceinformers.DeviceClassInformer, opts Options) (*Tracker, error) {
	t := &Tracker{
		enableDeviceTaints:    opts.EnableDeviceTaints,
		resourceSlices:        sliceInformer.Informer(),
		deviceTaints:          taintInformer.Informer(),
		deviceClasses:         classInformer.Informer(),
		celCache:              cel.NewCache(10),
		patchedResourceSlices: cache.NewStore(cache.MetaNamespaceKeyFunc),
		handleError:           utilruntime.HandleErrorWithContext,
	}
	err := t.resourceSlices.AddIndexers(cache.Indexers{driverPoolDeviceIndexName: sliceDriverPoolDeviceIndexFunc})
	if err != nil {
		return nil, fmt.Errorf("failed to add %s index to ResourceSlice informer: %w", driverPoolDeviceIndexName, err)
	}
	t.handlerRegistration = handlerRegistrationFunc(func() bool {
		return t.resourceSlices.HasSynced() &&
			t.deviceTaints.HasSynced() &&
			t.deviceClasses.HasSynced()
	})
	// KubeClient is not always set in unit tests.
	if opts.KubeClient != nil {
		eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
		eventBroadcaster.StartLogging(klog.Infof)
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: opts.KubeClient.CoreV1().Events("")})
		t.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "resource_slice_tracker"})
	}
	return t, nil
}

// ListPatchedResourceSlices returns all ResourceSlices in the cluster with
// modifications from DeviceTaints applied.
func (t *Tracker) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	return typedSlice[*resourceapi.ResourceSlice](t.patchedResourceSlices.List()), nil
}

// AddEventHandler adds an event handler to the tracker. Events to a
// single handler are delivered sequentially, but there is no
// coordination between different handlers. A handler may use the
// tracker.
//
// The return value can be used to wait for cache synchronization.
func (t *Tracker) AddEventHandler(handler cache.ResourceEventHandler) cache.ResourceEventHandlerRegistration {
	defer t.emitEvents()
	t.rwMutex.Lock()
	defer t.rwMutex.Unlock()

	t.eventHandlers = append(t.eventHandlers, handler)
	allObjs, _ := t.ListPatchedResourceSlices()
	for _, obj := range allObjs {
		t.eventQueue.Push(func() {
			handler.OnAdd(obj, true)
		})
	}

	return t.handlerRegistration
}

// emitEvents delivers all pending events that are in the queue, in the order
// in which they were stored there (FIFO).
func (t *Tracker) emitEvents() {
	for {
		t.rwMutex.Lock()
		deliver, ok := t.eventQueue.Pop()
		t.rwMutex.Unlock()

		if !ok {
			return
		}
		func() {
			defer utilruntime.HandleCrash()
			deliver()
		}()
	}
}

// pushEvent ensures that all currently registered event handlers get
// notified about a change when the caller starts delivering
// those with emitEvents.
//
// For a delete event, newObj is nil. For an add, oldObj is nil.
// An update has both as non-nil.
func (t *Tracker) pushEvent(oldObj, newObj any) {
	t.rwMutex.Lock()
	defer t.rwMutex.Unlock()
	for _, handler := range t.eventHandlers {
		handler := handler
		if oldObj == nil {
			t.eventQueue.Push(func() {
				handler.OnAdd(newObj, false)
			})
		} else if newObj == nil {
			t.eventQueue.Push(func() {
				handler.OnDelete(oldObj)
			})
		} else {
			t.eventQueue.Push(func() {
				handler.OnUpdate(oldObj, newObj)
			})
		}
	}
}

func sliceDriverPoolDeviceIndexFunc(obj any) ([]string, error) {
	slice := obj.(*resourceapi.ResourceSlice)
	drivers := []string{
		anyDriver,
		slice.Spec.Driver,
	}
	pools := []string{
		anyPool,
		slice.Spec.Pool.Name,
	}
	indexValues := make([]string, 0, len(drivers)*len(pools)*(1+len(slice.Spec.Devices)))
	for _, driver := range drivers {
		for _, pool := range pools {
			indexValues = append(indexValues, deviceID(driver, pool, anyDevice))
			for _, device := range slice.Spec.Devices {
				indexValues = append(indexValues, deviceID(driver, pool, device.Name))
			}
		}
	}
	return indexValues, nil
}

func driverPoolDeviceIndexPatchKey(patch *resourcealphaapi.DeviceTaint) string {
	filter := ptr.Deref(patch.Spec.Filter, resourcealphaapi.DeviceTaintFilter{})
	driverKey := ptr.Deref(filter.Driver, anyDriver)
	poolKey := ptr.Deref(filter.Pool, anyPool)
	deviceKey := ptr.Deref(filter.Device, anyDevice)
	return deviceID(driverKey, poolKey, deviceKey)
}

func (t *Tracker) sliceNamesForPatch(ctx context.Context, patch *resourcealphaapi.DeviceTaint) []string {
	patchKey := driverPoolDeviceIndexPatchKey(patch)
	sliceNames, err := t.resourceSlices.GetIndexer().IndexKeys(driverPoolDeviceIndexName, patchKey)
	if err != nil {
		t.handleError(ctx, err, "failed listing ResourceSlices for driver/pool/device key", "key", patchKey)
		return nil
	}
	return sliceNames
}

func (t *Tracker) initInformers(ctx context.Context) error {
	sliceHandler := cache.ResourceEventHandlerFuncs{
		AddFunc:    t.resourceSliceAdd(ctx),
		UpdateFunc: t.resourceSliceUpdate(ctx),
		DeleteFunc: t.resourceSliceDelete(ctx),
	}
	sliceHandlerReg, err := t.resourceSlices.AddEventHandler(sliceHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for ResourceSlices: %w", err)
	}

	patchHandler := cache.ResourceEventHandlerFuncs{
		AddFunc:    t.deviceTaintAdd(ctx),
		UpdateFunc: t.deviceTaintUpdate(ctx),
		DeleteFunc: t.deviceTaintDelete(ctx),
	}
	patchHandlerReg, err := t.deviceTaints.AddEventHandler(patchHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for DeviceTaints: %w", err)
	}

	classHandler := cache.ResourceEventHandlerFuncs{
		AddFunc:    t.deviceClassAdd(ctx),
		UpdateFunc: t.deviceClassUpdate(ctx),
		DeleteFunc: t.deviceClassDelete(ctx),
	}
	classHandlerReg, err := t.deviceClasses.AddEventHandler(classHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for DeviceClasses: %w", err)
	}

	t.handlerRegistration = handlerRegistrationFunc(func() bool {
		return sliceHandlerReg.HasSynced() &&
			patchHandlerReg.HasSynced() &&
			classHandlerReg.HasSynced()
	})

	return nil
}

func (t *Tracker) resourceSliceAdd(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		slice, ok := obj.(*resourceapi.ResourceSlice)
		if !ok {
			return
		}
		logger.V(5).Info("ResourceSlice add", "slice", klog.KObj(slice))
		t.syncSlice(ctx, slice.Name, true)
	}
}

func (t *Tracker) resourceSliceUpdate(ctx context.Context) func(oldObj, newObj any) {
	logger := klog.FromContext(ctx)
	return func(oldObj, newObj any) {
		oldSlice, ok := oldObj.(*resourceapi.ResourceSlice)
		if !ok {
			return
		}
		newSlice, ok := newObj.(*resourceapi.ResourceSlice)
		if !ok {
			return
		}
		if loggerV := logger.V(6); loggerV.Enabled() {
			loggerV.Info("ResourceSlice update", "slice", klog.KObj(newSlice), "diff", cmp.Diff(oldSlice, newSlice))
		} else {
			logger.V(5).Info("ResourceSlice update", "slice", klog.KObj(newSlice))
		}
		t.syncSlice(ctx, newSlice.Name, true)
	}
}

func (t *Tracker) resourceSliceDelete(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		slice, ok := obj.(*resourceapi.ResourceSlice)
		if !ok {
			return
		}
		logger.V(5).Info("ResourceSlice delete", "slice", klog.KObj(slice))
		t.syncSlice(ctx, slice.Name, true)
	}
}

func (t *Tracker) deviceTaintAdd(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		patch, ok := obj.(*resourcealphaapi.DeviceTaint)
		if !ok {
			return
		}
		logger.V(5).Info("DeviceTaint add", "patch", klog.KObj(patch))
		for _, sliceName := range t.sliceNamesForPatch(ctx, patch) {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

func (t *Tracker) deviceTaintUpdate(ctx context.Context) func(oldObj, newObj any) {
	logger := klog.FromContext(ctx)
	return func(oldObj, newObj any) {
		oldPatch, ok := oldObj.(*resourcealphaapi.DeviceTaint)
		if !ok {
			return
		}
		newPatch, ok := newObj.(*resourcealphaapi.DeviceTaint)
		if !ok {
			return
		}
		if loggerV := logger.V(6); loggerV.Enabled() {
			loggerV.Info("DeviceTaint update", "patch", klog.KObj(newPatch), "diff", cmp.Diff(oldPatch, newPatch))
		} else {
			logger.V(5).Info("DeviceTaint update", "patch", klog.KObj(newPatch))
		}

		// Slices that matched the old patch may need to be updated, in
		// case they no longer match the new patch and need to have the
		// patch's changes reverted.
		slicesToSync := sets.New[string]()
		slicesToSync.Insert(t.sliceNamesForPatch(ctx, oldPatch)...)
		slicesToSync.Insert(t.sliceNamesForPatch(ctx, newPatch)...)
		for _, sliceName := range slicesToSync.UnsortedList() {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

func (t *Tracker) deviceTaintDelete(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		patch, ok := obj.(*resourcealphaapi.DeviceTaint)
		if !ok {
			return
		}
		logger.V(5).Info("DeviceTaint delete", "patch", klog.KObj(patch))
		for _, sliceName := range t.sliceNamesForPatch(ctx, patch) {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

func (t *Tracker) deviceClassAdd(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		class, ok := obj.(*resourceapi.DeviceClass)
		if !ok {
			return
		}
		logger.V(5).Info("DeviceClass add", "class", klog.KObj(class))
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

func (t *Tracker) deviceClassUpdate(ctx context.Context) func(oldObj, newObj any) {
	logger := klog.FromContext(ctx)
	return func(oldObj, newObj any) {
		oldClass, ok := oldObj.(*resourceapi.DeviceClass)
		if !ok {
			return
		}
		newClass, ok := newObj.(*resourceapi.DeviceClass)
		if !ok {
			return
		}
		if loggerV := logger.V(6); loggerV.Enabled() {
			loggerV.Info("DeviceClass update", "class", klog.KObj(newClass), "diff", cmp.Diff(oldClass, newClass))
		} else {
			logger.V(5).Info("DeviceClass update", "class", klog.KObj(newClass))
		}
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

func (t *Tracker) deviceClassDelete(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		class, ok := obj.(*resourceapi.ResourceSlice)
		if !ok {
			return
		}
		logger.V(5).Info("DeviceClass delete", "class", klog.KObj(class))
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName, false)
		}
	}
}

// syncSlice updates the slice with the given name, applying
// DeviceTaints that match. sendEvent is used to force the Tracker
// to publish an event for listeners added by [Tracker.AddEventHandler]. It
// is set when syncSlice is triggered by a ResourceSlice event to avoid
// doing costly DeepEqual comparisons where possible.
func (t *Tracker) syncSlice(ctx context.Context, name string, sendEvent bool) {
	defer t.emitEvents()

	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "resourceslice", name)
	ctx = klog.NewContext(ctx, logger)
	logger.V(5).Info("syncing ResourceSlice")

	obj, sliceExists, err := t.resourceSlices.GetIndexer().GetByKey(name)
	if err != nil {
		t.handleError(ctx, err, "failed to lookup existing resource slice", "resourceslice", name)
		return
	}
	oldPatchedObj, oldSliceExists, err := t.patchedResourceSlices.GetByKey(name)
	if err != nil {
		t.handleError(ctx, err, "failed to lookup cached patched resource slice", "resourceslice", name)
		return
	}
	if !sliceExists {
		err := t.patchedResourceSlices.Delete(oldPatchedObj)
		if err != nil {
			t.handleError(ctx, err, "failed to delete cached patched resource slice", "resourceslice", name)
			return
		}
		t.pushEvent(oldPatchedObj, nil)
		logger.V(5).Info("patched ResourceSlice deleted")
		return
	}
	var oldPatchedSlice *resourceapi.ResourceSlice
	if oldSliceExists {
		var ok bool
		oldPatchedSlice, ok = oldPatchedObj.(*resourceapi.ResourceSlice)
		if !ok {
			t.handleError(ctx, errors.New("invalid type in resource slice cache"), "expectedType", fmt.Sprintf("%T", (*resourceapi.ResourceSlice)(nil)), "gotType", fmt.Sprintf("%T", oldPatchedObj))
			return
		}
	}
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		t.handleError(ctx, errors.New("invalid type in resource slice cache"), fmt.Sprintf("expected type to be %T, got %T", (*resourceapi.ResourceSlice)(nil), obj))
		return
	}

	patches := typedSlice[*resourcealphaapi.DeviceTaint](t.deviceTaints.GetIndexer().List())
	patchedSlice, err := t.applyPatches(ctx, slice, patches)
	if err != nil {
		t.handleError(ctx, err, "failed to apply patches to ResourceSlice", "resourceslice", klog.KObj(slice))
		return
	}

	// When syncSlice is triggered by something other than a ResourceSlice
	// event, only the device attributes and capacity might change. We
	// deliberately avoid any costly DeepEqual-style comparisons here.
	if !sendEvent && oldPatchedSlice != nil {
		for i := range patchedSlice.Spec.Devices {
			oldDevice := oldPatchedSlice.Spec.Devices[i]
			newDevice := patchedSlice.Spec.Devices[i]
			sendEvent = sendEvent ||
				!slices.EqualFunc(getTaints(oldDevice), getTaints(newDevice), taintsEqual)
		}
	}

	err = t.patchedResourceSlices.Add(patchedSlice)
	if err != nil {
		t.handleError(ctx, err, "failed to add patched resource slice to cache", "resourceslice", klog.KObj(patchedSlice))
		return
	}
	if sendEvent {
		t.pushEvent(oldPatchedObj, patchedSlice)
	}

	if loggerV := logger.V(6); loggerV.Enabled() {
		loggerV.Info("ResourceSlice synced", "diff", cmp.Diff(oldPatchedObj, patchedSlice))
	} else {
		logger.V(5).Info("ResourceSlice synced")
	}
}

func (t *Tracker) applyPatches(ctx context.Context, slice *resourceapi.ResourceSlice, taints []*resourcealphaapi.DeviceTaint) (*resourceapi.ResourceSlice, error) {
	logger := klog.FromContext(ctx)

	// slice will be DeepCopied just-in-time, only when necessary.
	patchedSlice := slice

	for _, taint := range taints {
		logger := klog.LoggerWithValues(logger, "devicetaint", klog.KObj(taint))
		logger.V(6).Info("processing DeviceTaint")

		filter := taint.Spec.Filter
		var filterDeviceClassExprs []cel.CompilationResult
		var filterSelectorExprs []cel.CompilationResult
		var filterDevice *string
		if filter != nil {
			if filter.Driver != nil && *filter.Driver != slice.Spec.Driver {
				logger.V(7).Info("DeviceTaint does not apply, mismatched driver", "sliceDriver", slice.Spec.Driver, "taintDriver", *filter.Driver)
				continue
			}
			if filter.Pool != nil && *filter.Pool != slice.Spec.Pool.Name {
				logger.V(7).Info("DeviceTaint does not apply, mismatched pool", "slicePool", slice.Spec.Pool.Name, "taintPool", *filter.Pool)
				continue
			}
			filterDevice = filter.Device
			if filter.DeviceClassName != nil {
				logger := logger.WithValues("deviceClassName", *filter.DeviceClassName)
				classObj, exists, err := t.deviceClasses.GetIndexer().GetByKey(*filter.DeviceClassName)
				if err != nil {
					return nil, fmt.Errorf("failed to get device class %s for DeviceTaint %s", *filter.DeviceClassName, taint.Name)
				}
				if !exists {
					logger.V(7).Info("DeviceTaint does not apply, DeviceClass does not exist")
					continue
				}
				class := classObj.(*resourceapi.DeviceClass)
				for _, selector := range class.Spec.Selectors {
					if selector.CEL != nil {
						expr := t.celCache.GetOrCompile(selector.CEL.Expression)
						filterDeviceClassExprs = append(filterDeviceClassExprs, expr)
					}
				}
			}
			for _, selector := range filter.Selectors {
				if selector.CEL != nil {
					expr := t.celCache.GetOrCompile(selector.CEL.Expression)
					filterSelectorExprs = append(filterSelectorExprs, expr)
				}
			}
		}
	devices:
		for dIndex, device := range slice.Spec.Devices {
			deviceID := deviceID(slice.Spec.Driver, slice.Spec.Pool.Name, device.Name)
			logger := logger.WithValues("device", deviceID)

			if filterDevice != nil && *filterDevice != device.Name {
				logger.V(7).Info("DeviceTaint does not apply, mismatched device", "sliceDevice", device.Name, "taintDevice", *filter.Device)
				continue
			}

			deviceAttributes := getAttributes(device)
			deviceCapacity := getCapacity(device)

			for i, expr := range filterDeviceClassExprs {
				if expr.Error != nil {
					// Could happen if some future apiserver accepted some
					// future expression and then got downgraded. Normally
					// the "stored expression" mechanism prevents that, but
					// this code here might be more than one release older
					// than the cluster it runs in.
					return nil, fmt.Errorf("DeviceTaint %s: class %s: selector #%d: CEL compile error: %w", taint.Name, *filter.DeviceClassName, i, expr.Error)
				}
				matches, details, err := expr.DeviceMatches(ctx, cel.Device{Driver: slice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
				logger.V(7).Info("CEL result", "class", *filter.DeviceClassName, "selector", i, "expression", expr.Expression, "matches", matches, "actualCost", ptr.Deref(details.ActualCost(), 0), "err", err)
				if err != nil {
					continue devices
				}
				if !matches {
					continue devices
				}
			}

			for i, expr := range filterSelectorExprs {
				if expr.Error != nil {
					// Could happen if some future apiserver accepted some
					// future expression and then got downgraded. Normally
					// the "stored expression" mechanism prevents that, but
					// this code here might be more than one release older
					// than the cluster it runs in.
					return nil, fmt.Errorf("DeviceTaint %s: selector #%d: CEL compile error: %w", taint.Name, i, expr.Error)
				}
				matches, details, err := expr.DeviceMatches(ctx, cel.Device{Driver: slice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
				logger.V(7).Info("CEL result", "selector", i, "expression", expr.Expression, "matches", matches, "actualCost", ptr.Deref(details.ActualCost(), 0), "err", err)
				if err != nil {
					if t.recorder != nil {
						t.recorder.Eventf(taint, v1.EventTypeWarning, "CELRuntimeError", "selector #%d: runtime error: %v", i, err)
					}
					continue devices
				}
				if !matches {
					continue devices
				}
			}

			logger.V(6).Info("applying matching DeviceTaint")

			// TODO: remove conversion once taint is already in the right API package.
			ta := resourceapi.DeviceTaintAtom{
				Key:       taint.Spec.Taint.Key,
				Value:     taint.Spec.Taint.Value,
				Effect:    resourceapi.DeviceTaintEffect(taint.Spec.Taint.Effect),
				TimeAdded: taint.Spec.Taint.TimeAdded,
			}

			if patchedSlice == slice {
				patchedSlice = slice.DeepCopy()
			}

			appendTaint(&patchedSlice.Spec.Devices[dIndex], ta)
		}
	}

	return patchedSlice, nil
}

func getAttributes(device resourceapi.Device) map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	if device.Basic != nil {
		return device.Basic.Attributes
	}
	return nil
}

func getCapacity(device resourceapi.Device) map[resourceapi.QualifiedName]resourceapi.DeviceCapacity {
	if device.Basic != nil {
		return device.Basic.Capacity
	}
	return nil
}

func getTaints(device resourceapi.Device) []resourceapi.DeviceTaintAtom {
	if device.Basic != nil {
		return device.Basic.Taints
	}
	return nil
}

func appendTaint(device *resourceapi.Device, taint resourceapi.DeviceTaintAtom) {
	if device.Basic != nil {
		device.Basic.Taints = append(device.Basic.Taints, taint)
		return
	}
}

func taintsEqual(a, b resourceapi.DeviceTaintAtom) bool {
	return a.Key == b.Key &&
		a.Effect == b.Effect &&
		a.Value == b.Value &&
		a.TimeAdded.Equal(b.TimeAdded) // Equal deals with nil.
}

func deviceID(driver, pool, device string) string {
	return driver + "/" + pool + "/" + device
}

func typedSlice[T any](objs []any) []T {
	if objs == nil {
		return nil
	}
	typed := make([]T, 0, len(objs))
	for _, obj := range objs {
		typed = append(typed, obj.(T))
	}
	return typed
}

type handlerRegistrationFunc func() bool

func (f handlerRegistrationFunc) HasSynced() bool {
	return f()
}
