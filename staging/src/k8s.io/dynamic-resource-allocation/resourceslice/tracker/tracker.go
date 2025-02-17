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
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/google/go-cmp/cmp" //nolint:depguard

	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
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

type Tracker struct {
	enableAdminControlledAttributes bool

	resourceSlices        cache.SharedIndexInformer
	resourceSlicePatches  cache.SharedIndexInformer
	deviceClasses         cache.SharedIndexInformer
	celCache              *cel.Cache
	patchedResourceSlices cache.Store
	recorder              record.EventRecorder
	handleError           func(context.Context, error, string, ...any)

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

type Options struct {
	// EnableAdminControlledAttributes controls whether device
	// attributes and capacities will be reflected in patched
	// ResourceSlices.
	EnableAdminControlledAttributes bool
	// KubeClient is used to generate Events when CEL expressions in
	// ResourceSlicePatches encounter runtime errors.
	KubeClient kubernetes.Interface
}

func StartTracker(ctx context.Context, informerFactory informers.SharedInformerFactory, opts Options) (*Tracker, error) {
	t := newTracker(ctx, informerFactory, opts)
	err := t.initInformers(ctx)
	if err != nil {
		return nil, err
	}
	return t, nil
}

func newTracker(ctx context.Context, informerFactory informers.SharedInformerFactory, opts Options) *Tracker {
	t := &Tracker{
		enableAdminControlledAttributes: opts.EnableAdminControlledAttributes,
		resourceSlices:                  informerFactory.Resource().V1beta1().ResourceSlices().Informer(),
		resourceSlicePatches:            informerFactory.Resource().V1alpha3().ResourceSlicePatches().Informer(),
		deviceClasses:                   informerFactory.Resource().V1beta1().DeviceClasses().Informer(),
		celCache:                        cel.NewCache(10), // TODO: share cache with scheduler
		patchedResourceSlices:           cache.NewStore(cache.MetaNamespaceKeyFunc),
		handleError:                     utilruntime.HandleErrorWithContext,
	}
	t.handlerRegistration = handlerRegistrationFunc(func() bool {
		return t.resourceSlices.HasSynced() &&
			t.resourceSlicePatches.HasSynced() &&
			t.deviceClasses.HasSynced()
	})
	// KubeClient is not always set in unit tests.
	if opts.KubeClient != nil {
		eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
		eventBroadcaster.StartLogging(klog.Infof)
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: opts.KubeClient.CoreV1().Events("")})
		t.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "resource_slice_tracker"})
	}
	return t
}

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
func (t *Tracker) pushEvent(oldObj, newObj interface{}) {
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
		AddFunc:    t.resourceSlicePatchAdd(ctx),
		UpdateFunc: t.resourceSlicePatchUpdate(ctx),
		DeleteFunc: t.resourceSlicePatchDelete(ctx),
	}
	patchHandlerReg, err := t.resourceSlicePatches.AddEventHandler(patchHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for ResourceSlicePatches: %w", err)
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
		t.syncSlice(ctx, slice.Name)
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
		t.syncSlice(ctx, newSlice.Name)
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
		t.syncSlice(ctx, slice.Name)
	}
}

func (t *Tracker) resourceSlicePatchAdd(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		patch, ok := obj.(*resourcealphaapi.ResourceSlicePatch)
		if !ok {
			return
		}
		logger.V(5).Info("ResourceSlicePatch add", "patch", klog.KObj(patch))
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName)
		}
	}
}

func (t *Tracker) resourceSlicePatchUpdate(ctx context.Context) func(oldObj, newObj any) {
	logger := klog.FromContext(ctx)
	return func(oldObj, newObj any) {
		oldPatch, ok := oldObj.(*resourcealphaapi.ResourceSlicePatch)
		if !ok {
			return
		}
		newPatch, ok := newObj.(*resourcealphaapi.ResourceSlicePatch)
		if !ok {
			return
		}
		if loggerV := logger.V(6); loggerV.Enabled() {
			loggerV.Info("ResourceSlicePatch update", "patch", klog.KObj(newPatch), "diff", cmp.Diff(oldPatch, newPatch))
		} else {
			logger.V(5).Info("ResourceSlicePatch update", "patch", klog.KObj(newPatch))
		}
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName)
		}
	}
}

func (t *Tracker) resourceSlicePatchDelete(ctx context.Context) func(obj any) {
	logger := klog.FromContext(ctx)
	return func(obj any) {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		patch, ok := obj.(*resourcealphaapi.ResourceSlicePatch)
		if !ok {
			return
		}
		logger.V(5).Info("ResourceSlicePatch delete", "patch", klog.KObj(patch))
		for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
			t.syncSlice(ctx, sliceName)
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
			t.syncSlice(ctx, sliceName)
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
			t.syncSlice(ctx, sliceName)
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
			t.syncSlice(ctx, sliceName)
		}
	}
}

func (t *Tracker) syncSlice(ctx context.Context, name string) {
	defer t.emitEvents()

	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "sliceName", name)
	logger.V(5).Info("syncing slice")

	obj, sliceExists, err := t.resourceSlices.GetIndexer().GetByKey(name)
	if err != nil {
		t.handleError(ctx, err, "failed to lookup existing resource slice")
		return
	}
	oldPatchedObj, _, err := t.patchedResourceSlices.GetByKey(name)
	if err != nil {
		t.handleError(ctx, err, "failed to lookup cached patched resource slice")
		return
	}
	if !sliceExists {
		err := t.patchedResourceSlices.Delete(oldPatchedObj)
		if err != nil {
			t.handleError(ctx, err, "failed to delete cached patched resource slice")
			return
		}
		t.pushEvent(oldPatchedObj, nil)
		return
	}
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		t.handleError(ctx, errors.New("invalid type in resource slice cache"), fmt.Sprintf("expected type to be %T, got %T", (*resourceapi.ResourceSlice)(nil), obj))
		return
	}
	patchedSlice := slice.DeepCopy()

	patches := typedSlice[*resourcealphaapi.ResourceSlicePatch](t.resourceSlicePatches.GetIndexer().List())
	slices.SortFunc(patches, func(p1, p2 *resourcealphaapi.ResourceSlicePatch) int {
		priority1 := ptr.Deref(p1.Spec.Devices.Priority, 0)
		priority2 := ptr.Deref(p2.Spec.Devices.Priority, 0)
		if priority1 != priority2 {
			return int(priority1 - priority2)
		}
		return -p1.CreationTimestamp.Compare(p2.CreationTimestamp.Time)
	})

	for _, patch := range patches {
		filter := patch.Spec.Devices.Filter
		var filterDeviceClassExprs []cel.CompilationResult
		var filterSelectorExprs []cel.CompilationResult
		var filterDevice *string
		if filter != nil {
			if filter.Driver != nil && *filter.Driver != patchedSlice.Spec.Driver {
				continue
			}
			if filter.Pool != nil && *filter.Pool != patchedSlice.Spec.Pool.Name {
				continue
			}
			filterDevice = filter.Device
			if filter.DeviceClassName != nil {
				classObj, exists, err := t.deviceClasses.GetIndexer().GetByKey(*filter.DeviceClassName)
				if err != nil {
					t.handleError(ctx, err, "failed to get device class", "deviceClassName", *filter.DeviceClassName)
					return
				}
				if !exists {
					continue
				}
				class := classObj.(*resourceapi.DeviceClass)
				for _, selector := range class.Spec.Selectors {
					if selector.CEL == nil {
						continue
					}
					expr := t.celCache.GetOrCompile(selector.CEL.Expression)
					filterDeviceClassExprs = append(filterDeviceClassExprs, expr)
				}
			}
			for _, selector := range filter.Selectors {
				if selector.CEL == nil {
					continue
				}
				expr := t.celCache.GetOrCompile(selector.CEL.Expression)
				filterSelectorExprs = append(filterSelectorExprs, expr)
			}
		}
	devices:
		for dIndex, device := range patchedSlice.Spec.Devices {
			if filterDevice != nil && *filterDevice != device.Name {
				continue
			}

			var deviceAttributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
			var deviceCapacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity
			if device.Basic != nil {
				deviceAttributes = device.Basic.Attributes
				deviceCapacity = device.Basic.Capacity
			}

			for i, expr := range filterDeviceClassExprs {
				if expr.Error != nil {
					// Could happen if some future apiserver accepted some
					// future expression and then got downgraded. Normally
					// the "stored expression" mechanism prevents that, but
					// this code here might be more than one release older
					// than the cluster it runs in.
					err := fmt.Errorf("class %s: selector #%d: CEL compile error: %w", *filter.DeviceClassName, i, expr.Error)
					t.handleError(ctx, err, "failed to compile CEL")
					return
				}
				match, _, err := expr.DeviceMatches(ctx, cel.Device{Driver: patchedSlice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
				// TODO: scheduler logs a lot more info about CEL expression results
				if err != nil {
					continue devices
				}
				if !match {
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
					err := fmt.Errorf("patch %s: selector #%d: CEL compile error: %w", patch.Name, i, expr.Error)
					t.handleError(ctx, err, "failed to compile CEL")
					return
				}
				match, _, err := expr.DeviceMatches(ctx, cel.Device{Driver: patchedSlice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
				if err != nil {
					if t.recorder != nil {
						t.recorder.Eventf(patch, v1.EventTypeWarning, "CELRuntimeError", "selector #%d: runtime error: %v", i, err)
					}
					continue devices
				}
				if !match {
					continue devices
				}
			}

			if t.enableAdminControlledAttributes {
				newAttrs := maps.Clone(deviceAttributes)
				if newAttrs == nil {
					newAttrs = make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
				}
				for key, val := range patch.Spec.Devices.Attributes {
					keyWithoutDomain := strings.TrimPrefix(string(key), patchedSlice.Spec.Driver+"/")
					delete(newAttrs, resourceapi.QualifiedName(keyWithoutDomain))
					if val.NullValue != nil {
						delete(newAttrs, resourceapi.QualifiedName(key))
					} else {
						newKey := resourceapi.QualifiedName(key)
						newVal := resourceapi.DeviceAttribute(val.DeviceAttribute)
						newAttrs[newKey] = newVal
					}
				}
				if len(newAttrs) == 0 {
					newAttrs = nil
				}

				newCaps := maps.Clone(deviceCapacity)
				if newCaps == nil {
					newCaps = make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
				}
				for key, val := range patch.Spec.Devices.Capacity {
					newKey := resourceapi.QualifiedName(key)
					newVal := resourceapi.DeviceCapacity(val)
					newCaps[newKey] = newVal
				}
				if len(newCaps) == 0 {
					newCaps = nil
				}

				if device.Basic != nil {
					patchedSlice.Spec.Devices[dIndex].Basic.Attributes = newAttrs
					patchedSlice.Spec.Devices[dIndex].Basic.Capacity = newCaps
				}
			}
		}
	}

	err = t.patchedResourceSlices.Add(patchedSlice)
	if err != nil {
		t.handleError(ctx, err, "failed to add patched resource slice to cache")
		return
	}
	if !apiequality.Semantic.DeepEqual(oldPatchedObj, patchedSlice) {
		t.pushEvent(oldPatchedObj, patchedSlice)
	}
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
