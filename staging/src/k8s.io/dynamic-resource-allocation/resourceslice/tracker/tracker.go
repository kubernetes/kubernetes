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
	"time"

	"github.com/google/go-cmp/cmp" //nolint:depguard

	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

const defaultSyncDelay = 30 * time.Second

type Tracker struct {
	resourceSlices        cache.SharedIndexInformer
	resourceSlicePatches  cache.SharedIndexInformer
	deviceClasses         cache.SharedIndexInformer
	celCache              *cel.Cache
	patchedResourceSlices cache.Store
	queue                 workqueue.TypedRateLimitingInterface[string]
	wg                    sync.WaitGroup
	cancel                func(cause error)
	recorder              record.EventRecorder
}

func newTracker(ctx context.Context, kubeClient kubernetes.Interface, informerFactory informers.SharedInformerFactory) (*Tracker, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	t := &Tracker{
		resourceSlices:        informerFactory.Resource().V1beta1().ResourceSlices().Informer(),
		resourceSlicePatches:  informerFactory.Resource().V1alpha3().ResourceSlicePatches().Informer(),
		deviceClasses:         informerFactory.Resource().V1beta1().DeviceClasses().Informer(),
		celCache:              cel.NewCache(10), // TODO: share cache with scheduler
		patchedResourceSlices: cache.NewStore(cache.MetaNamespaceKeyFunc),
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "resource_slice_tracker"},
		),
		cancel: cancel,
	}
	err := t.addeventHandlers(ctx)
	if err != nil {
		return nil, err
	}
	if kubeClient != nil {
		eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
		eventBroadcaster.StartLogging(klog.Infof)
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
		t.recorder = eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "resource_slice_tracker"})
	}
	return t, nil
}

func StartTracker(ctx context.Context, kubeClient kubernetes.Interface, informerFactory informers.SharedInformerFactory) (*Tracker, error) {
	logger := klog.FromContext(ctx)
	t, err := newTracker(ctx, kubeClient, informerFactory)
	if err != nil {
		return nil, fmt.Errorf("create controller: %w", err)
	}

	logger.V(3).Info("Starting")
	t.wg.Add(1)
	go func() {
		defer t.wg.Done()
		defer logger.V(3).Info("Stopping")
		t.run(ctx)
	}()
	return t, nil
}

// Stop cancels all background activity and blocks until the tracker has stopped.
func (t *Tracker) Stop() {
	if t == nil {
		return
	}
	t.cancel(errors.New("ResourceSlice tracker was asked to stop"))
	t.wg.Wait()
}

func (t *Tracker) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	return typedSlice[*resourceapi.ResourceSlice](t.patchedResourceSlices.List()), nil
}

func (t *Tracker) addeventHandlers(ctx context.Context) error {
	logger := klog.FromContext(ctx)

	sliceHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice add", "slice", klog.KObj(slice))
			t.queue.Add(slice.Name)
		},
		UpdateFunc: func(old, new any) {
			oldSlice, ok := old.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			newSlice, ok := new.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			if loggerV := logger.V(6); loggerV.Enabled() {
				loggerV.Info("ResourceSlice update", "slice", klog.KObj(newSlice), "diff", cmp.Diff(oldSlice, newSlice))
			} else {
				logger.V(5).Info("ResourceSlice update", "slice", klog.KObj(newSlice))
			}
			t.queue.Add(newSlice.Name)
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			slice, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlice delete", "slice", klog.KObj(slice))
			t.queue.Add(slice.Name)
		},
	}
	_, err := t.resourceSlices.AddEventHandler(sliceHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for ResourceSlices: %w", err)
	}

	patchHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			patch, ok := obj.(*resourcealphaapi.ResourceSlicePatch)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlicePatch add", "patch", klog.KObj(patch))
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
		UpdateFunc: func(old, new any) {
			oldPatch, ok := old.(*resourcealphaapi.ResourceSlicePatch)
			if !ok {
				return
			}
			newPatch, ok := new.(*resourcealphaapi.ResourceSlicePatch)
			if !ok {
				return
			}
			if loggerV := logger.V(6); loggerV.Enabled() {
				loggerV.Info("ResourceSlicePatch update", "patch", klog.KObj(newPatch), "diff", cmp.Diff(oldPatch, newPatch))
			} else {
				logger.V(5).Info("ResourceSlicePatch update", "patch", klog.KObj(newPatch))
			}
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			patch, ok := obj.(*resourcealphaapi.ResourceSlicePatch)
			if !ok {
				return
			}
			logger.V(5).Info("ResourceSlicePatch delete", "patch", klog.KObj(patch))
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
	}
	_, err = t.resourceSlicePatches.AddEventHandler(patchHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for ResourceSlicePatches: %w", err)
	}

	classHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			class, ok := obj.(*resourceapi.DeviceClass)
			if !ok {
				return
			}
			logger.V(5).Info("DeviceClass add", "class", klog.KObj(class))
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
		UpdateFunc: func(old, new any) {
			oldClass, ok := old.(*resourceapi.DeviceClass)
			if !ok {
				return
			}
			newClass, ok := new.(*resourceapi.DeviceClass)
			if !ok {
				return
			}
			if loggerV := logger.V(6); loggerV.Enabled() {
				loggerV.Info("DeviceClass update", "class", klog.KObj(newClass), "diff", cmp.Diff(oldClass, newClass))
			} else {
				logger.V(5).Info("DeviceClass update", "class", klog.KObj(newClass))
			}
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
		DeleteFunc: func(obj any) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			class, ok := obj.(*resourceapi.ResourceSlice)
			if !ok {
				return
			}
			logger.V(5).Info("DeviceClass delete", "class", klog.KObj(class))
			for _, sliceName := range t.resourceSlices.GetIndexer().ListKeys() {
				t.queue.Add(sliceName)
			}
		},
	}
	_, err = t.deviceClasses.AddEventHandler(classHandler)
	if err != nil {
		return fmt.Errorf("failed to add event handler for DeviceClasses: %w", err)
	}

	return nil
}

// run is running in the background.
func (t *Tracker) run(ctx context.Context) {
	for t.processNextWorkItem(ctx) {
	}
}

func (t *Tracker) processNextWorkItem(ctx context.Context) bool {
	sliceName, shutdown := t.queue.Get()
	if shutdown {
		return false
	}
	logger := klog.FromContext(ctx)
	defer t.queue.Done(sliceName)

	// Panics are caught and treated like errors.
	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("internal error: %v", r)
			}
		}()
		err = t.syncSlice(klog.NewContext(ctx, klog.LoggerWithValues(logger, "sliceName", sliceName)), sliceName)
	}()

	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "processing ResourceSlice objects")
		t.queue.AddRateLimited(sliceName)

		// Return without removing the work item from the queue.
		// It will be retried.
		return true
	}

	t.queue.Forget(sliceName)
	return true
}

func (t *Tracker) syncSlice(ctx context.Context, name string) error {
	logger := klog.FromContext(ctx)
	obj, exists, err := t.resourceSlices.GetIndexer().GetByKey(name)
	if err != nil {
		return err
	}
	if !exists {
		return t.patchedResourceSlices.Delete(obj)
	}
	slice, ok := obj.(*resourceapi.ResourceSlice)
	if !ok {
		return fmt.Errorf("expected type to be %T, got %T", (*resourceapi.ResourceSlice)(nil), obj)
	}
	logger.V(5).Info("syncing slice")
	patchedSlice := slice.DeepCopy()

	patches := typedSlice[*resourcealphaapi.ResourceSlicePatch](t.resourceSlicePatches.GetIndexer().List())
	slices.SortFunc(patches, func(p1, p2 *resourcealphaapi.ResourceSlicePatch) int {
		return int(ptr.Deref(p1.Spec.Devices.Priority, 0) - ptr.Deref(p2.Spec.Devices.Priority, 0))
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
					return err
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
			switch {
			case device.Basic != nil:
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
					return fmt.Errorf("class %s: selector #%d: CEL compile error: %w", *filter.DeviceClassName, i, expr.Error)
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
					return fmt.Errorf("patch %s: selector #%d: CEL compile error: %w", patch.Name, i, expr.Error)
				}
				match, _, err := expr.DeviceMatches(ctx, cel.Device{Driver: patchedSlice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
				if err != nil {
					t.recorder.Eventf(patch, v1.EventTypeWarning, "CELRuntimeError", "selector #%d: runtime error: %v", i, err)
					continue devices
				}
				if !match {
					continue devices
				}
			}

			if utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminControlledDeviceAttributes) {
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

				switch {
				case device.Basic != nil:
					patchedSlice.Spec.Devices[dIndex].Basic.Attributes = newAttrs
					patchedSlice.Spec.Devices[dIndex].Basic.Capacity = newCaps
				}
			}
		}
	}

	return t.patchedResourceSlices.Add(patchedSlice)
}

func typedSlice[T any](objs []any) []T {
	typed := make([]T, 0, len(objs))
	for _, obj := range objs {
		typed = append(typed, obj.(T))
	}
	return typed
}
