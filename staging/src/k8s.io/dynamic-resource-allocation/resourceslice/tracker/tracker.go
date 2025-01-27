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
	"fmt"
	"maps"
	"slices"
	"strings"

	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/labels"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	resourcealphalisters "k8s.io/client-go/listers/resource/v1alpha3"
	resourcelisters "k8s.io/client-go/listers/resource/v1beta1"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

type Tracker struct {
	resourceSlices       resourcelisters.ResourceSliceLister
	resourceSlicePatches resourcealphalisters.ResourceSlicePatchLister
	deviceClasses        resourcelisters.DeviceClassLister
	celCache             *cel.Cache
}

func NewTracker(informerFactory informers.SharedInformerFactory) *Tracker {
	t := &Tracker{
		resourceSlices:       informerFactory.Resource().V1beta1().ResourceSlices().Lister(),
		resourceSlicePatches: informerFactory.Resource().V1alpha3().ResourceSlicePatches().Lister(),
		deviceClasses:        informerFactory.Resource().V1beta1().DeviceClasses().Lister(),
		celCache:             cel.NewCache(10), // TODO: share cache with scheduler
	}
	return t
}

func (t *Tracker) ListPatchedResourceSlices(ctx context.Context) ([]*resourceapi.ResourceSlice, error) {
	patches, err := t.resourceSlicePatches.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	slices.SortFunc(patches, func(p1, p2 *resourcealphaapi.ResourceSlicePatch) int {
		return int(ptr.Deref(p1.Spec.Devices.Priority, 0) - ptr.Deref(p2.Spec.Devices.Priority, 0))
	})
	slices, err := t.resourceSlices.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	patchedSlices := make([]*resourceapi.ResourceSlice, 0, len(slices))
	for _, slice := range slices {
		patchedSlice := slice.DeepCopy()
		patchedSlices = append(patchedSlices, patchedSlice)
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
					class, err := t.deviceClasses.Get(*filter.DeviceClassName)
					if err != nil {
						return nil, err
					}
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
						return nil, fmt.Errorf("class %s: selector #%d: CEL compile error: %w", *filter.DeviceClassName, i, expr.Error)
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
						return nil, fmt.Errorf("patch %s: selector #%d: CEL compile error: %w", patch.Name, i, expr.Error)
					}
					match, _, err := expr.DeviceMatches(ctx, cel.Device{Driver: patchedSlice.Spec.Driver, Attributes: deviceAttributes, Capacity: deviceCapacity})
					if err != nil {
						// TODO: generate event for ResourceSlicePatch
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
						if val.NullValue != nil {
							unqualifiedKey := strings.TrimPrefix(string(key), patchedSlice.Spec.Driver+"/")
							delete(newAttrs, resourceapi.QualifiedName(key))
							delete(newAttrs, resourceapi.QualifiedName(unqualifiedKey))
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
	}
	return patchedSlices, nil
}
