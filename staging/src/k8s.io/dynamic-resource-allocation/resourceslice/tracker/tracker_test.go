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
	stdcmp "cmp"
	"context"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
	"k8s.io/utils/ptr"
)

type handlerEventType string

const (
	handlerEventAdd    handlerEventType = "add"
	handlerEventUpdate handlerEventType = "update"
	handlerEventDelete handlerEventType = "delete"
)

type handlerEvent struct {
	event  handlerEventType
	oldObj *resourceapi.ResourceSlice
	newObj *resourceapi.ResourceSlice
}

type inputEventGenerator struct {
	addResourceSlice         func(slice *resourceapi.ResourceSlice)
	deleteResourceSlice      func(name string)
	addResourceSlicePatch    func(patch *resourcealphaapi.ResourceSlicePatch)
	deleteResourceSlicePatch func(name string)
	addDeviceClass           func(class *resourceapi.DeviceClass)
	deleteDeviceClass        func(name string)
}

func inputEventGeneratorForTest(ctx context.Context, t *testing.T, tracker *Tracker) inputEventGenerator {
	return inputEventGenerator{
		addResourceSlice: func(slice *resourceapi.ResourceSlice) {
			oldObj, exists, err := tracker.resourceSlices.GetIndexer().Get(slice)
			require.NoError(t, err)
			err = tracker.resourceSlices.GetIndexer().Add(slice)
			require.NoError(t, err)
			if !exists {
				tracker.resourceSliceAdd(ctx)(slice)
			} else if !apiequality.Semantic.DeepEqual(oldObj, slice) {
				tracker.resourceSliceUpdate(ctx)(oldObj, slice)
			}
		},
		deleteResourceSlice: func(name string) {
			oldObj, exists, err := tracker.resourceSlices.GetIndexer().GetByKey(name)
			require.NoError(t, err)
			require.True(t, exists, "deleting resource slice that was never created")
			err = tracker.resourceSlices.GetIndexer().Delete(oldObj)
			require.NoError(t, err)
			tracker.resourceSliceDelete(ctx)(oldObj)
		},
		addResourceSlicePatch: func(patch *resourcealphaapi.ResourceSlicePatch) {
			oldObj, exists, err := tracker.resourceSlicePatches.GetIndexer().Get(patch)
			require.NoError(t, err)
			err = tracker.resourceSlicePatches.GetIndexer().Add(patch)
			require.NoError(t, err)
			if !exists {
				tracker.resourceSlicePatchAdd(ctx)(patch)
			} else if !apiequality.Semantic.DeepEqual(oldObj, patch) {
				tracker.resourceSlicePatchUpdate(ctx)(oldObj, patch)
			}
		},
		deleteResourceSlicePatch: func(name string) {
			oldObj, exists, err := tracker.resourceSlicePatches.GetIndexer().GetByKey(name)
			require.NoError(t, err)
			require.True(t, exists, "deleting resource slice patch that was never created")
			err = tracker.resourceSlicePatches.GetIndexer().Delete(oldObj)
			require.NoError(t, err)
			tracker.resourceSlicePatchDelete(ctx)(oldObj)
		},
		addDeviceClass: func(class *resourceapi.DeviceClass) {
			oldObj, exists, err := tracker.deviceClasses.GetIndexer().Get(class)
			require.NoError(t, err)
			err = tracker.deviceClasses.GetIndexer().Add(class)
			require.NoError(t, err)
			if !exists {
				tracker.deviceClassAdd(ctx)(class)
			} else if !apiequality.Semantic.DeepEqual(oldObj, class) {
				tracker.deviceClassUpdate(ctx)(oldObj, class)
			}
		},
		deleteDeviceClass: func(name string) {
			oldObj, exists, err := tracker.deviceClasses.GetIndexer().GetByKey(name)
			require.NoError(t, err)
			require.True(t, exists, "deleting device class that was never created")
			err = tracker.deviceClasses.GetIndexer().Delete(oldObj)
			require.NoError(t, err)
			tracker.deviceClassDelete(ctx)(oldObj)
		},
	}
}

func TestListPatchedResourceSlices(t *testing.T) {
	tests := map[string]struct {
		adminAttrsDisabled    bool
		inputEvents           func(event inputEventGenerator)
		expectedPatchedSlices []*resourceapi.ResourceSlice
		expectHandlerEvents   func(t *testing.T, events []handlerEvent)
		expectEvents          func(t *assert.CollectT, events *v1.EventList)
		expectUnhandledErrors func(t *testing.T, errs []error)
	}{
		"add-slices-no-patches": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "s1"}})
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "s2"}})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{ObjectMeta: metav1.ObjectMeta{Name: "s1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "s2"}},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "s1", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "s2", events[1].newObj.Name)
			},
		},
		"update-slices-no-patches": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// no devices
						Devices: nil,
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// no devices
						Devices: nil,
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "no-change"}})

				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// devices!
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// devices!
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "no-change"}})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "no-change"}},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 5) {
					return
				}
				// The first events are adds.
				assert.Equal(t, handlerEventUpdate, events[3].event)
				assert.Equal(t, "s1", events[3].newObj.Name)
				assert.Equal(t, "s1", events[3].oldObj.Name)
				assert.Nil(t, events[3].oldObj.Spec.Devices)
				assert.NotNil(t, events[3].newObj.Spec.Devices)
				assert.Equal(t, handlerEventUpdate, events[4].event)
				assert.Equal(t, "s2", events[4].newObj.Name)
				assert.Equal(t, "s2", events[4].oldObj.Name)
				assert.Nil(t, events[4].oldObj.Spec.Devices)
				assert.NotNil(t, events[4].newObj.Spec.Devices)
			},
		},
		"delete-slices": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "s1"}})
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "s2"}})
				event.addResourceSlice(&resourceapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{Name: "keep-me"}})
				event.deleteResourceSlice("s1")
				event.deleteResourceSlice("s2")
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{ObjectMeta: metav1.ObjectMeta{Name: "keep-me"}},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 5) {
					return
				}
				// The first events are adds.
				assert.Equal(t, handlerEventDelete, events[3].event)
				assert.Equal(t, "s1", events[3].oldObj.Name)
				assert.Equal(t, handlerEventDelete, events[4].event)
				assert.Equal(t, "s2", events[4].oldObj.Name)
			},
		},
		"admin-attributes-disabled": {
			adminAttrsDisabled: true,
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-slices",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: nil,
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"patch-all-slices": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-slices",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: nil,
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventUpdate, events[1].event)
				assert.Equal(t, "slice", events[1].newObj.Name)
			},
		},
		"update-patch": {
			inputEvents: func(event inputEventGenerator) {
				patch := &resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Pool: ptr.To("pool-1"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				}
				event.addResourceSlicePatch(patch.DeepCopy())
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice-1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool-1",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice-2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool-2",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				patch.Spec.Devices.Filter.Pool = ptr.To("pool-2")
				event.addResourceSlicePatch(patch)
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice-1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool-1",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice-2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool-2",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 4) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice-1", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "slice-2", events[1].newObj.Name)

				assert.Equal(t, handlerEventUpdate, events[2].event)
				assert.Equal(t, handlerEventUpdate, events[3].event)
				assert.ElementsMatch(t, []string{"slice-1", "slice-2"}, []string{events[2].newObj.Name, events[3].newObj.Name})
			},
		},
		"merge-attributes": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "merge",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: nil,
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/removeMe": {
									NullValue: &resourcealphaapi.NullValue{},
								},
								"test.example.com/removeMeToo": {
									NullValue: &resourcealphaapi.NullValue{},
								},
								"test.example.com/keepMe": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{StringValue: ptr.To("patch")},
								},
								"test.example.com/keepMeToo": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{StringValue: ptr.To("patch")},
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/removeMe": {StringValue: ptr.To("slice")},
										"removeMeToo":               {StringValue: ptr.To("slice")},
										"test.example.com/keepMe":   {StringValue: ptr.To("slice")},
										"keepMeToo":                 {StringValue: ptr.To("slice")},
									},
								},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/keepMe":    {StringValue: ptr.To("patch")},
										"test.example.com/keepMeToo": {StringValue: ptr.To("patch")},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"add-attribute-for-driver": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "driver",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Driver: ptr.To("test.example.com"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "wrong-driver", events[1].newObj.Name)
			},
		},
		"add-attribute-for-pool": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pool",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Pool: ptr.To("pool"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-pool",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "other",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-pool",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "other",
						},
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "wrong-pool", events[1].newObj.Name)
			},
		},
		"add-attribute-for-device": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "device",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Device: ptr.To("device"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Name:  "device",
								Basic: &resourceapi.BasicDevice{},
							},
							{
								Name:  "wrong-device",
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Name: "device",
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
							{
								Name:  "wrong-device",
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"add-attribute-for-selector": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `device.driver == "test.example.com"`,
										},
									},
								},
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "wrong-driver", events[1].newObj.Name)
			},
		},
		"selector-does-not-match": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `true`,
										},
									},
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `false`,
										},
									},
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `true`,
										},
									},
								},
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"runtime-CEL-errors-skip-devices": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `device.attributes["test.example.com"].deviceAttr`,
										},
									},
								},
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"deviceAttr": {BoolValue: ptr.To(true)},
									},
								},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"deviceAttr": {BoolValue: ptr.To(true)},
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectEvents: func(t *assert.CollectT, events *v1.EventList) {
				if !assert.Len(t, events.Items, 1) {
					return
				}
				assert.Equal(t, "selector", events.Items[0].InvolvedObject.Name)
				assert.Equal(t, "CELRuntimeError", events.Items[0].Reason)
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"invalid-CEL-expression-throws-error": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `invalid`,
										},
									},
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{},
			expectUnhandledErrors: func(t *testing.T, errs []error) {
				if !assert.Len(t, errs, 1) {
					return
				}
				assert.ErrorContains(t, errs[0], "CEL compile error")
			},
		},
		"add-attribute-for-device-class": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceClass(&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: "class.example.com",
					},
					Spec: resourceapi.DeviceClassSpec{
						Selectors: []resourceapi.DeviceSelector{
							{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: `device.driver == "test.example.com"`,
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "device-class",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								DeviceClassName: ptr.To("class.example.com"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "wrong-driver", events[1].newObj.Name)
			},
		},
		"filter-all-criteria": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceClass(&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: "class.example.com",
					},
					Spec: resourceapi.DeviceClassSpec{
						Selectors: []resourceapi.DeviceSelector{
							{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: `device.driver == "test.example.com"`,
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-criteria",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								DeviceClassName: ptr.To("class.example.com"),
								Driver:          ptr.To("test.example.com"),
								Pool:            ptr.To("pool"),
								Device:          ptr.To("device"),
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `true`,
										},
									},
								},
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Name:  "device",
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "test.example.com",
						Pool: resourceapi.ResourcePool{
							Name: "pool",
						},
						Devices: []resourceapi.Device{
							{
								Name: "device",
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/patchCap": {
											Value: resource.MustParse("1"),
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "wrong-driver",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Driver: "wrong.example.com",
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 2) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[1].event)
				assert.Equal(t, "wrong-driver", events[1].newObj.Name)
			},
		},
		"priority": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "negative-priority",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](-1),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/negativePriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("negative"),
									},
								},
								"test.example.com/noPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("negative"),
									},
								},
								"test.example.com/lowPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("negative"),
									},
								},
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("negative"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("negative"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/negativePriority": {
									Value: resource.MustParse("-1"),
								},
								"test.example.com/noPriority": {
									Value: resource.MustParse("-1"),
								},
								"test.example.com/lowPriority": {
									Value: resource.MustParse("-1"),
								},
								"test.example.com/mediumPriority": {
									Value: resource.MustParse("-1"),
								},
								"test.example.com/highPriority": {
									Value: resource.MustParse("-1"),
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "no-priority",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: nil,
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/noPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("none"),
									},
								},
								"test.example.com/lowPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("none"),
									},
								},
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("none"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("none"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/noPriority": {
									Value: resource.MustParse("0"),
								},
								"test.example.com/lowPriority": {
									Value: resource.MustParse("0"),
								},
								"test.example.com/mediumPriority": {
									Value: resource.MustParse("0"),
								},
								"test.example.com/highPriority": {
									Value: resource.MustParse("0"),
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "low-priority",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](1),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/lowPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("low"),
									},
								},
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("low"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("low"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/lowPriority": {
									Value: resource.MustParse("1"),
								},
								"test.example.com/mediumPriority": {
									Value: resource.MustParse("1"),
								},
								"test.example.com/highPriority": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "medium-priority-old",
						CreationTimestamp: metav1.Date(2024, time.January, 1, 0, 0, 0, 0, time.UTC),
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](100),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium-old"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium-old"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/mediumPriority": {
									Value: resource.MustParse("100"),
								},
								"test.example.com/highPriority": {
									Value: resource.MustParse("100"),
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "medium-priority-new",
						CreationTimestamp: metav1.Date(2025, time.January, 1, 0, 0, 0, 0, time.UTC),
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](100),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium-new"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium-new"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/mediumPriority": {
									Value: resource.MustParse("100"),
								},
								"test.example.com/highPriority": {
									Value: resource.MustParse("100"),
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "high-priority",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](1000),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("high"),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/highPriority": {
									Value: resource.MustParse("1000"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/negativePriority": {
											StringValue: ptr.To("negative"),
										},
										"test.example.com/noPriority": {
											StringValue: ptr.To("none"),
										},
										"test.example.com/lowPriority": {
											StringValue: ptr.To("low"),
										},
										"test.example.com/mediumPriority": {
											StringValue: ptr.To("medium-old"),
										},
										"test.example.com/highPriority": {
											StringValue: ptr.To("high"),
										},
									},
									Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
										"test.example.com/negativePriority": {
											Value: resource.MustParse("-1"),
										},
										"test.example.com/noPriority": {
											Value: resource.MustParse("0"),
										},
										"test.example.com/lowPriority": {
											Value: resource.MustParse("1"),
										},
										"test.example.com/mediumPriority": {
											Value: resource.MustParse("100"),
										},
										"test.example.com/highPriority": {
											Value: resource.MustParse("1000"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
		"update-patched-slice": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-slices",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Device: ptr.To("device-1"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
						},
					},
				})
				oneDevice := []resourceapi.Device{
					{Name: "device-1", Basic: &resourceapi.BasicDevice{}},
				}
				threeDevices := []resourceapi.Device{
					{Name: "device-0", Basic: &resourceapi.BasicDevice{}},
					{Name: "device-1", Basic: &resourceapi.BasicDevice{}},
					{Name: "device-2", Basic: &resourceapi.BasicDevice{}},
				}
				devicesAdded := &resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "devices-added",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: oneDevice,
					},
				}
				devicesRemoved := &resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "devices-removed",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: threeDevices,
					},
				}
				event.addResourceSlice(devicesAdded.DeepCopy())
				devicesAdded.Spec.Devices = threeDevices
				event.addResourceSlice(devicesAdded)
				event.addResourceSlice(devicesRemoved.DeepCopy())
				devicesRemoved.Spec.Devices = oneDevice
				event.addResourceSlice(devicesRemoved)
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "devices-added",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{Name: "device-0", Basic: &resourceapi.BasicDevice{}},
							{
								Name: "device-1",
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
								},
							},
							{Name: "device-2", Basic: &resourceapi.BasicDevice{}},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "devices-removed",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Name: "device-1",
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 4) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "devices-added", events[0].newObj.Name)
				assert.Equal(t, handlerEventUpdate, events[1].event)
				assert.Equal(t, "devices-added", events[1].newObj.Name)
				assert.Equal(t, handlerEventAdd, events[2].event)
				assert.Equal(t, "devices-removed", events[2].newObj.Name)
				assert.Equal(t, handlerEventUpdate, events[3].event)
				assert.Equal(t, "devices-removed", events[3].newObj.Name)
			},
		},
		"patches-cannot-match-patched-attributes": {
			inputEvents: func(event inputEventGenerator) {
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "add-attr",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](50),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/patchAttr": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("value"),
									},
								},
							},
						},
					},
				})
				event.addResourceSlicePatch(&resourcealphaapi.ResourceSlicePatch{
					ObjectMeta: metav1.ObjectMeta{
						Name: "add-cap-for-attr",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](100),
							Filter: &resourcealphaapi.DevicePatchFilter{
								Selectors: []resourcealphaapi.DeviceSelector{
									{
										CEL: &resourcealphaapi.CELDeviceSelector{
											Expression: `"test.example.com" in device.attributes`,
										},
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"test.example.com/patchCap": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				})
				event.addResourceSlice(&resourceapi.ResourceSlice{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				})
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "slice",
					},
					Spec: resourceapi.ResourceSliceSpec{
						Devices: []resourceapi.Device{
							{
								Basic: &resourceapi.BasicDevice{
									Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
										"test.example.com/patchAttr": {
											StringValue: ptr.To("value"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectHandlerEvents: func(t *testing.T, events []handlerEvent) {
				if !assert.Len(t, events, 1) {
					return
				}
				assert.Equal(t, handlerEventAdd, events[0].event)
				assert.Equal(t, "slice", events[0].newObj.Name)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			kubeClient := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactoryWithOptions(kubeClient, 10*time.Minute)

			var handlerEvents []handlerEvent
			handler := cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					handlerEvents = append(handlerEvents, handlerEvent{event: handlerEventAdd, newObj: obj.(*resourceapi.ResourceSlice)})
				},
				UpdateFunc: func(oldObj, newObj interface{}) {
					handlerEvents = append(handlerEvents, handlerEvent{event: handlerEventUpdate, oldObj: oldObj.(*resourceapi.ResourceSlice), newObj: newObj.(*resourceapi.ResourceSlice)})
				},
				DeleteFunc: func(obj interface{}) {
					handlerEvents = append(handlerEvents, handlerEvent{event: handlerEventDelete, oldObj: obj.(*resourceapi.ResourceSlice)})
				},
			}

			opts := Options{
				EnableAdminControlledAttributes: !test.adminAttrsDisabled,
				KubeClient:                      kubeClient,
			}
			tracker, err := newTracker(ctx, informerFactory, opts)
			require.NoError(t, err)
			var unhandledErrors []error
			tracker.handleError = func(_ context.Context, err error, _ string, _ ...any) {
				unhandledErrors = append(unhandledErrors, err)
			}
			tracker.AddEventHandler(handler)

			if test.inputEvents != nil {
				test.inputEvents(inputEventGeneratorForTest(ctx, t, tracker))
			}

			expectHandlerEvents := test.expectHandlerEvents
			if expectHandlerEvents == nil {
				expectHandlerEvents = func(t *testing.T, events []handlerEvent) {
					assert.Empty(t, events)
				}
			}
			expectHandlerEvents(t, handlerEvents)

			expectUnhandledErrors := test.expectUnhandledErrors
			if expectUnhandledErrors == nil {
				expectUnhandledErrors = func(t *testing.T, errs []error) {
					assert.Empty(t, errs)
				}
			}
			expectUnhandledErrors(t, unhandledErrors)

			// Check ResourceSlices
			patchedResourceSlices, err := tracker.ListPatchedResourceSlices()
			require.NoError(t, err, "list patched resource slices")
			sortResourceSlicesFunc := func(s1, s2 *resourceapi.ResourceSlice) int {
				return stdcmp.Compare(s1.Name, s2.Name)
			}
			slices.SortFunc(test.expectedPatchedSlices, sortResourceSlicesFunc)
			slices.SortFunc(patchedResourceSlices, sortResourceSlicesFunc)
			assert.Equal(t, test.expectedPatchedSlices, patchedResourceSlices)
			expectEvents := test.expectEvents
			if expectEvents == nil {
				expectEvents = func(t *assert.CollectT, events *v1.EventList) {
					assert.Empty(t, events.Items)
				}
			}
			// Events are generated asynchronously. While shutting down the event recorder will flush all
			// pending events, it is not possible to determine when exactly that flush is complete.
			assert.EventuallyWithT(
				t,
				func(t *assert.CollectT) {
					events, err := kubeClient.CoreV1().Events("").List(ctx, metav1.ListOptions{})
					require.NoError(t, err, "list events")
					expectEvents(t, events)
				},
				1*time.Second,
				10*time.Millisecond,
				"did not observe expected events",
			)
		})
	}
}

func BenchmarkEventHandlers(b *testing.B) {
	benchmarks := map[string]struct {
		resourceSlices []*resourceapi.ResourceSlice
		patches        []*resourcealphaapi.ResourceSlicePatch
		loop           func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int)
	}{
		"resource-slice-add-no-patches": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				resourceSlices := make([]*resourceapi.ResourceSlice, 1000)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Devices: slices.Repeat([]resourceapi.Device{{Basic: &resourceapi.BasicDevice{}}}, 64),
						},
					}
				}
				return resourceSlices
			}(),
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[i%len(resourceSlices)])
			},
		},
		"one-patch-to-many-slices-add-patch": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				resourceSlices := make([]*resourceapi.ResourceSlice, 500)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Devices: slices.Repeat([]resourceapi.Device{{Basic: &resourceapi.BasicDevice{}}}, 64),
						},
					}
				}
				return resourceSlices
			}(),
			patches: []*resourcealphaapi.ResourceSlicePatch{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: nil, // all slices
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"dra.example.com/bench": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										BoolValue: ptr.To(true),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"dra.example.com/bench": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSlicePatchAdd(ctx)(patches[i%len(patches)])
			},
		},
		"one-patch-to-many-slices-add-slice": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				resourceSlices := make([]*resourceapi.ResourceSlice, 500)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Devices: slices.Repeat([]resourceapi.Device{{Basic: &resourceapi.BasicDevice{}}}, 64),
						},
					}
				}
				return resourceSlices
			}(),
			patches: []*resourcealphaapi.ResourceSlicePatch{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: nil, // all slices
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"dra.example.com/bench": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										BoolValue: ptr.To(true),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"dra.example.com/bench": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[i%len(resourceSlices)])
			},
		},
		"one-patched-device-among-many-slices-add-patch": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				nSlices := 500
				nDevices := 64
				resourceSlices := make([]*resourceapi.ResourceSlice, nSlices)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Pool: resourceapi.ResourcePool{
								Name: "pool-" + strconv.Itoa(i),
							},
							Devices: func() []resourceapi.Device {
								devices := make([]resourceapi.Device, nDevices)
								for j := range devices {
									devices[j] = resourceapi.Device{
										Name:  "device-" + strconv.Itoa(j),
										Basic: &resourceapi.BasicDevice{},
									}
								}
								return devices
							}(),
						},
					}
				}
				resourceSlices[nSlices/2].Spec.Devices[nDevices/2].Name = "patchme"
				return resourceSlices
			}(),
			patches: []*resourcealphaapi.ResourceSlicePatch{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Device: ptr.To("patchme"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"dra.example.com/bench": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										BoolValue: ptr.To(true),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"dra.example.com/bench": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSlicePatchAdd(ctx)(patches[i%len(patches)])
			},
		},
		"one-patched-device-among-many-slices-add-slice": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				resourceSlices := make([]*resourceapi.ResourceSlice, 500)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Pool: resourceapi.ResourcePool{
								Name: "pool-" + strconv.Itoa(i),
							},
							Devices: func() []resourceapi.Device {
								nDevices := 64
								devices := slices.Repeat([]resourceapi.Device{{Basic: &resourceapi.BasicDevice{}}}, nDevices)
								devices[nDevices/2].Name = "patchme"
								return devices
							}(),
						},
					}
				}
				return resourceSlices
			}(),
			patches: []*resourcealphaapi.ResourceSlicePatch{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Filter: &resourcealphaapi.DevicePatchFilter{
								Pool:   ptr.To("pool-250"),
								Device: ptr.To("patchme"),
							},
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"dra.example.com/bench": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										BoolValue: ptr.To(true),
									},
								},
							},
							Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
								"dra.example.com/bench": {
									Value: resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[250]) // the slice affected by the patch
			},
		},
		"one-patch-for-each-of-many-slices-add-patch": {
			resourceSlices: func() []*resourceapi.ResourceSlice {
				resourceSlices := make([]*resourceapi.ResourceSlice, 500)
				for i := range resourceSlices {
					resourceSlices[i] = &resourceapi.ResourceSlice{
						ObjectMeta: metav1.ObjectMeta{
							Name: "slice-" + strconv.Itoa(i),
						},
						Spec: resourceapi.ResourceSliceSpec{
							Pool: resourceapi.ResourcePool{
								Name: "pool-" + strconv.Itoa(i),
							},
							Devices: slices.Repeat([]resourceapi.Device{{Basic: &resourceapi.BasicDevice{}}}, 64),
						},
					}
				}
				return resourceSlices
			}(),
			patches: func() []*resourcealphaapi.ResourceSlicePatch {
				patches := make([]*resourcealphaapi.ResourceSlicePatch, 500)
				for i := range patches {
					patches[i] = &resourcealphaapi.ResourceSlicePatch{
						ObjectMeta: metav1.ObjectMeta{
							Name: "patch-" + strconv.Itoa(i),
						},
						Spec: resourcealphaapi.ResourceSlicePatchSpec{
							Devices: resourcealphaapi.DevicePatch{
								Filter: &resourcealphaapi.DevicePatchFilter{
									Pool: ptr.To("pool-" + strconv.Itoa(i)),
								},
								Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
									"dra.example.com/bench": {
										DeviceAttribute: resourcealphaapi.DeviceAttribute{
											BoolValue: ptr.To(true),
										},
									},
								},
								Capacity: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.DeviceCapacity{
									"dra.example.com/bench": {
										Value: resource.MustParse("1"),
									},
								},
							},
						},
					}
				}
				return patches
			}(),
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.ResourceSlicePatch, i int) {
				tracker.resourceSlicePatchAdd(ctx)(patches[i%len(patches)])
			},
		},
	}

	newBenchTracker := func(ctx context.Context) *Tracker {
		kubeClient := fake.NewSimpleClientset()
		informerFactory := informers.NewSharedInformerFactoryWithOptions(kubeClient, 10*time.Minute)
		opts := Options{
			EnableAdminControlledAttributes: true,
			KubeClient:                      kubeClient,
		}
		tracker, err := newTracker(ctx, informerFactory, opts)
		require.NoError(b, err)
		tracker.handleError = func(_ context.Context, err error, _ string, _ ...any) {
			b.Error("unexpected unhandled error:", err)
		}
		return tracker
	}

	for name, benchmark := range benchmarks {
		b.Run(name, func(b *testing.B) {
			logger, ctx := ktesting.NewTestContext(b)
			ctx = klog.NewContext(ctx, logger.V(2))
			tracker := newBenchTracker(ctx)

			for _, slice := range benchmark.resourceSlices {
				err := tracker.resourceSlices.GetIndexer().Add(slice)
				require.NoError(b, err)
			}

			for _, patch := range benchmark.patches {
				err := tracker.resourceSlicePatches.GetIndexer().Add(patch)
				require.NoError(b, err)
			}

			b.ResetTimer()
			for i := range b.N {
				benchmark.loop(ctx, b, tracker, benchmark.resourceSlices, benchmark.patches, i)
			}
		})
	}
}

func BenchmarkInitialSync(b *testing.B) {
	newInformerFactory := func() informers.SharedInformerFactory {
		resourceSlices := make([]runtime.Object, 500)
		for i := range resourceSlices {
			resourceSlices[i] = &resourceapi.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name: "slice-" + strconv.Itoa(i),
				},
				Spec: resourceapi.ResourceSliceSpec{
					Pool: resourceapi.ResourcePool{
						Name: "pool-" + strconv.Itoa(i),
					},
					Driver: "test-" + strconv.Itoa(i%10) + ".example.com",
					Devices: func() []resourceapi.Device {
						devices := make([]resourceapi.Device, 128)
						for j := range devices {
							devices[j] = resourceapi.Device{
								Name:  "dev-" + strconv.Itoa(j),
								Basic: &resourceapi.BasicDevice{},
							}
						}
						return devices
					}(),
				},
			}
		}
		kubeClient := fake.NewSimpleClientset(resourceSlices...)
		informerFactory := informers.NewSharedInformerFactoryWithOptions(kubeClient, 10*time.Minute)
		return informerFactory
	}

	waitForSync := func(b *testing.B, reg cache.ResourceEventHandlerRegistration) {
		require.Eventually(b, reg.HasSynced, 1*time.Second, 1*time.Millisecond)
	}

	handler := cache.ResourceEventHandlerFuncs{}

	b.Run("plain-informer", func(b *testing.B) {
		logger, ctx := ktesting.NewTestContext(b)
		ctx = klog.NewContext(ctx, logger.V(2))

		loop := func() {
			b.StopTimer()
			ctx, cancel := context.WithCancel(ctx)
			informerFactory := newInformerFactory()
			b.StartTimer()
			defer func() {
				b.StopTimer()
				cancel()
				informerFactory.Shutdown()
				b.StartTimer()
			}()

			reg, err := informerFactory.Resource().V1beta1().ResourceSlices().Informer().AddEventHandler(handler)
			require.NoError(b, err)
			informerFactory.Start(ctx.Done())
			waitForSync(b, reg)
		}

		b.ResetTimer()
		for range b.N {
			loop()
		}
	})

	b.Run("tracker", func(b *testing.B) {
		logger, ctx := ktesting.NewTestContext(b)
		ctx = klog.NewContext(ctx, logger.V(2))
		ctx, cancel := context.WithCancel(ctx)
		b.Cleanup(cancel)

		loop := func() {
			b.StopTimer()
			ctx, cancel := context.WithCancel(ctx)
			informerFactory := newInformerFactory()
			b.StartTimer()
			defer func() {
				b.StopTimer()
				cancel()
				informerFactory.Shutdown()
				b.StartTimer()
			}()

			opts := Options{
				EnableAdminControlledAttributes: false,
			}
			tracker, err := StartTracker(ctx, informerFactory, opts)
			require.NoError(b, err)
			reg := tracker.AddEventHandler(handler)
			informerFactory.Start(ctx.Done())
			waitForSync(b, reg)
		}

		b.ResetTimer()
		for range b.N {
			loop()
		}
	})
}
