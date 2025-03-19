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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	addResourceSlice      func(slice *resourceapi.ResourceSlice)
	deleteResourceSlice   func(name string)
	addDeviceTaintRule    func(taintRule *resourcealphaapi.DeviceTaintRule)
	deleteDeviceTaintRule func(name string)
	addDeviceClass        func(class *resourceapi.DeviceClass)
	deleteDeviceClass     func(name string)
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
		addDeviceTaintRule: func(taintRule *resourcealphaapi.DeviceTaintRule) {
			oldObj, exists, err := tracker.deviceTaints.GetIndexer().Get(taintRule)
			require.NoError(t, err)
			err = tracker.deviceTaints.GetIndexer().Add(taintRule)
			require.NoError(t, err)
			if !exists {
				tracker.deviceTaintAdd(ctx)(taintRule)
			} else if !apiequality.Semantic.DeepEqual(oldObj, taintRule) {
				tracker.deviceTaintUpdate(ctx)(oldObj, taintRule)
			}
		},
		deleteDeviceTaintRule: func(name string) {
			oldObj, exists, err := tracker.deviceTaints.GetIndexer().GetByKey(name)
			require.NoError(t, err)
			require.True(t, exists, "deleting DeviceTaintRule that was never created")
			err = tracker.deviceTaints.GetIndexer().Delete(oldObj)
			require.NoError(t, err)
			tracker.deviceTaintDelete(ctx)(oldObj)
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
	now, _ := time.Parse(time.RFC3339, "2006-01-02T15:04:05Z")

	tests := map[string]struct {
		deviceTaintsDisabled  bool
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-slices",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: nil,
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				taintRule := &resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "taintRule",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Pool: ptr.To("pool-1"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
						},
					},
				}
				event.addDeviceTaintRule(taintRule.DeepCopy())
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
				taintRule.Spec.DeviceSelector.Pool = ptr.To("pool-2")
				event.addDeviceTaintRule(taintRule)
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
		"merge-taints": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "merge",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: nil,
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:    "example.com/taint2",
										Value:  "tainted2",
										Effect: resourceapi.DeviceTaintEffectNoSchedule,
									}},
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
									Taints: []resourceapi.DeviceTaint{
										{
											Key:    "example.com/taint2",
											Value:  "tainted2",
											Effect: resourceapi.DeviceTaintEffectNoSchedule,
										},
										{
											Key:       "example.com/taint",
											Value:     "tainted",
											Effect:    resourceapi.DeviceTaintEffectNoExecute,
											TimeAdded: &metav1.Time{Time: now},
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
		"add-taint-for-driver": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "driver",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Driver: ptr.To("test.example.com"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
		"add-taint-for-pool": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pool",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Pool: ptr.To("pool"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
		"add-taint-for-device": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "device",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Device: ptr.To("device"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Selectors: []resourcealphaapi.DeviceSelector{
								{
									CEL: &resourcealphaapi.CELDeviceSelector{
										Expression: `device.driver == "test.example.com"`,
									},
								},
							},
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
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
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Selectors: []resourcealphaapi.DeviceSelector{
								{
									CEL: &resourcealphaapi.CELDeviceSelector{
										Expression: `device.attributes["test.example.com"].deviceAttr`,
									},
								},
							},
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "selector",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Selectors: []resourcealphaapi.DeviceSelector{
								{
									CEL: &resourcealphaapi.CELDeviceSelector{
										Expression: `invalid`,
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
		"add-taint-for-device-class": {
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "device-class",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							DeviceClassName: ptr.To("class.example.com"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-criteria",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
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
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
		"update-patched-slice": {
			inputEvents: func(event inputEventGenerator) {
				event.addDeviceTaintRule(&resourcealphaapi.DeviceTaintRule{
					ObjectMeta: metav1.ObjectMeta{
						Name: "all-slices",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Device: ptr.To("device-1"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
									Taints: []resourceapi.DeviceTaint{{
										Key:       "example.com/taint",
										Value:     "tainted",
										Effect:    resourceapi.DeviceTaintEffectNoExecute,
										TimeAdded: &metav1.Time{Time: now},
									}},
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
				EnableDeviceTaints: !test.deviceTaintsDisabled,
				SliceInformer:      informerFactory.Resource().V1beta1().ResourceSlices(),
				TaintInformer:      informerFactory.Resource().V1alpha3().DeviceTaintRules(),
				ClassInformer:      informerFactory.Resource().V1beta1().DeviceClasses(),
				KubeClient:         kubeClient,
			}
			tracker, err := newTracker(ctx, opts)
			require.NoError(t, err)
			var unhandledErrors []error
			tracker.handleError = func(_ context.Context, err error, _ string, _ ...any) {
				unhandledErrors = append(unhandledErrors, err)
			}
			_, _ = tracker.AddEventHandler(handler)

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
	now := time.Now()
	benchmarks := map[string]struct {
		resourceSlices []*resourceapi.ResourceSlice
		taintRules     []*resourcealphaapi.DeviceTaintRule
		loop           func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, taintRules []*resourcealphaapi.DeviceTaintRule, i int)
	}{
		"resource-slice-add-no-taint-rules": {
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
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, _ []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[i%len(resourceSlices)])
			},
		},
		"one-patch-to-many-slices-add-taint-rule": {
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
			taintRules: []*resourcealphaapi.DeviceTaintRule{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "taintRule",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: nil, // all slices
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, taintRules []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.deviceTaintAdd(ctx)(taintRules[i%len(taintRules)])
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
			taintRules: []*resourcealphaapi.DeviceTaintRule{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "taintRule",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: nil, // all slices
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, _ []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[i%len(resourceSlices)])
			},
		},
		"one-patched-device-among-many-slices-add-taint-rule": {
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
			taintRules: []*resourcealphaapi.DeviceTaintRule{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "taintRule",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Device: ptr.To("patchme"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, taintRules []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.deviceTaintAdd(ctx)(taintRules[i%len(taintRules)])
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
			taintRules: []*resourcealphaapi.DeviceTaintRule{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "patch",
					},
					Spec: resourcealphaapi.DeviceTaintRuleSpec{
						DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
							Pool:   ptr.To("pool-250"),
							Device: ptr.To("patchme"),
						},
						Taint: resourcealphaapi.DeviceTaint{
							Key:       "example.com/taint",
							Value:     "tainted",
							Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
							TimeAdded: &metav1.Time{Time: now},
						},
					},
				},
			},
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, patches []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.resourceSliceAdd(ctx)(resourceSlices[250]) // the slice affected by the patch
			},
		},
		"one-patch-for-each-of-many-slices-add-taint-rule": {
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
			taintRules: func() []*resourcealphaapi.DeviceTaintRule {
				patches := make([]*resourcealphaapi.DeviceTaintRule, 500)
				for i := range patches {
					patches[i] = &resourcealphaapi.DeviceTaintRule{
						ObjectMeta: metav1.ObjectMeta{
							Name: "taint-rule-" + strconv.Itoa(i),
						},
						Spec: resourcealphaapi.DeviceTaintRuleSpec{
							DeviceSelector: &resourcealphaapi.DeviceTaintSelector{
								Pool: ptr.To("pool-" + strconv.Itoa(i)),
							},
							Taint: resourcealphaapi.DeviceTaint{
								Key:       "example.com/taint",
								Value:     "tainted",
								Effect:    resourcealphaapi.DeviceTaintEffectNoExecute,
								TimeAdded: &metav1.Time{Time: now},
							},
						},
					}
				}
				return patches
			}(),
			loop: func(ctx context.Context, b *testing.B, tracker *Tracker, resourceSlices []*resourceapi.ResourceSlice, taintRules []*resourcealphaapi.DeviceTaintRule, i int) {
				tracker.deviceTaintAdd(ctx)(taintRules[i%len(taintRules)])
			},
		},
	}

	newBenchTracker := func(ctx context.Context) *Tracker {
		kubeClient := fake.NewSimpleClientset()
		informerFactory := informers.NewSharedInformerFactoryWithOptions(kubeClient, 10*time.Minute)
		opts := Options{
			EnableDeviceTaints: true,
			SliceInformer:      informerFactory.Resource().V1beta1().ResourceSlices(),
			TaintInformer:      informerFactory.Resource().V1alpha3().DeviceTaintRules(),
			ClassInformer:      informerFactory.Resource().V1beta1().DeviceClasses(),
			KubeClient:         kubeClient,
		}
		tracker, err := newTracker(ctx, opts)
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

			for _, taintRule := range benchmark.taintRules {
				err := tracker.deviceTaints.GetIndexer().Add(taintRule)
				require.NoError(b, err)
			}

			b.ResetTimer()
			for i := range b.N {
				benchmark.loop(ctx, b, tracker, benchmark.resourceSlices, benchmark.taintRules, i)
			}
		})
	}
}
