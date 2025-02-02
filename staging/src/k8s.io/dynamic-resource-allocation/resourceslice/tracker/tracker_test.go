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
	"cmp"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/internal/workqueue"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestListPatchedResourceSlices(t *testing.T) {
	tests := map[string]struct {
		initialClasses        []*resourceapi.DeviceClass
		initialSlices         []*resourceapi.ResourceSlice
		initialPatches        []*resourcealphaapi.ResourceSlicePatch
		initialCachedSlices   []*resourceapi.ResourceSlice
		expectedPatchedSlices []*resourceapi.ResourceSlice
		adminAttrsDisabled    bool
	}{
		"add-slices-no-patches": {
			initialSlices: []*resourceapi.ResourceSlice{
				{ObjectMeta: metav1.ObjectMeta{Name: "s1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "s2"}},
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{
				{ObjectMeta: metav1.ObjectMeta{Name: "s1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "s2"}},
			},
		},
		"update-slices-no-patches": {
			initialCachedSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// no devices
						Devices: nil,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s2",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// no devices
						Devices: nil,
					},
				},
			},
			initialSlices: []*resourceapi.ResourceSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "s1",
					},
					Spec: resourceapi.ResourceSliceSpec{
						// devices!
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
						// devices!
						Devices: []resourceapi.Device{
							{Basic: &resourceapi.BasicDevice{}},
						},
					},
				},
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
			},
		},
		"delete-slices": {
			initialCachedSlices: []*resourceapi.ResourceSlice{
				{ObjectMeta: metav1.ObjectMeta{Name: "s1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "s2"}},
			},
			expectedPatchedSlices: []*resourceapi.ResourceSlice{},
		},
		"admin-attributes-disabled": {
			adminAttrsDisabled: true,
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"patch-all-slices": {
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"merge-attributes": {
			initialSlices: []*resourceapi.ResourceSlice{
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
										"test.example.com/removeMe": {StringValue: ptr.To("slice")},
										"removeMeToo":               {StringValue: ptr.To("slice")},
										"test.example.com/keepMe":   {StringValue: ptr.To("slice")},
										"keepMeToo":                 {StringValue: ptr.To("slice")},
									},
								},
							},
						},
					},
				},
			},
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"add-attribute-for-driver": {
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"add-attribute-for-pool": {
			initialSlices: []*resourceapi.ResourceSlice{
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
								Basic: &resourceapi.BasicDevice{},
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"add-attribute-for-device": {
			initialSlices: []*resourceapi.ResourceSlice{
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
								Name:  "device",
								Basic: &resourceapi.BasicDevice{},
							},
							{
								Name:  "wrong-device",
								Basic: &resourceapi.BasicDevice{},
							},
						},
					},
				},
			},
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"add-attribute-for-selector": {
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"selector-does-not-match": {
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"runtime-CEL-errors-skip-devices": {
			initialSlices: []*resourceapi.ResourceSlice{
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
									},
								},
							},
						},
					},
				},
			},
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		// TODO: how to check errors?
		// "invalid-CEL-expression-returns-error": {
		// 	initialSlices: []*resourceapi.ResourceSlice{
		// 		{
		// 			ObjectMeta: metav1.ObjectMeta{
		// 				Name: "slice",
		// 			},
		// 			Spec: resourceapi.ResourceSliceSpec{
		// 				Devices: []resourceapi.Device{
		// 					{
		// 						Basic: &resourceapi.BasicDevice{},
		// 					},
		// 				},
		// 			},
		// 		},
		// 	},
		// 	initialPatches: []*resourcealphaapi.ResourceSlicePatch{
		// 		{
		// 			ObjectMeta: metav1.ObjectMeta{
		// 				Name: "selector",
		// 			},
		// 			Spec: resourcealphaapi.ResourceSlicePatchSpec{
		// 				Devices: resourcealphaapi.DevicePatch{
		// 					Filter: &resourcealphaapi.DevicePatchFilter{
		// 						Selectors: []resourcealphaapi.DeviceSelector{
		// 							{
		// 								CEL: &resourcealphaapi.CELDeviceSelector{
		// 									Expression: `invalid`,
		// 								},
		// 							},
		// 						},
		// 					},
		// 				},
		// 			},
		// 		},
		// 	},
		// 	matchErr: gomega.MatchError(gomega.ContainSubstring("CEL compile error")),
		// },
		"add-attribute-for-device-class": {
			initialClasses: []*resourceapi.DeviceClass{
				{
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
				},
			},
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"filter-all-criteria": {
			initialClasses: []*resourceapi.DeviceClass{
				{
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
				},
			},
			initialSlices: []*resourceapi.ResourceSlice{
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
								Name:  "device",
								Basic: &resourceapi.BasicDevice{},
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
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
		},
		"priority": {
			initialSlices: []*resourceapi.ResourceSlice{
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
			initialPatches: []*resourcealphaapi.ResourceSlicePatch{
				{
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
				},
				{
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
				},
				{
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
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "medium-priority",
					},
					Spec: resourcealphaapi.ResourceSlicePatchSpec{
						Devices: resourcealphaapi.DevicePatch{
							Priority: ptr.To[int32](100),
							Attributes: map[resourcealphaapi.FullyQualifiedName]resourcealphaapi.NullableDeviceAttribute{
								"test.example.com/mediumPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium"),
									},
								},
								"test.example.com/highPriority": {
									DeviceAttribute: resourcealphaapi.DeviceAttribute{
										StringValue: ptr.To("medium"),
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
				},
				{
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
				},
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
											StringValue: ptr.To("medium"),
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
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminControlledDeviceAttributes, !test.adminAttrsDisabled)

			inputObjects := make([]runtime.Object, 0, len(test.initialSlices)+len(test.initialClasses))
			for _, obj := range test.initialSlices {
				inputObjects = append(inputObjects, obj.DeepCopyObject())
			}
			for _, obj := range test.initialClasses {
				inputObjects = append(inputObjects, obj.DeepCopyObject())
			}
			// Passing ResourceSlicePatches directly through here doesn't work
			// because that ultimately results in an incorrect guess at the
			// resource name based on the kind (adding "s" instead of "es"). The
			// same happens even for the Create workaround with the
			// managedFields-tracking client from NewClientset().
			kubeClient := fake.NewSimpleClientset(inputObjects...)
			for _, resourceSlicePatch := range test.initialPatches {
				_, err := kubeClient.ResourceV1alpha3().ResourceSlicePatches().Create(ctx, resourceSlicePatch, metav1.CreateOptions{})
				require.NoError(t, err)
			}
			informerFactory := informers.NewSharedInformerFactoryWithOptions(kubeClient, 10*time.Minute)

			var queue workqueue.Mock[string]
			tracker, err := newTracker(ctx, informerFactory)
			require.NoError(t, err, "unexpected tracker creation error")
			tracker.queue = &queue

			informerStop := make(chan struct{})
			informerFactory.Start(informerStop)
			var unsynced []reflect.Type
			for typ, isSynced := range informerFactory.WaitForCacheSync(informerStop) {
				if !isSynced {
					unsynced = append(unsynced, typ)
				}
			}
			require.Empty(t, unsynced, "informers failed to sync")
			t.Cleanup(func() {
				close(informerStop)
				informerFactory.Shutdown()
			})

			// Process work items in the queue until the queue is empty.
			// Processing races with informers adding new work items,
			// but the desired state should already be reached in the
			// first iteration, so all following iterations should be nops.
			tracker.run(ctx)
			t.Cleanup(tracker.Stop)

			// Check ResourceSlices
			patchedResourceSlices, err := tracker.ListPatchedResourceSlices()
			require.NoError(t, err, "list patched resource slices")
			sortResourceSlicesFunc := func(s1, s2 *resourceapi.ResourceSlice) int {
				return cmp.Compare(s1.Name, s2.Name)
			}
			slices.SortFunc(test.expectedPatchedSlices, sortResourceSlicesFunc)
			slices.SortFunc(patchedResourceSlices, sortResourceSlicesFunc)
			assert.Equal(t, test.expectedPatchedSlices, patchedResourceSlices)
		})
	}
}
