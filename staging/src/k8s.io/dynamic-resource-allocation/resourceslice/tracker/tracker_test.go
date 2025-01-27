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
	"reflect"
	"testing"
	"time"

	"github.com/onsi/gomega"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestListPatchedResourceSlices(t *testing.T) {
	tests := map[string]struct {
		adminAttrsDisabled   bool
		resourceSlices       []*resourceapi.ResourceSlice
		resourceSlicePatches []*resourcealphaapi.ResourceSlicePatch
		deviceClasses        []*resourceapi.DeviceClass
		expected             []*resourceapi.ResourceSlice
		expectedErr          error
		matchErr             gomega.OmegaMatcher
	}{
		"no slices": {
			resourceSlices:       []*resourceapi.ResourceSlice{},
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{},
			expected:             []*resourceapi.ResourceSlice{},
		},
		"no patches": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{},
			expected: []*resourceapi.ResourceSlice{
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
		"admin attributes disabled": {
			adminAttrsDisabled: true,
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"add capacity and attribute to all slices": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"remove attribute": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
										"test.example.com/removeMe": {BoolValue: ptr.To(true)},
										"removeMeToo":               {BoolValue: ptr.To(true)},
										"test.example.com/keepMe":   {BoolValue: ptr.To(true)},
										"keepMeToo":                 {BoolValue: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
							},
						},
					},
				},
			},
			expected: []*resourceapi.ResourceSlice{
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
										"test.example.com/keepMe": {BoolValue: ptr.To(true)},
										"keepMeToo":               {BoolValue: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
		},
		"add attribute for driver": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"add attribute for pool": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"add attribute for device": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"add attribute for selector": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"no match when any selector does not match": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"runtime CEL errors skip devices": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"invalid CEL expression returns error": {
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
											Expression: `invalid`,
										},
									},
								},
							},
						},
					},
				},
			},
			matchErr: gomega.MatchError(gomega.ContainSubstring("CEL compile error")),
		},
		"add attribute for device class": {
			deviceClasses: []*resourceapi.DeviceClass{
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
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
		"filter on all criteria": {
			deviceClasses: []*resourceapi.DeviceClass{
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
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
			resourceSlices: []*resourceapi.ResourceSlice{
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
			resourceSlicePatches: []*resourcealphaapi.ResourceSlicePatch{
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
			expected: []*resourceapi.ResourceSlice{
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
			g := gomega.NewWithT(t)

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminControlledDeviceAttributes, !test.adminAttrsDisabled)

			objCount := len(test.resourceSlices) + len(test.deviceClasses)
			objs := make([]runtime.Object, 0, objCount)
			for _, resourceSlice := range test.resourceSlices {
				objs = append(objs, resourceSlice)
			}
			for _, deviceClass := range test.deviceClasses {
				objs = append(objs, deviceClass)
			}

			// Passing ResourceSlicePatches directly through here
			// doesn't work because that ultimately results in an
			// incorrect guess at the resource name based on the kind
			// (adding "s" instead of "es"). The same happens even for
			// the Create workaround with the managedFields-tracking
			// client from NewClientset().
			clientset := fake.NewSimpleClientset(objs...)
			for _, resourceSlicePatch := range test.resourceSlicePatches {
				_, err := clientset.ResourceV1alpha3().ResourceSlicePatches().Create(ctx, resourceSlicePatch, metav1.CreateOptions{})
				g.Expect(err).NotTo(gomega.HaveOccurred())
			}

			informerFactory := informers.NewSharedInformerFactoryWithOptions(clientset, 10*time.Minute)
			tracker := NewTracker(informerFactory)

			informerStop := make(chan struct{})
			informerFactory.Start(informerStop)
			var unsynced []reflect.Type
			for typ, isSynced := range informerFactory.WaitForCacheSync(informerStop) {
				if !isSynced {
					unsynced = append(unsynced, typ)
				}
			}
			g.Expect(unsynced).To(gomega.BeEmpty())
			t.Cleanup(func() {
				close(informerStop)
				informerFactory.Shutdown()
			})

			patched, err := tracker.ListPatchedResourceSlices(ctx)
			matchErr := test.matchErr
			if matchErr == nil {
				matchErr = gomega.Not(gomega.HaveOccurred())
			}
			g.Expect(err).To(matchErr)
			g.Expect(patched).To(gomega.ConsistOf(test.expected))
		})
	}
}
