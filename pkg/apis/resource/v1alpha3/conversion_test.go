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

package v1alpha3

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	k8sresource "k8s.io/apimachinery/pkg/api/resource"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/resource"
)

func TestConversion(t *testing.T) {
	testcases := []struct {
		name      string
		in        runtime.Object
		out       runtime.Object
		expectOut runtime.Object
		expectErr string
	}{
		{
			name: "ResourceClaim: v1alpha3 to internal without alternatives",
			in: &resourcev1alpha3.ResourceClaim{
				Spec: resourcev1alpha3.ResourceClaimSpec{
					Devices: resourcev1alpha3.DeviceClaim{
						Requests: []resourcev1alpha3.DeviceRequest{
							{
								Name:            "foo",
								DeviceClassName: "class-a",
								Selectors: []resourcev1alpha3.DeviceSelector{
									{
										CEL: &resourcev1alpha3.CELDeviceSelector{
											Expression: `device.attributes["driver-a"].exists`,
										},
									},
								},
								AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
								Count:          2,
							},
						},
					},
				},
			},
			out: &resource.ResourceClaim{},
			expectOut: &resource.ResourceClaim{
				Spec: resource.ResourceClaimSpec{
					Devices: resource.DeviceClaim{
						Requests: []resource.DeviceRequest{
							{
								Name: "foo",
								Exactly: &resource.SpecificDeviceRequest{
									DeviceClassName: "class-a",
									Selectors: []resource.DeviceSelector{
										{
											CEL: &resource.CELDeviceSelector{
												Expression: `device.attributes["driver-a"].exists`,
											},
										},
									},
									AllocationMode: resource.DeviceAllocationModeExactCount,
									Count:          2,
								},
							},
						},
					},
				},
			},
		},
		{
			name: "ResourceClaim: internal to v1alpha3 without alternatives",
			in: &resource.ResourceClaim{
				Spec: resource.ResourceClaimSpec{
					Devices: resource.DeviceClaim{
						Requests: []resource.DeviceRequest{
							{
								Name: "foo",
								Exactly: &resource.SpecificDeviceRequest{
									DeviceClassName: "class-a",
									Selectors: []resource.DeviceSelector{
										{
											CEL: &resource.CELDeviceSelector{
												Expression: `device.attributes["driver-a"].exists`,
											},
										},
									},
									AllocationMode: resource.DeviceAllocationModeExactCount,
									Count:          2,
								},
							},
						},
					},
				},
			},
			out: &resourcev1alpha3.ResourceClaim{},
			expectOut: &resourcev1alpha3.ResourceClaim{
				Spec: resourcev1alpha3.ResourceClaimSpec{
					Devices: resourcev1alpha3.DeviceClaim{
						Requests: []resourcev1alpha3.DeviceRequest{
							{
								Name:            "foo",
								DeviceClassName: "class-a",
								Selectors: []resourcev1alpha3.DeviceSelector{
									{
										CEL: &resourcev1alpha3.CELDeviceSelector{
											Expression: `device.attributes["driver-a"].exists`,
										},
									},
								},
								AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
								Count:          2,
							},
						},
					},
				},
			},
		},

		{
			name: "ResourceClaim: v1alpha3 to internal with alternatives",
			in: &resourcev1alpha3.ResourceClaim{
				Spec: resourcev1alpha3.ResourceClaimSpec{
					Devices: resourcev1alpha3.DeviceClaim{
						Requests: []resourcev1alpha3.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resourcev1alpha3.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resourcev1alpha3.DeviceSelector{
											{
												CEL: &resourcev1alpha3.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resourcev1alpha3.DeviceSelector{
											{
												CEL: &resourcev1alpha3.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
										Count:          1,
									},
								},
							},
						},
					},
				},
			},
			out: &resource.ResourceClaim{},
			expectOut: &resource.ResourceClaim{
				Spec: resource.ResourceClaimSpec{
					Devices: resource.DeviceClaim{
						Requests: []resource.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resource.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resource.DeviceSelector{
											{
												CEL: &resource.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resource.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resource.DeviceSelector{
											{
												CEL: &resource.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resource.DeviceAllocationModeExactCount,
										Count:          1,
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "ResourceClaim: internal to v1alpha3 with alternatives",
			in: &resource.ResourceClaim{
				Spec: resource.ResourceClaimSpec{
					Devices: resource.DeviceClaim{
						Requests: []resource.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resource.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resource.DeviceSelector{
											{
												CEL: &resource.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resource.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resource.DeviceSelector{
											{
												CEL: &resource.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resource.DeviceAllocationModeExactCount,
										Count:          1,
									},
								},
							},
						},
					},
				},
			},
			out: &resourcev1alpha3.ResourceClaim{},
			expectOut: &resourcev1alpha3.ResourceClaim{
				Spec: resourcev1alpha3.ResourceClaimSpec{
					Devices: resourcev1alpha3.DeviceClaim{
						Requests: []resourcev1alpha3.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resourcev1alpha3.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resourcev1alpha3.DeviceSelector{
											{
												CEL: &resourcev1alpha3.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resourcev1alpha3.DeviceSelector{
											{
												CEL: &resourcev1alpha3.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1alpha3.DeviceAllocationModeExactCount,
										Count:          1,
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "ResourceSlice: v1alpha3 to internal",
			in: &resourcev1alpha3.ResourceSlice{
				Spec: resourcev1alpha3.ResourceSliceSpec{
					Driver: "",
					Pool: resourcev1alpha3.ResourcePool{
						Name:               "pool-1",
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: "node-1",
					CapacityPools: []resourcev1alpha3.CapacityPool{
						{
							Name: "capacity-pool",
							Includes: []resourcev1alpha3.CapacityPoolMixinRef{
								{
									Name: "capacity-pool-mixin",
								},
							},
						},
					},
					Mixins: &resourcev1alpha3.ResourceSliceMixins{
						CapacityPool: []resourcev1alpha3.CapacityPoolMixin{
							{
								Name: "capacity-pool-mixin",
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("memory"): k8sresource.MustParse("40Gi"),
								},
							},
						},
						Device: []resourcev1alpha3.DeviceMixin{
							{
								Name: "device-mixin",
								Composite: &resourcev1alpha3.CompositeDeviceMixin{
									Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
										resourcev1alpha3.QualifiedName("attr-1"): {
											IntValue: func() *int64 {
												val := int64(42)
												return &val
											}(),
										},
									},
								},
							},
						},
					},
					Devices: []resourcev1alpha3.Device{
						{
							Name: "device-1",
							Basic: &resourcev1alpha3.BasicDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
							},
						},
						{
							Name: "device-2",
							Composite: &resourcev1alpha3.CompositeDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
								Includes: []resourcev1alpha3.DeviceMixinRef{
									{
										Name: "device-mixin",
									},
								},
								ConsumesCapacity: []resourcev1alpha3.DeviceCapacityConsumption{
									{
										CapacityPool: "capacity-pool",
										Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
											resourcev1alpha3.QualifiedName("memory"): k8sresource.MustParse("20Gi"),
										},
									},
								},
							},
						},
						{
							Name: "device-3",
							Composite: &resourcev1alpha3.CompositeDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
							},
						},
					},
				},
			},
			out: &resource.ResourceSlice{},
			expectOut: &resource.ResourceSlice{
				ObjectMeta: v1.ObjectMeta{
					Annotations: map[string]string{
						"resource.k8s.io/basic-device-names": "device-1",
					},
				},
				Spec: resource.ResourceSliceSpec{
					Driver: "",
					Pool: resource.ResourcePool{
						Name:               "pool-1",
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: "node-1",
					CapacityPools: []resource.CapacityPool{
						{
							Name: "capacity-pool",
							Includes: []resource.CapacityPoolMixinRef{
								{
									Name: "capacity-pool-mixin",
								},
							},
						},
					},
					Mixins: &resource.ResourceSliceMixins{
						CapacityPool: []resource.CapacityPoolMixin{
							{
								Name: "capacity-pool-mixin",
								Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
									resource.QualifiedName("memory"): {
										Value: k8sresource.MustParse("40Gi"),
									},
								},
							},
						},
						Device: []resource.DeviceMixin{
							{
								Name: "device-mixin",
								Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
									resource.QualifiedName("attr-1"): {
										IntValue: func() *int64 {
											val := int64(42)
											return &val
										}(),
									},
								},
							},
						},
					},
					Devices: []resource.Device{
						{
							Name: "device-1",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
						},
						{
							Name: "device-2",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
							Includes: []resource.DeviceMixinRef{
								{
									Name: "device-mixin",
								},
							},
							ConsumesCapacity: []resource.DeviceCapacityConsumption{
								{
									CapacityPool: "capacity-pool",
									Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
										resource.QualifiedName("memory"): {
											Value: k8sresource.MustParse("20Gi"),
										},
									},
								},
							},
						},
						{
							Name: "device-3",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
						},
					},
				},
			},
		},

		{
			name: "ResourceSlice: internal to v1beta",
			in: &resource.ResourceSlice{
				ObjectMeta: v1.ObjectMeta{
					Annotations: map[string]string{
						"resource.k8s.io/basic-device-names": "device-1",
					},
				},
				Spec: resource.ResourceSliceSpec{
					Driver: "",
					Pool: resource.ResourcePool{
						Name:               "pool-1",
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: "node-1",
					CapacityPools: []resource.CapacityPool{
						{
							Name: "capacity-pool",
							Includes: []resource.CapacityPoolMixinRef{
								{
									Name: "capacity-pool-mixin",
								},
							},
						},
					},
					Mixins: &resource.ResourceSliceMixins{
						CapacityPool: []resource.CapacityPoolMixin{
							{
								Name: "capacity-pool-mixin",
								Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
									resource.QualifiedName("memory"): {
										Value: k8sresource.MustParse("40Gi"),
									},
								},
							},
						},
						Device: []resource.DeviceMixin{
							{
								Name: "device-mixin",
								Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
									resource.QualifiedName("attr-1"): {
										IntValue: func() *int64 {
											val := int64(42)
											return &val
										}(),
									},
								},
							},
						},
					},
					Devices: []resource.Device{
						{
							Name: "device-1",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
						},
						{
							Name: "device-2",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
							Includes: []resource.DeviceMixinRef{
								{
									Name: "device-mixin",
								},
							},
							ConsumesCapacity: []resource.DeviceCapacityConsumption{
								{
									CapacityPool: "capacity-pool",
									Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
										resource.QualifiedName("memory"): {
											Value: k8sresource.MustParse("20Gi"),
										},
									},
								},
							},
						},
						{
							Name: "device-3",
							Attributes: map[resource.QualifiedName]resource.DeviceAttribute{
								resource.QualifiedName("attr-2"): {
									StringValue: func() *string {
										val := "foo"
										return &val
									}(),
								},
							},
							Capacity: map[resource.QualifiedName]resource.DeviceCapacity{
								resource.QualifiedName("cpus"): {
									Value: k8sresource.MustParse("42"),
								},
							},
						},
					},
				},
			},
			out: &resourcev1alpha3.ResourceSlice{},
			expectOut: &resourcev1alpha3.ResourceSlice{
				Spec: resourcev1alpha3.ResourceSliceSpec{
					Driver: "",
					Pool: resourcev1alpha3.ResourcePool{
						Name:               "pool-1",
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: "node-1",
					CapacityPools: []resourcev1alpha3.CapacityPool{
						{
							Name: "capacity-pool",
							Includes: []resourcev1alpha3.CapacityPoolMixinRef{
								{
									Name: "capacity-pool-mixin",
								},
							},
						},
					},
					Mixins: &resourcev1alpha3.ResourceSliceMixins{
						CapacityPool: []resourcev1alpha3.CapacityPoolMixin{
							{
								Name: "capacity-pool-mixin",
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("memory"): k8sresource.MustParse("40Gi"),
								},
							},
						},
						Device: []resourcev1alpha3.DeviceMixin{
							{
								Name: "device-mixin",
								Composite: &resourcev1alpha3.CompositeDeviceMixin{
									Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
										resourcev1alpha3.QualifiedName("attr-1"): {
											IntValue: func() *int64 {
												val := int64(42)
												return &val
											}(),
										},
									},
								},
							},
						},
					},
					Devices: []resourcev1alpha3.Device{
						{
							Name: "device-1",
							Basic: &resourcev1alpha3.BasicDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
							},
						},
						{
							Name: "device-2",
							Composite: &resourcev1alpha3.CompositeDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
								Includes: []resourcev1alpha3.DeviceMixinRef{
									{
										Name: "device-mixin",
									},
								},
								ConsumesCapacity: []resourcev1alpha3.DeviceCapacityConsumption{
									{
										CapacityPool: "capacity-pool",
										Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
											resourcev1alpha3.QualifiedName("memory"): k8sresource.MustParse("20Gi"),
										},
									},
								},
							},
						},
						{
							Name: "device-3",
							Composite: &resourcev1alpha3.CompositeDevice{
								Attributes: map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute{
									resourcev1alpha3.QualifiedName("attr-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1alpha3.QualifiedName]k8sresource.Quantity{
									resourcev1alpha3.QualifiedName("cpus"): k8sresource.MustParse("42"),
								},
							},
						},
					},
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	if err := resource.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	for i := range testcases {
		name := testcases[i].name
		tc := testcases[i]
		t.Run(name, func(t *testing.T) {
			err := scheme.Convert(tc.in, tc.out, nil)
			if err != nil {
				if len(tc.expectErr) == 0 {
					t.Fatalf("unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tc.expectErr) {
					t.Fatalf("expected error %s, got %v", tc.expectErr, err)
				}
				return
			}
			if len(tc.expectErr) > 0 {
				t.Fatalf("expected error %s, got none", tc.expectErr)
			}
			if !reflect.DeepEqual(tc.out, tc.expectOut) {
				t.Fatalf("unexpected result:\n %s", cmp.Diff(tc.expectOut, tc.out))
			}
		})
	}

}
