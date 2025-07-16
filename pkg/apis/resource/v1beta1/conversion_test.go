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

package v1beta1

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
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
			name: "v1beta1 to internal without alternatives",
			in: &resourcev1beta1.ResourceClaim{
				Spec: resourcev1beta1.ResourceClaimSpec{
					Devices: resourcev1beta1.DeviceClaim{
						Requests: []resourcev1beta1.DeviceRequest{
							{
								Name:            "foo",
								DeviceClassName: "class-a",
								Selectors: []resourcev1beta1.DeviceSelector{
									{
										CEL: &resourcev1beta1.CELDeviceSelector{
											Expression: `device.attributes["driver-a"].exists`,
										},
									},
								},
								AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
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
								Exactly: &resource.ExactDeviceRequest{
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
			name: "internal to v1beta1 without alternatives",
			in: &resource.ResourceClaim{
				Spec: resource.ResourceClaimSpec{
					Devices: resource.DeviceClaim{
						Requests: []resource.DeviceRequest{
							{
								Name: "foo",
								Exactly: &resource.ExactDeviceRequest{
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
			out: &resourcev1beta1.ResourceClaim{},
			expectOut: &resourcev1beta1.ResourceClaim{
				Spec: resourcev1beta1.ResourceClaimSpec{
					Devices: resourcev1beta1.DeviceClaim{
						Requests: []resourcev1beta1.DeviceRequest{
							{
								Name:            "foo",
								DeviceClassName: "class-a",
								Selectors: []resourcev1beta1.DeviceSelector{
									{
										CEL: &resourcev1beta1.CELDeviceSelector{
											Expression: `device.attributes["driver-a"].exists`,
										},
									},
								},
								AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
								Count:          2,
							},
						},
					},
				},
			},
		},

		{
			name: "v1beta1 to internal with alternatives",
			in: &resourcev1beta1.ResourceClaim{
				Spec: resourcev1beta1.ResourceClaimSpec{
					Devices: resourcev1beta1.DeviceClaim{
						Requests: []resourcev1beta1.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resourcev1beta1.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resourcev1beta1.DeviceSelector{
											{
												CEL: &resourcev1beta1.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resourcev1beta1.DeviceSelector{
											{
												CEL: &resourcev1beta1.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
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
			name: "v1beta1 to internal with alternatives",
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
			out: &resourcev1beta1.ResourceClaim{},
			expectOut: &resourcev1beta1.ResourceClaim{
				Spec: resourcev1beta1.ResourceClaimSpec{
					Devices: resourcev1beta1.DeviceClaim{
						Requests: []resourcev1beta1.DeviceRequest{
							{
								Name: "foo",
								FirstAvailable: []resourcev1beta1.DeviceSubRequest{
									{
										Name:            "sub-1",
										DeviceClassName: "class-a",
										Selectors: []resourcev1beta1.DeviceSelector{
											{
												CEL: &resourcev1beta1.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
										Count:          2,
									},
									{
										Name:            "sub-2",
										DeviceClassName: "class-a",
										Selectors: []resourcev1beta1.DeviceSelector{
											{
												CEL: &resourcev1beta1.CELDeviceSelector{
													Expression: `device.attributes["driver-a"].exists`,
												},
											},
										},
										AllocationMode: resourcev1beta1.DeviceAllocationModeExactCount,
										Count:          1,
									},
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
