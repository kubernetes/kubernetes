/*
Copyright 2024 The Kubernetes Authors.

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

package dynamicresources

import (
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	namedresourcesmodel "k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/structured/namedresources"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestModel(t *testing.T) {
	testcases := map[string]struct {
		slices   resourceSliceLister
		claims   assumeCacheLister
		inFlight map[types.UID]resourceapi.ResourceClaimStatus

		wantResources resources
		wantErr       bool
	}{
		"empty": {},

		"slice-list-error": {
			slices: sliceError("slice list error"),

			wantErr: true,
		},

		"unknown-model": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:      "node",
					DriverName:    "driver",
					ResourceModel: resourceapi.ResourceModel{ /* empty! */ },
				},
			},

			// Not an error. It is safe to ignore unknown resources until a claim requests them.
			// The unknown model in that claim then triggers an error for that claim.
			wantResources: resources{"node": map[string]ResourceModels{
				"driver": {
					NamedResources: namedresourcesmodel.Model{},
				},
			}},
		},

		"one-instance": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node",
					DriverName: "driver",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{
								Name: "one",
								Attributes: []resourceapi.NamedResourcesAttribute{{
									Name: "size",
									NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{
										IntValue: ptr.To(int64(1)),
									},
								}},
							}},
						},
					},
				},
			},

			wantResources: resources{"node": map[string]ResourceModels{
				"driver": {
					NamedResources: namedresourcesmodel.Model{
						Instances: []namedresourcesmodel.InstanceAllocation{
							{
								Instance: &resourceapi.NamedResourcesInstance{
									Name: "one",
									Attributes: []resourceapi.NamedResourcesAttribute{{
										Name: "size",
										NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{
											IntValue: ptr.To(int64(1)),
										},
									}},
								},
							},
						},
					},
				},
			}},
		},

		"two-instances": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node",
					DriverName: "driver",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
			},

			wantResources: resources{"node": map[string]ResourceModels{
				"driver": {
					NamedResources: namedresourcesmodel.Model{
						Instances: []namedresourcesmodel.InstanceAllocation{
							{
								Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
							},
							{
								Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
							},
						},
					},
				},
			}},
		},

		"two-nodes": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"two-nodes-two-drivers": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"in-use-simple": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			claims: claimList{
				&resourceapi.ResourceClaim{
					/* not allocated */
				},
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "driver-a",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{{
								// Claims not allocated via structured parameters can be ignored.
							}},
						},
					},
				},
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "driver-a",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{{
								DriverName: "driver-a",
								StructuredData: &resourceapi.StructuredResourceHandle{
									NodeName: "node-b",
									// Unknown allocations can be ignored.
									Results: []resourceapi.DriverAllocationResult{{
										AllocationResultModel: resourceapi.AllocationResultModel{},
									}},
								},
							}},
						},
					},
				},
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "driver-a",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{{
								DriverName: "driver-a",
								StructuredData: &resourceapi.StructuredResourceHandle{
									NodeName: "node-a",
									Results: []resourceapi.DriverAllocationResult{{
										AllocationResultModel: resourceapi.AllocationResultModel{
											NamedResources: &resourceapi.NamedResourcesAllocationResult{
												Name: "two",
											},
										},
									}},
								},
							}},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"in-use-meta-driver": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			claims: claimList{
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "meta-driver",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{{
								DriverName: "driver-b",
								StructuredData: &resourceapi.StructuredResourceHandle{
									NodeName: "node-b",
									Results: []resourceapi.DriverAllocationResult{{
										AllocationResultModel: resourceapi.AllocationResultModel{
											NamedResources: &resourceapi.NamedResourcesAllocationResult{
												Name: "X",
											},
										},
									}},
								},
							}},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"in-use-many-results": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			claims: claimList{
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "driver-a",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{{
								DriverName: "driver-a",
								StructuredData: &resourceapi.StructuredResourceHandle{
									NodeName: "node-a",
									Results: []resourceapi.DriverAllocationResult{
										{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "one",
												},
											},
										},
										{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "two",
												},
											},
										},
									},
								},
							}},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"in-use-many-handles": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-a",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}, {Name: "two"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-a",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
				&resourceapi.ResourceSlice{
					NodeName:   "node-b",
					DriverName: "driver-b",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "X"}, {Name: "Y"}},
						},
					},
				},
			},

			claims: claimList{
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "meta-driver",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{
								{
									DriverName: "driver-a",
									StructuredData: &resourceapi.StructuredResourceHandle{
										NodeName: "node-b",
										Results: []resourceapi.DriverAllocationResult{{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "X",
												},
											},
										}},
									},
								},
								{
									DriverName: "driver-b",
									StructuredData: &resourceapi.StructuredResourceHandle{
										NodeName: "node-b",
										Results: []resourceapi.DriverAllocationResult{{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "X",
												},
											},
										}},
									},
								},
							},
						},
					},
				},
			},

			wantResources: resources{
				"node-a": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "one"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "two"},
								},
							},
						},
					},
				},
				"node-b": map[string]ResourceModels{
					"driver-a": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
					"driver-b": {
						NamedResources: namedresourcesmodel.Model{
							Instances: []namedresourcesmodel.InstanceAllocation{
								{
									Allocated: true,
									Instance:  &resourceapi.NamedResourcesInstance{Name: "X"},
								},
								{
									Instance: &resourceapi.NamedResourcesInstance{Name: "Y"},
								},
							},
						},
					},
				},
			},
		},

		"orphaned-allocations": {
			claims: claimList{
				&resourceapi.ResourceClaim{
					Status: resourceapi.ResourceClaimStatus{
						DriverName: "meta-driver",
						Allocation: &resourceapi.AllocationResult{
							ResourceHandles: []resourceapi.ResourceHandle{
								{
									DriverName: "driver-a",
									StructuredData: &resourceapi.StructuredResourceHandle{
										NodeName: "node-b",
										Results: []resourceapi.DriverAllocationResult{{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "X",
												},
											},
										}},
									},
								},
								{
									DriverName: "driver-b",
									StructuredData: &resourceapi.StructuredResourceHandle{
										NodeName: "node-b",
										Results: []resourceapi.DriverAllocationResult{{
											AllocationResultModel: resourceapi.AllocationResultModel{
												NamedResources: &resourceapi.NamedResourcesAllocationResult{
													Name: "X",
												},
											},
										}},
									},
								},
							},
						},
					},
				},
			},

			wantResources: resources{
				"node-b": map[string]ResourceModels{},
			},
		},

		"in-flight": {
			slices: sliceList{
				&resourceapi.ResourceSlice{
					NodeName:   "node",
					DriverName: "driver",
					ResourceModel: resourceapi.ResourceModel{
						NamedResources: &resourceapi.NamedResourcesResources{
							Instances: []resourceapi.NamedResourcesInstance{{Name: "one"}},
						},
					},
				},
			},

			claims: claimList{
				&resourceapi.ResourceClaim{
					ObjectMeta: metav1.ObjectMeta{
						UID: "abc",
					},
					// Allocation not recorded yet.
				},
			},

			inFlight: map[types.UID]resourceapi.ResourceClaimStatus{
				"abc": {
					DriverName: "driver",
					Allocation: &resourceapi.AllocationResult{
						ResourceHandles: []resourceapi.ResourceHandle{{
							DriverName: "driver",
							StructuredData: &resourceapi.StructuredResourceHandle{
								NodeName: "node",
								Results: []resourceapi.DriverAllocationResult{{
									AllocationResultModel: resourceapi.AllocationResultModel{
										NamedResources: &resourceapi.NamedResourcesAllocationResult{
											Name: "one",
										},
									},
								}},
							},
						}},
					},
				},
			},

			wantResources: resources{"node": map[string]ResourceModels{
				"driver": {
					NamedResources: namedresourcesmodel.Model{
						Instances: []namedresourcesmodel.InstanceAllocation{
							{
								Allocated: true,
								Instance:  &resourceapi.NamedResourcesInstance{Name: "one"},
							},
						},
					},
				},
			}},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			var inFlightAllocations sync.Map
			for uid, claimStatus := range tc.inFlight {
				inFlightAllocations.Store(uid, &resourceapi.ResourceClaim{Status: claimStatus})
			}

			slices := tc.slices
			if slices == nil {
				slices = sliceList{}
			}
			claims := tc.claims
			if claims == nil {
				claims = claimList{}
			}
			actualResources, actualErr := newResourceModel(tCtx.Logger(), slices, claims, &inFlightAllocations)

			if actualErr != nil {
				if !tc.wantErr {
					tCtx.Fatalf("unexpected error: %v", actualErr)
				}
				return
			}
			if tc.wantErr {
				tCtx.Fatalf("did not get expected error")
			}

			expectResources := tc.wantResources
			if expectResources == nil {
				expectResources = resources{}
			}
			require.Equal(tCtx, expectResources, actualResources)
		})
	}
}

type sliceList []*resourceapi.ResourceSlice

func (l sliceList) List(selector labels.Selector) ([]*resourceapi.ResourceSlice, error) {
	return l, nil
}

type sliceError string

func (l sliceError) List(selector labels.Selector) ([]*resourceapi.ResourceSlice, error) {
	return nil, errors.New(string(l))
}

type claimList []any

func (l claimList) List(indexObj any) []any {
	return l
}

func TestController(t *testing.T) {
	driver1 := "driver-1"
	class1 := &resourceapi.ResourceClass{
		DriverName: driver1,
	}

	classParametersEmpty := &resourceapi.ResourceClassParameters{}
	classParametersAny := &resourceapi.ResourceClassParameters{
		Filters: []resourceapi.ResourceFilter{{
			DriverName: driver1,
			ResourceFilterModel: resourceapi.ResourceFilterModel{
				NamedResources: &resourceapi.NamedResourcesFilter{
					Selector: "true",
				},
			},
		}},
	}

	claimParametersEmpty := &resourceapi.ResourceClaimParameters{}
	claimParametersAny := &resourceapi.ResourceClaimParameters{
		DriverRequests: []resourceapi.DriverRequests{{
			DriverName: driver1,
		}},
	}
	claimParametersOne := &resourceapi.ResourceClaimParameters{
		DriverRequests: []resourceapi.DriverRequests{{
			DriverName: driver1,
			Requests: []resourceapi.ResourceRequest{{
				ResourceRequestModel: resourceapi.ResourceRequestModel{
					NamedResources: &resourceapi.NamedResourcesRequest{
						Selector: "true",
					},
				},
			}},
		}},
	}
	claimParametersBroken := &resourceapi.ResourceClaimParameters{
		DriverRequests: []resourceapi.DriverRequests{{
			DriverName: driver1,
			Requests: []resourceapi.ResourceRequest{{
				ResourceRequestModel: resourceapi.ResourceRequestModel{
					NamedResources: &resourceapi.NamedResourcesRequest{
						Selector: `attributes.bool["no-such-attribute"]`,
					},
				},
			}},
		}},
	}

	instance1 := "instance-1"

	node1 := "node-1"
	node1Selector := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchExpressions: []v1.NodeSelectorRequirement{{
				Key:      "kubernetes.io/hostname",
				Operator: v1.NodeSelectorOpIn,
				Values:   []string{node1},
			}},
		}},
	}
	node1Resources := resources{node1: map[string]ResourceModels{
		driver1: {
			NamedResources: namedresourcesmodel.Model{
				Instances: []namedresourcesmodel.InstanceAllocation{{
					Instance: &resourceapi.NamedResourcesInstance{
						Name: instance1,
					},
				}},
			},
		},
	}}
	node1Allocation := &resourceapi.AllocationResult{
		AvailableOnNodes: node1Selector,
	}

	instance1Allocation := &resourceapi.AllocationResult{
		AvailableOnNodes: node1Selector,
		ResourceHandles: []resourceapi.ResourceHandle{{
			DriverName: driver1,
			StructuredData: &resourceapi.StructuredResourceHandle{
				NodeName: node1,
				Results: []resourceapi.DriverAllocationResult{{
					AllocationResultModel: resourceapi.AllocationResultModel{
						NamedResources: &resourceapi.NamedResourcesAllocationResult{
							Name: instance1,
						},
					},
				}},
			},
		}},
	}

	type nodeResult struct {
		isSuitable  bool
		suitableErr string

		driverName  string
		allocation  *resourceapi.AllocationResult
		allocateErr string
	}
	type nodeResults map[string]nodeResult

	testcases := map[string]struct {
		resources       resources
		class           *resourceapi.ResourceClass
		classParameters *resourceapi.ResourceClassParameters
		claimParameters *resourceapi.ResourceClaimParameters

		expectCreateErr   bool
		expectNodeResults nodeResults
	}{
		"empty": {
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: claimParametersEmpty,

			expectNodeResults: nodeResults{
				node1: {isSuitable: true, driverName: driver1, allocation: node1Allocation},
			},
		},

		"any": {
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: claimParametersAny,

			expectNodeResults: nodeResults{
				node1: {isSuitable: true, driverName: driver1, allocation: node1Allocation},
			},
		},

		"missing-model": {
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: &resourceapi.ResourceClaimParameters{
				DriverRequests: []resourceapi.DriverRequests{{
					Requests: []resourceapi.ResourceRequest{{ /* empty model */ }},
				}},
			},

			expectCreateErr: true,
		},

		"no-resources": {
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: claimParametersOne,

			expectNodeResults: nodeResults{
				node1: {isSuitable: false, allocateErr: "allocating via named resources structured model: insufficient resources"},
			},
		},

		"have-resources": {
			resources:       node1Resources,
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: claimParametersOne,

			expectNodeResults: nodeResults{
				node1: {isSuitable: true, driverName: driver1, allocation: instance1Allocation},
			},
		},

		"broken-cel": {
			resources:       node1Resources,
			class:           class1,
			classParameters: classParametersEmpty,
			claimParameters: claimParametersBroken,

			expectNodeResults: nodeResults{
				node1: {suitableErr: `checking node "node-1" and resources of driver "driver-1": evaluate request CEL expression: no such key: no-such-attribute`},
			},
		},

		"class-filter": {
			resources:       node1Resources,
			class:           class1,
			classParameters: classParametersAny,
			claimParameters: claimParametersOne,

			expectNodeResults: nodeResults{
				node1: {isSuitable: true, driverName: driver1, allocation: instance1Allocation},
			},
		},

		"vendor-parameters": {
			resources: node1Resources,
			class:     class1,
			classParameters: func() *resourceapi.ResourceClassParameters {
				parameters := classParametersAny.DeepCopy()
				parameters.VendorParameters = []resourceapi.VendorParameters{{
					DriverName: driver1,
					Parameters: runtime.RawExtension{Raw: []byte("class-parameters")},
				}}
				return parameters
			}(),

			claimParameters: func() *resourceapi.ResourceClaimParameters {
				parameters := claimParametersOne.DeepCopy()
				parameters.DriverRequests[0].VendorParameters = runtime.RawExtension{Raw: []byte("claim-parameters")}
				parameters.DriverRequests[0].Requests[0].VendorParameters = runtime.RawExtension{Raw: []byte("request-parameters")}
				return parameters
			}(),

			expectNodeResults: nodeResults{
				node1: {isSuitable: true, driverName: driver1,
					allocation: func() *resourceapi.AllocationResult {
						allocation := instance1Allocation.DeepCopy()
						allocation.ResourceHandles[0].StructuredData.VendorClassParameters = runtime.RawExtension{Raw: []byte("class-parameters")}
						allocation.ResourceHandles[0].StructuredData.VendorClaimParameters = runtime.RawExtension{Raw: []byte("claim-parameters")}
						allocation.ResourceHandles[0].StructuredData.Results[0].VendorRequestParameters = runtime.RawExtension{Raw: []byte("request-parameters")}
						return allocation
					}(),
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			controller, err := newClaimController(tCtx.Logger(), tc.class, tc.classParameters, tc.claimParameters)
			if err != nil {
				if !tc.expectCreateErr {
					tCtx.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if tc.expectCreateErr {
				tCtx.Fatalf("did not get expected error")
			}

			for nodeName, expect := range tc.expectNodeResults {
				t.Run(nodeName, func(t *testing.T) {
					tCtx := ktesting.Init(t)

					isSuitable, err := controller.nodeIsSuitable(tCtx, nodeName, tc.resources)
					if err != nil {
						if expect.suitableErr == "" {
							tCtx.Fatalf("unexpected nodeIsSuitable error: %v", err)
						}
						require.Equal(tCtx, expect.suitableErr, err.Error())
						return
					}
					if expect.suitableErr != "" {
						tCtx.Fatalf("did not get expected nodeIsSuitable error: %v", expect.suitableErr)
					}
					assert.Equal(tCtx, expect.isSuitable, isSuitable, "is suitable")

					driverName, allocation, err := controller.allocate(tCtx, nodeName, tc.resources)
					if err != nil {
						if expect.allocateErr == "" {
							tCtx.Fatalf("unexpected allocate error: %v", err)
						}
						require.Equal(tCtx, expect.allocateErr, err.Error())
						return
					}
					if expect.allocateErr != "" {
						tCtx.Fatalf("did not get expected allocate error: %v", expect.allocateErr)
					}
					assert.Equal(tCtx, expect.driverName, driverName, "driver name")
					assert.Equal(tCtx, expect.allocation, allocation)
				})
			}
		})
	}
}
