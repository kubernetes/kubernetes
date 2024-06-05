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

package namedresources

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func instance(allocated bool, name string, attributes ...resourceapi.NamedResourcesAttribute) InstanceAllocation {
	return InstanceAllocation{
		Allocated: allocated,
		Instance: &resourceapi.NamedResourcesInstance{
			Name:       name,
			Attributes: attributes,
		},
	}
}

func TestModel(t *testing.T) {
	testcases := map[string]struct {
		resources   []*resourceapi.NamedResourcesResources
		allocations []*resourceapi.NamedResourcesAllocationResult

		expectModel Model
	}{
		"empty": {},

		"nil": {
			resources:   []*resourceapi.NamedResourcesResources{nil},
			allocations: []*resourceapi.NamedResourcesAllocationResult{nil},
		},

		"available": {
			resources: []*resourceapi.NamedResourcesResources{
				{
					Instances: []resourceapi.NamedResourcesInstance{
						{Name: "a"},
						{Name: "b"},
					},
				},
				{
					Instances: []resourceapi.NamedResourcesInstance{
						{Name: "x"},
						{Name: "y"},
					},
				},
			},

			expectModel: Model{Instances: []InstanceAllocation{instance(false, "a"), instance(false, "b"), instance(false, "x"), instance(false, "y")}},
		},

		"allocated": {
			resources: []*resourceapi.NamedResourcesResources{
				{
					Instances: []resourceapi.NamedResourcesInstance{
						{Name: "a"},
						{Name: "b"},
					},
				},
				{
					Instances: []resourceapi.NamedResourcesInstance{
						{Name: "x"},
						{Name: "y"},
					},
				},
			},
			allocations: []*resourceapi.NamedResourcesAllocationResult{
				{
					Name: "something-else",
				},
				{
					Name: "a",
				},
			},

			expectModel: Model{Instances: []InstanceAllocation{instance(true, "a"), instance(false, "b"), instance(false, "x"), instance(false, "y")}},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			var actualModel Model
			for _, resources := range tc.resources {
				AddResources(&actualModel, resources)
			}
			for _, allocation := range tc.allocations {
				AddAllocation(&actualModel, allocation)
			}

			require.Equal(t, tc.expectModel, actualModel)
		})
	}

}

func TestController(t *testing.T) {
	filterAny := &resourceapi.NamedResourcesFilter{
		Selector: "true",
	}
	filterNone := &resourceapi.NamedResourcesFilter{
		Selector: "false",
	}
	filterBrokenType := &resourceapi.NamedResourcesFilter{
		Selector: "1",
	}
	filterBrokenEvaluation := &resourceapi.NamedResourcesFilter{
		Selector: `attributes.bool["no-such-attribute"]`,
	}
	filterAttribute := &resourceapi.NamedResourcesFilter{
		Selector: `attributes.bool["usable"]`,
	}

	requestAny := &resourceapi.NamedResourcesRequest{
		Selector: "true",
	}
	requestNone := &resourceapi.NamedResourcesRequest{
		Selector: "false",
	}
	requestBrokenType := &resourceapi.NamedResourcesRequest{
		Selector: "1",
	}
	requestBrokenEvaluation := &resourceapi.NamedResourcesRequest{
		Selector: `attributes.bool["no-such-attribute"]`,
	}
	requestAttribute := &resourceapi.NamedResourcesRequest{
		Selector: `attributes.bool["usable"]`,
	}

	instance1 := "instance-1"
	oneInstance := Model{
		Instances: []InstanceAllocation{{
			Instance: &resourceapi.NamedResourcesInstance{
				Name: instance1,
			},
		}},
	}

	instance2 := "instance-2"
	twoInstances := Model{
		Instances: []InstanceAllocation{
			{
				Instance: &resourceapi.NamedResourcesInstance{
					Name: instance1,
					Attributes: []resourceapi.NamedResourcesAttribute{{
						Name: "usable",
						NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{
							BoolValue: ptr.To(false),
						},
					}},
				},
			},
			{
				Instance: &resourceapi.NamedResourcesInstance{
					Name: instance2,
					Attributes: []resourceapi.NamedResourcesAttribute{{
						Name: "usable",
						NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{
							BoolValue: ptr.To(true),
						},
					}},
				},
			},
		},
	}

	testcases := map[string]struct {
		model    Model
		filter   *resourceapi.NamedResourcesFilter
		requests []*resourceapi.NamedResourcesRequest

		expectCreateErr   bool
		expectAllocation  []string
		expectAllocateErr bool
	}{
		"empty": {},

		"broken-filter": {
			filter: filterBrokenType,

			expectCreateErr: true,
		},

		"broken-request": {
			requests: []*resourceapi.NamedResourcesRequest{requestBrokenType},

			expectCreateErr: true,
		},

		"no-resources": {
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestAny},

			expectAllocateErr: true,
		},

		"okay": {
			model:    oneInstance,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestAny},

			expectAllocation: []string{instance1},
		},

		"filter-mismatch": {
			model:    oneInstance,
			filter:   filterNone,
			requests: []*resourceapi.NamedResourcesRequest{requestAny},

			expectAllocateErr: true,
		},

		"request-mismatch": {
			model:    oneInstance,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestNone},

			expectAllocateErr: true,
		},

		"many": {
			model:    twoInstances,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestAny, requestAny},

			expectAllocation: []string{instance1, instance2},
		},

		"too-many": {
			model:    oneInstance,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestAny, requestAny},

			expectAllocateErr: true,
		},

		"filter-evaluation-error": {
			model:    oneInstance,
			filter:   filterBrokenEvaluation,
			requests: []*resourceapi.NamedResourcesRequest{requestAny},

			expectAllocateErr: true,
		},

		"request-evaluation-error": {
			model:    oneInstance,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestBrokenEvaluation},

			expectAllocateErr: true,
		},

		"filter-attribute": {
			model:    twoInstances,
			filter:   filterAttribute,
			requests: []*resourceapi.NamedResourcesRequest{requestAny},

			expectAllocation: []string{instance2},
		},

		"request-attribute": {
			model:    twoInstances,
			filter:   filterAny,
			requests: []*resourceapi.NamedResourcesRequest{requestAttribute},

			expectAllocation: []string{instance2},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			controller, createErr := NewClaimController(tc.filter, tc.requests)
			if createErr != nil {
				if !tc.expectCreateErr {
					tCtx.Fatalf("unexpected create error: %v", createErr)
				}
				return
			}
			if tc.expectCreateErr {
				tCtx.Fatalf("did not get expected create error")
			}

			allocation, createErr := controller.Allocate(tCtx, tc.model)
			if createErr != nil {
				if !tc.expectAllocateErr {
					tCtx.Fatalf("unexpected allocate error: %v", createErr)
				}
				return
			}
			if tc.expectAllocateErr {
				tCtx.Fatalf("did not get expected allocate error")
			}

			expectAllocation := []*resourceapi.NamedResourcesAllocationResult{}
			for _, name := range tc.expectAllocation {
				expectAllocation = append(expectAllocation, &resourceapi.NamedResourcesAllocationResult{Name: name})
			}
			require.Equal(tCtx, expectAllocation, allocation)

			isSuitable, isSuitableErr := controller.NodeIsSuitable(tCtx, tc.model)
			assert.Equal(tCtx, len(expectAllocation) == len(tc.requests), isSuitable, "is suitable")
			assert.Equal(tCtx, createErr, isSuitableErr)
		})
	}
}
