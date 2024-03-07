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
	"context"
	"errors"
	"fmt"
	"slices"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/dynamic-resource-allocation/structured/namedresources/cel"
)

// These types and fields are all exported to allow logging them with
// pretty-printed JSON.

type Model struct {
	Instances []InstanceAllocation
}

type InstanceAllocation struct {
	Allocated bool
	Instance  *resourceapi.NamedResourcesInstance
}

// AddResources must be called first to create entries for all existing
// resource instances. The resources parameter may be nil.
func AddResources(m *Model, resources *resourceapi.NamedResourcesResources) {
	if resources == nil {
		return
	}

	for i := range resources.Instances {
		m.Instances = append(m.Instances, InstanceAllocation{Instance: &resources.Instances[i]})
	}
}

// AddAllocation may get called after AddResources to mark some resource
// instances as allocated. The result parameter may be nil.
func AddAllocation(m *Model, result *resourceapi.NamedResourcesAllocationResult) {
	if result == nil {
		return
	}
	for i := range m.Instances {
		if m.Instances[i].Instance.Name == result.Name {
			m.Instances[i].Allocated = true
			break
		}
	}
}

func NewClaimController(filter *resourceapi.NamedResourcesFilter, requests []*resourceapi.NamedResourcesRequest) (*Controller, error) {
	c := &Controller{}
	if filter != nil {
		compilation := cel.Compiler.CompileCELExpression(filter.Selector, environment.StoredExpressions)
		if compilation.Error != nil {
			// Shouldn't happen because of validation.
			return nil, fmt.Errorf("compile class filter CEL expression: %w", compilation.Error)
		}
		c.filter = &compilation
	}
	for _, request := range requests {
		compilation := cel.Compiler.CompileCELExpression(request.Selector, environment.StoredExpressions)
		if compilation.Error != nil {
			// Shouldn't happen because of validation.
			return nil, fmt.Errorf("compile request CEL expression: %w", compilation.Error)
		}
		c.requests = append(c.requests, compilation)
	}
	return c, nil
}

type Controller struct {
	filter   *cel.CompilationResult
	requests []cel.CompilationResult
}

func (c *Controller) NodeIsSuitable(ctx context.Context, model Model) (bool, error) {
	indices, err := c.allocate(ctx, model)
	return len(indices) == len(c.requests), err
}

func (c *Controller) Allocate(ctx context.Context, model Model) ([]*resourceapi.NamedResourcesAllocationResult, error) {
	indices, err := c.allocate(ctx, model)
	if err != nil {
		return nil, err
	}
	if len(indices) != len(c.requests) {
		return nil, errors.New("insufficient resources")
	}
	results := make([]*resourceapi.NamedResourcesAllocationResult, len(c.requests))
	for i := range c.requests {
		results[i] = &resourceapi.NamedResourcesAllocationResult{Name: model.Instances[indices[i]].Instance.Name}
	}
	return results, nil
}

func (c *Controller) allocate(ctx context.Context, model Model) ([]int, error) {
	// Shallow copy, we need to modify the allocated boolean.
	instances := slices.Clone(model.Instances)
	indices := make([]int, 0, len(c.requests))

	for _, request := range c.requests {
		for i, instance := range instances {
			if instance.Allocated {
				continue
			}
			if c.filter != nil {
				okay, err := c.filter.Evaluate(ctx, instance.Instance.Attributes)
				if err != nil {
					return nil, fmt.Errorf("evaluate filter CEL expression: %w", err)
				}
				if !okay {
					continue
				}
			}
			okay, err := request.Evaluate(ctx, instance.Instance.Attributes)
			if err != nil {
				return nil, fmt.Errorf("evaluate request CEL expression: %w", err)
			}
			if !okay {
				continue
			}
			// Found a matching, unallocated instance. Let's use it.
			//
			// A more thorough search would include backtracking because
			// allocating one "large" instances for a "small" request may
			// make a following "large" request impossible to satisfy when
			// only "small" instances are left.
			instances[i].Allocated = true
			indices = append(indices, i)
			break
		}
	}
	return indices, nil

}
