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
	"context"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	resourcev1alpha2listers "k8s.io/client-go/listers/resource/v1alpha2"
	"k8s.io/klog/v2"
	namedresourcesmodel "k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/structured/namedresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
)

// resources is a map "node name" -> "driver name" -> available and
// allocated resources per structured parameter model.
type resources map[string]map[string]ResourceModels

// ResourceModels may have more than one entry because it is valid for a driver to
// use more than one structured parameter model.
type ResourceModels struct {
	NamedResources namedresourcesmodel.Model
}

// newResourceModel parses the available information about resources. Objects
// with an unknown structured parameter model silently ignored. An error gets
// logged later when parameters required for a pod depend on such an unknown
// model.
func newResourceModel(logger klog.Logger, resourceSliceLister resourcev1alpha2listers.ResourceSliceLister, claimAssumeCache volumebinding.AssumeCache, inFlightAllocations *sync.Map) (resources, error) {
	model := make(resources)

	slices, err := resourceSliceLister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("list node resource slices: %w", err)
	}
	for _, slice := range slices {
		if model[slice.NodeName] == nil {
			model[slice.NodeName] = make(map[string]ResourceModels)
		}
		resource := model[slice.NodeName][slice.DriverName]
		namedresourcesmodel.AddResources(&resource.NamedResources, slice.NamedResources)
		model[slice.NodeName][slice.DriverName] = resource
	}

	objs := claimAssumeCache.List(nil)
	for _, obj := range objs {
		claim, ok := obj.(*resourcev1alpha2.ResourceClaim)
		if !ok {
			return nil, fmt.Errorf("got unexpected object of type %T from claim assume cache", obj)
		}
		if obj, ok := inFlightAllocations.Load(claim.UID); ok {
			// If the allocation is in-flight, then we have to use the allocation
			// from that claim.
			claim = obj.(*resourcev1alpha2.ResourceClaim)
		}
		if claim.Status.Allocation == nil {
			continue
		}
		for _, handle := range claim.Status.Allocation.ResourceHandles {
			structured := handle.StructuredData
			if structured == nil {
				continue
			}
			if model[structured.NodeName] == nil {
				model[structured.NodeName] = make(map[string]ResourceModels)
			}
			resource := model[structured.NodeName][handle.DriverName]
			for _, result := range structured.Results {
				// Call AddAllocation for each known model. Each call itself needs to check for nil.
				namedresourcesmodel.AddAllocation(&resource.NamedResources, result.NamedResources)
			}
		}
	}

	return model, nil
}

func newClaimController(logger klog.Logger, class *resourcev1alpha2.ResourceClass, classParameters *resourcev1alpha2.ResourceClassParameters, claimParameters *resourcev1alpha2.ResourceClaimParameters) (*claimController, error) {
	// Each node driver is separate from the others. Each driver may have
	// multiple requests which need to be allocated together, so here
	// we have to collect them per model.
	type perDriverRequests struct {
		parameters []runtime.RawExtension
		requests   []*resourcev1alpha2.NamedResourcesRequest
	}
	namedresourcesRequests := make(map[string]perDriverRequests)
	for i, request := range claimParameters.DriverRequests {
		driverName := request.DriverName
		p := namedresourcesRequests[driverName]
		for e, request := range request.Requests {
			switch {
			case request.ResourceRequestModel.NamedResources != nil:
				p.parameters = append(p.parameters, request.VendorParameters)
				p.requests = append(p.requests, request.ResourceRequestModel.NamedResources)
			default:
				return nil, fmt.Errorf("claim parameters %s: driverRequersts[%d].requests[%d]: no supported structured parameters found", klog.KObj(claimParameters), i, e)
			}
		}
		if len(p.requests) > 0 {
			namedresourcesRequests[driverName] = p
		}
	}

	c := &claimController{
		class:           class,
		classParameters: classParameters,
		claimParameters: claimParameters,
		namedresources:  make(map[string]perDriverController, len(namedresourcesRequests)),
	}
	for driverName, perDriver := range namedresourcesRequests {
		var filter *resourcev1alpha2.NamedResourcesFilter
		for _, f := range classParameters.Filters {
			if f.DriverName == driverName && f.ResourceFilterModel.NamedResources != nil {
				filter = f.ResourceFilterModel.NamedResources
				break
			}
		}
		controller, err := namedresourcesmodel.NewClaimController(filter, perDriver.requests)
		if err != nil {
			return nil, fmt.Errorf("creating claim controller for named resources structured model: %w", err)
		}
		c.namedresources[driverName] = perDriverController{
			parameters: perDriver.parameters,
			controller: controller,
		}
	}
	return c, nil
}

// claimController currently wraps exactly one structured parameter model.

type claimController struct {
	class           *resourcev1alpha2.ResourceClass
	classParameters *resourcev1alpha2.ResourceClassParameters
	claimParameters *resourcev1alpha2.ResourceClaimParameters
	namedresources  map[string]perDriverController
}

type perDriverController struct {
	parameters []runtime.RawExtension
	controller *namedresourcesmodel.Controller
}

func (c claimController) nodeIsSuitable(ctx context.Context, nodeName string, resources resources) (bool, error) {
	nodeResources := resources[nodeName]
	for driverName, perDriver := range c.namedresources {
		okay, err := perDriver.controller.NodeIsSuitable(ctx, nodeResources[driverName].NamedResources)
		if err != nil {
			// This is an error in the CEL expression which needs
			// to be fixed. Better fail very visibly instead of
			// ignoring the node.
			return false, fmt.Errorf("checking node %q and resources of driver %q: %w", nodeName, driverName, err)
		}
		if !okay {
			return false, nil
		}
	}
	return true, nil
}

func (c claimController) allocate(ctx context.Context, nodeName string, resources resources) (string, *resourcev1alpha2.AllocationResult, error) {
	allocation := &resourcev1alpha2.AllocationResult{
		Shareable: c.claimParameters.Shareable,
		AvailableOnNodes: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{Key: "kubernetes.io/hostname", Operator: v1.NodeSelectorOpIn, Values: []string{nodeName}},
					},
				},
			},
		},
	}

	nodeResources := resources[nodeName]
	for driverName, perDriver := range c.namedresources {
		// Must return one entry for each request. The entry may be nil. This way,
		// the result can be correlated with the per-request parameters.
		results, err := perDriver.controller.Allocate(ctx, nodeResources[driverName].NamedResources)
		if err != nil {
			return "", nil, fmt.Errorf("allocating via named resources structured model: %w", err)
		}
		handle := resourcev1alpha2.ResourceHandle{
			DriverName: driverName,
			StructuredData: &resourcev1alpha2.StructuredResourceHandle{
				NodeName: nodeName,
			},
		}
		for i, result := range results {
			if result == nil {
				continue
			}
			handle.StructuredData.Results = append(handle.StructuredData.Results,
				resourcev1alpha2.DriverAllocationResult{
					VendorRequestParameters: perDriver.parameters[i],
					AllocationResultModel: resourcev1alpha2.AllocationResultModel{
						NamedResources: result,
					},
				},
			)
		}
		if c.classParameters != nil {
			for _, p := range c.classParameters.VendorParameters {
				if p.DriverName == driverName {
					handle.StructuredData.VendorClassParameters = p.Parameters
					break
				}
			}
		}
		for _, request := range c.claimParameters.DriverRequests {
			if request.DriverName == driverName {
				handle.StructuredData.VendorClaimParameters = request.VendorParameters
				break
			}
		}
		allocation.ResourceHandles = append(allocation.ResourceHandles, handle)
	}

	return c.class.DriverName, allocation, nil
}
