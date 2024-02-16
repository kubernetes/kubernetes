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

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/labels"
	resourcev1alpha2listers "k8s.io/client-go/listers/resource/v1alpha2"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
)

// resources is a map "node name" -> "driver name" -> available and
// allocated resources per structured parameter model.
type resources map[string]map[string]resourceModels

// resourceModels may have more than one entry because it is valid for a driver to
// use more than one structured parameter model.
type resourceModels struct {
	// TODO: add some structured parameter model
}

// newResourceModel parses the available information about resources. Objects
// with an unknown structured parameter model silently ignored. An error gets
// logged later when parameters required for a pod depend on such an unknown
// model.
func newResourceModel(logger klog.Logger, nodeResourceSliceLister resourcev1alpha2listers.NodeResourceSliceLister, claimAssumeCache volumebinding.AssumeCache) (resources, error) {
	model := make(resources)

	slices, err := nodeResourceSliceLister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("list node resource slices: %w", err)
	}
	for _, slice := range slices {
		if model[slice.NodeName] == nil {
			model[slice.NodeName] = make(map[string]resourceModels)
		}
		resource := model[slice.NodeName][slice.DriverName]
		// TODO: add some structured parameter model
		model[slice.NodeName][slice.DriverName] = resource
	}

	objs := claimAssumeCache.List(nil)
	for _, obj := range objs {
		claim, ok := obj.(*resourcev1alpha2.ResourceClaim)
		if !ok {
			return nil, fmt.Errorf("got unexpected object of type %T from claim assume cache", obj)
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
				model[structured.NodeName] = make(map[string]resourceModels)
			}
			// resource := model[structured.NodeName][handle.DriverName]
			// TODO: add some structured parameter model
			// for _, result := range structured.Results {
			//     // Call AddAllocation for each known model. Each call itself needs to check for nil.
			// }
		}
	}

	return model, nil
}

func newClaimController(logger klog.Logger, class *resourcev1alpha2.ResourceClass, classParameters *resourcev1alpha2.ResourceClassParameters, claimParameters *resourcev1alpha2.ResourceClaimParameters) (*claimController, error) {
	// Each node driver is separate from the others. Each driver may have
	// multiple requests which need to be allocated together, so here
	// we have to collect them per model.
	// TODO: implement some structured parameters model

	c := &claimController{
		class:           class,
		classParameters: classParameters,
		claimParameters: claimParameters,
	}
	return c, nil
}

// claimController currently wraps exactly one structured parameter model.

type claimController struct {
	class           *resourcev1alpha2.ResourceClass
	classParameters *resourcev1alpha2.ResourceClassParameters
	claimParameters *resourcev1alpha2.ResourceClaimParameters
	// TODO: implement some structured parameters model
}

func (c claimController) nodeIsSuitable(ctx context.Context, nodeName string, resources resources) (bool, error) {
	// TODO: implement some structured parameters model
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

	// TODO: implement some structured parameters model

	return c.class.DriverName, allocation, nil
}
