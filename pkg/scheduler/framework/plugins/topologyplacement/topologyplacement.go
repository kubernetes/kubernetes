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

package topologyplacement

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
)

const Name = "TopologyPlacement"

type TopologyPlacement struct {
	handle fwk.Handle
}

var _ fwk.PlacementGeneratorPlugin = &TopologyPlacement{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &TopologyPlacement{
		handle: fh,
	}, nil
}

func (pl *TopologyPlacement) Name() string {
	return Name
}

func (pl *TopologyPlacement) GeneratePlacements(ctx context.Context, state *fwk.CycleState, PodGroupInfo *fwk.PodGroupInfo, parentPlacements []*fwk.ParentPlacement) ([]*fwk.Placement, *fwk.Status) {
	if PodGroupInfo.PodGroup == nil || PodGroupInfo.PodGroup.Spec.SchedulingConstraints == nil {
		return nil, nil
	}

	constraints := PodGroupInfo.PodGroup.Spec.SchedulingConstraints.TopologyConstraints
	if len(constraints) == 0 {
		return nil, nil
	}

	// Alpha Limitation: Only support the first constraint
	topologyKey := constraints[0].Level
	if topologyKey == "" {
		return nil, fwk.AsStatus(fmt.Errorf("topology constraint level cannot be empty"))
	}

	var resultPlacements []*fwk.Placement

	// Iterate over each Parent Placement provided by the fwk.
	// In the first call, this will likely contain one "Root" placement with all cluster nodes.
	for _, parent := range parentPlacements {

		// 1. Group the parent's nodes by the new topology key (e.g., "rack")
		// map[topologyValue] -> count/existence
		distinctValues := sets.NewString()

		// Optimization: Use the nodes provided by the parent, do not List() from the cluster.
		for _, node := range parent.PlacementNodes {
			if node == nil {
				continue
			}
			if val, ok := node.Labels[topologyKey]; ok {
				distinctValues.Insert(val)
			}
		}

		if distinctValues.Len() == 0 {
			continue
		}

		// 2. Generate a Child Placement for each distinct topology value found
		for _, val := range distinctValues.List() {

			// Create the specific selector for this level (e.g., rack=rack1)
			newTerm := v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      topologyKey,
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{val},
					},
				},
			}

			// Merge with Parent's NodeSelector to ensure we stay within the parent domain
			// If parent has no selector (Root), we just use the new term.
			var combinedSelector *v1.NodeSelector
			if parent.NodeSelector == nil {
				combinedSelector = &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{newTerm},
				}
			} else {
				// Deep copy parent selector to avoid mutating the shared parent object
				combinedSelector = parent.NodeSelector.DeepCopy()

				// Append the new requirement to existing terms (AND logic distribution)
				// Note: NodeSelector terms are ORed, Requirements inside are ANDed.
				// To strictly intersect (Parent AND Child), we must add the child requirement
				// to EVERY term in the parent's selector.
				for i := range combinedSelector.NodeSelectorTerms {
					combinedSelector.NodeSelectorTerms[i].MatchExpressions = append(
						combinedSelector.NodeSelectorTerms[i].MatchExpressions,
						newTerm.MatchExpressions[0],
					)
				}

				// If the parent had empty terms (matches everything), we must handle that case
				if len(combinedSelector.NodeSelectorTerms) == 0 {
					combinedSelector.NodeSelectorTerms = []v1.NodeSelectorTerm{newTerm}
				}
			}

			placement := &fwk.Placement{
				// ID can be hierarchical "ZoneA/Rack1" or just "Rack1" depending on requirement.
				// Using simple value for Alpha.
				NodeSelector: combinedSelector,
			}

			resultPlacements = append(resultPlacements, placement)
		}
	}

	if len(resultPlacements) == 0 {
		klog.V(4).InfoS("No nodes found matching topology key", "key", topologyKey, "podGroup", PodGroupInfo.PodGroup.Name)
		return nil, fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("no nodes found with topology label '%s'", topologyKey))
	}

	return resultPlacements, nil
}
