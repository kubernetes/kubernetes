/*
Copyright The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"strings"

	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// placementNameSeparator joins names of placements that are combined into a merged placement.
const placementNameSeparator = "/"

// validatePlacementNames ensures names from one plugin can be joined without making merged
// placement names ambiguous.
func validatePlacementNames(placements []*fwk.Placement) error {
	seen := make(map[string]struct{}, len(placements))
	for _, placement := range placements {
		if placement == nil {
			return fmt.Errorf("plugin returned a nil placement")
		}
		if strings.Contains(placement.Name, placementNameSeparator) {
			return fmt.Errorf("placement name %q must not contain %q", placement.Name, placementNameSeparator)
		}
		if _, ok := seen[placement.Name]; ok {
			return fmt.Errorf("plugin returned duplicate placement name %q", placement.Name)
		}
		seen[placement.Name] = struct{}{}
	}
	return nil
}

// mergePlacements combines two sets of placements into their cross product.
// Each merged placement contains the node intersection of one placement from each set.
// The function drops pairs with an empty node intersection, merges per-placement state
// into each combined placement, and deletes the parent placement states from state.
func mergePlacements(as, bs []*fwk.Placement, state fwk.PodGroupCycleState) ([]*fwk.Placement, *fwk.Status) {
	stateImpl, _ := state.(*framework.CycleState)

	// bs is fixed across the outer loop, so build each node-name set once up front.
	bNodeNames := make([]map[string]struct{}, len(bs))
	for i, b := range bs {
		names := make(map[string]struct{}, len(b.Nodes))
		for _, n := range b.Nodes {
			names[n.Node().Name] = struct{}{}
		}
		bNodeNames[i] = names
	}

	var result []*fwk.Placement
	for _, a := range as {
		for i, b := range bs {
			nodes := intersectNodesWithSet(a.Nodes, bNodeNames[i])
			if len(nodes) == 0 {
				continue
			}
			mergedPlacement := &fwk.Placement{
				Name:  a.Name + placementNameSeparator + b.Name,
				Nodes: nodes,
			}
			if stateImpl != nil {
				if err := stateImpl.MergePlacementStatesInto(mergedPlacement, a, b); err != nil {
					return nil, fwk.AsStatus(err)
				}
			}
			result = append(result, mergedPlacement)
		}
	}
	if stateImpl != nil {
		stateImpl.DeletePlacementStates(as...)
		stateImpl.DeletePlacementStates(bs...)
	}
	return result, nil
}

// intersectNodesWithSet returns the NodeInfos from as whose names exist in bNames.
// It preserves the order and NodeInfo instances of as and removes duplicates.
func intersectNodesWithSet(as []fwk.NodeInfo, bNames map[string]struct{}) []fwk.NodeInfo {
	var out []fwk.NodeInfo
	var added map[string]struct{}
	for _, n := range as {
		name := n.Node().Name
		if _, ok := bNames[name]; !ok {
			continue
		}
		if _, dup := added[name]; dup {
			continue
		}
		if added == nil {
			added = make(map[string]struct{}, min(len(as), len(bNames)))
		}
		added[name] = struct{}{}
		out = append(out, n)
	}
	return out
}
