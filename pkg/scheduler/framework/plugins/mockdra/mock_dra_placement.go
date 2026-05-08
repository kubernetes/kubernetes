/*
Copyright 2026 The Kubernetes Authors.

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

package mockdra

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.MockDRAPlacement
	// StateKey is the key for the cycle state.
	StateKey fwk.StateKey = "mockdra.allocations"
)

// AllocationData simulates the allocations DRA makes and saves to the PlacementCycleState.
type AllocationData struct {
	DeviceID string
	Model    string
	MemoryGB int
}

// Clone implements fwk.StateData
func (m *AllocationData) Clone() fwk.StateData {
	return &AllocationData{
		DeviceID: m.DeviceID,
		Model:    m.Model,
		MemoryGB: m.MemoryGB,
	}
}

// MockDRAPlacement is a mock plugin that generates placements representing DRA allocations.
type MockDRAPlacement struct {
	handle fwk.Handle
}

var _ fwk.PlacementGeneratePlugin = &MockDRAPlacement{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (*MockDRAPlacement, error) {
	return &MockDRAPlacement{
		handle: fh,
	}, nil
}

// Name returns name of the plugin.
func (pl *MockDRAPlacement) Name() string {
	return Name
}

// GeneratePlacements simulates a DRA plugin generating placements.
func (pl *MockDRAPlacement) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	// A real DRA plugin would inspect podGroup ResourceClaims, find available devices on nodes,
	// and generate possible combinations (Placements).
	
	// Fallback if no nodes available in parent placement
	if len(parentPlacement.Nodes) == 0 {
		return &fwk.GeneratePlacementsResult{}, nil
	}

	placementANodes := []fwk.NodeInfo{}
	placementBNodes := []fwk.NodeInfo{}
	placementCNodes := []fwk.NodeInfo{}
	
	// Group nodes based on a mock "dra-group" label to simulate different clusters of available hardware.
	for _, node := range parentPlacement.Nodes {
		if node.Node().Labels["dra-group"] == "group-A" {
			placementANodes = append(placementANodes, node)
		} else if node.Node().Labels["dra-group"] == "group-B" {
			placementBNodes = append(placementBNodes, node)
		} else if node.Node().Labels["dra-group"] == "group-C" {
			placementCNodes = append(placementCNodes, node)
		}
	}

	var placements []*fwk.Placement

	if len(placementANodes) > 0 {
		pACycleState := framework.NewCycleState()
		pACycleState.Write(StateKey, &AllocationData{
			DeviceID: "gpu-a-123",
			Model:    "A100",
			MemoryGB: 80,
		})
		state.SetPlacementCycleStateForName("dra-a", pACycleState)
		placements = append(placements, &fwk.Placement{
			Name: "dra-a",
			Nodes: placementANodes,
		})
	}

	if len(placementBNodes) > 0 {
		pBCycleState := framework.NewCycleState()
		pBCycleState.Write(StateKey, &AllocationData{
			DeviceID: "gpu-b-456",
			Model:    "H100",
			MemoryGB: 80,
		})
		state.SetPlacementCycleStateForName("dra-b", pBCycleState)
		placements = append(placements, &fwk.Placement{
			Name: "dra-b",
			Nodes: placementBNodes,
		})
	}

	if len(placementCNodes) > 0 {
		pCCycleState := framework.NewCycleState()
		pCCycleState.Write(StateKey, &AllocationData{
			DeviceID: "gpu-c-789",
			Model:    "B200",
			MemoryGB: 192,
		})
		state.SetPlacementCycleStateForName("dra-c", pCCycleState)
		placements = append(placements, &fwk.Placement{
			Name: "dra-c",
			Nodes: placementCNodes,
		})
	}

	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}
