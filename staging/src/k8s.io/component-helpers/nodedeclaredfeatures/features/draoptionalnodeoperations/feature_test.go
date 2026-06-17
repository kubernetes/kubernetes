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

package draoptionalnodeoperations

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
	"k8s.io/utils/ptr"
)

func TestDiscover(t *testing.T) {
	tests := []struct {
		name    string
		enabled bool
		want    bool
	}{
		{"gate enabled", true, true},
		{"gate disabled", false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{DRAOptionalNodeOperationsFeatureGate: tt.enabled},
			}
			if got := Feature.Discover(cfg); got != tt.want {
				t.Errorf("Discover() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInferForScheduling(t *testing.T) {
	claimSkip := &resourceapi.ResourceClaim{
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{
						{SkipNodeOperations: ptr.To(true)},
					},
				},
			},
		},
	}
	claimNoSkip := &resourceapi.ResourceClaim{
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{
						{SkipNodeOperations: ptr.To(false)},
					},
				},
			},
		},
	}

	tests := []struct {
		name    string
		podInfo *types.PodInfo
		want    bool
	}{
		{"nil podInfo", nil, false},
		{"no claims", &types.PodInfo{}, false},
		{"claim without skip", &types.PodInfo{ResourceClaims: []*resourceapi.ResourceClaim{claimNoSkip}}, false},
		{"claim with skip", &types.PodInfo{ResourceClaims: []*resourceapi.ResourceClaim{claimSkip}}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Feature.InferForScheduling(tt.podInfo); got != tt.want {
				t.Errorf("InferForScheduling() = %v, want %v", got, tt.want)
			}
		})
	}
}
