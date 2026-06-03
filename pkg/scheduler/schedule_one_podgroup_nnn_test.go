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

package scheduler

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

func proposedAssignment(name, nominated, node string) algorithmResult {
	return algorithmResult{
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Status:     v1.PodStatus{NominatedNodeName: nominated},
		},
		scheduleResult: ScheduleResult{SuggestedHost: node},
	}
}

func TestNominatedNodesHonored(t *testing.T) {
	preempting := proposedAssignment("p1", "node1", "node1")
	preempting.requiresPreemption = true

	tests := []struct {
		name    string
		results []algorithmResult
		want    int
	}{
		{
			name: "no nominations",
			results: []algorithmResult{
				proposedAssignment("p1", "", "node1"),
				proposedAssignment("p2", "", "node2"),
			},
			want: 0,
		},
		{
			name: "all honored",
			results: []algorithmResult{
				proposedAssignment("p1", "node1", "node1"),
				proposedAssignment("p2", "node2", "node2"),
			},
			want: 2,
		},
		{
			name: "nominated but landed elsewhere does not count",
			results: []algorithmResult{
				proposedAssignment("p1", "node1", "node2"),
				proposedAssignment("p2", "node2", "node2"),
			},
			want: 1,
		},
		{
			name: "reaching the nominated node via preemption does not count",
			results: []algorithmResult{
				preempting,
				proposedAssignment("p2", "node2", "node2"),
			},
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := nominatedNodesHonored(&podGroupAlgorithmResult{podResults: tt.results})
			if got != tt.want {
				t.Errorf("nominatedNodesHonored() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestMorePreferredPlacement(t *testing.T) {
	pa := &fwk.Placement{Name: "a"}
	pb := &fwk.Placement{Name: "b"}

	tests := []struct {
		name      string
		candidate fwk.PlacementPluginScores
		current   fwk.PlacementPluginScores
		honored   map[*fwk.Placement]int
		want      bool
	}{
		{
			name:      "higher score wins despite fewer honored NNN",
			candidate: fwk.PlacementPluginScores{Placement: pa, TotalScore: 100, Randomizer: 0},
			current:   fwk.PlacementPluginScores{Placement: pb, TotalScore: 1, Randomizer: 100},
			honored:   map[*fwk.Placement]int{pa: 0, pb: 1},
			want:      true,
		},
		{
			name:      "lower score loses despite more honored NNN",
			candidate: fwk.PlacementPluginScores{Placement: pa, TotalScore: 1, Randomizer: 100},
			current:   fwk.PlacementPluginScores{Placement: pb, TotalScore: 100, Randomizer: 0},
			honored:   map[*fwk.Placement]int{pa: 1, pb: 0},
			want:      false,
		},
		{
			name:      "equal score breaks tie on honored NNN",
			candidate: fwk.PlacementPluginScores{Placement: pa, TotalScore: 5, Randomizer: 0},
			current:   fwk.PlacementPluginScores{Placement: pb, TotalScore: 5, Randomizer: 100},
			honored:   map[*fwk.Placement]int{pa: 1, pb: 0},
			want:      true,
		},
		{
			name:      "equal score and honored falls back to randomizer",
			candidate: fwk.PlacementPluginScores{Placement: pa, TotalScore: 5, Randomizer: 9},
			current:   fwk.PlacementPluginScores{Placement: pb, TotalScore: 5, Randomizer: 1},
			honored:   map[*fwk.Placement]int{pa: 0, pb: 0},
			want:      true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := morePreferredPlacement(tt.candidate, tt.current, tt.honored); got != tt.want {
				t.Errorf("morePreferredPlacement() = %v, want %v", got, tt.want)
			}
		})
	}
}
