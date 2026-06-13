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
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func placementWithNodes(name string, nodeNames ...string) *fwk.Placement {
	nodes := make([]fwk.NodeInfo, 0, len(nodeNames))
	for _, n := range nodeNames {
		ni := framework.NewNodeInfo()
		ni.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
		nodes = append(nodes, ni)
	}
	return &fwk.Placement{Name: name, Nodes: nodes}
}

func podGroupWithNominations(nominated ...string) *framework.QueuedPodGroupInfo {
	pgi := &framework.QueuedPodGroupInfo{PodGroupInfo: &framework.PodGroupInfo{}}
	for i, nnn := range nominated {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: string(rune('a' + i))},
			Status:     v1.PodStatus{NominatedNodeName: nnn},
		}
		pgi.QueuedPodInfos = append(pgi.QueuedPodInfos, &framework.QueuedPodInfo{
			PodInfo: &framework.PodInfo{Pod: pod},
		})
	}
	return pgi
}

func TestNominatedPlacement(t *testing.T) {
	rack1 := placementWithNodes("rack-1", "node-1")
	rack2 := placementWithNodes("rack-2", "node-2")
	rack12 := placementWithNodes("rack-12", "node-1", "node-2")
	rack23 := placementWithNodes("rack-23", "node-2", "node-3")

	tests := []struct {
		name       string
		placements []*fwk.Placement
		podGroup   *framework.QueuedPodGroupInfo
		want       *fwk.Placement
	}{
		{
			name:       "no nominations returns nil",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("", ""),
			want:       nil,
		},
		{
			name:       "nominated node not in any placement returns nil",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("node-3"),
			want:       nil,
		},
		{
			name:       "single placement holds the nominated node",
			placements: []*fwk.Placement{rack1, rack2},
			podGroup:   podGroupWithNominations("node-2", ""),
			want:       rack2,
		},
		{
			name:       "placement holding the most nominated nodes wins",
			placements: []*fwk.Placement{rack1, rack12},
			podGroup:   podGroupWithNominations("node-1", "node-2"),
			want:       rack12,
		},
		{
			name:       "placement honoring the most pods wins over one holding more nominated nodes",
			placements: []*fwk.Placement{rack1, rack23},
			podGroup:   podGroupWithNominations("node-1", "node-1", "node-1", "node-2", "node-3"),
			want:       rack1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nominatedPlacement(tt.placements, tt.podGroup); got != tt.want {
				t.Errorf("nominatedPlacement() = %v, want %v", got, tt.want)
			}
		})
	}
}
