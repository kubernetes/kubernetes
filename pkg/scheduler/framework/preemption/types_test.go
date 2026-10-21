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

package preemption

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestNewVictim(t *testing.T) {
	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-10 * time.Minute))
	later := metav1.NewTime(now.Add(10 * time.Minute))

	p1 := st.MakePod().Name("p1").StartTime(now).Obj()
	p2 := st.MakePod().Name("p2").StartTime(later).Obj()
	p3 := st.MakePod().Name("p3").StartTime(earlier).Obj()
	p4 := st.MakePod().Name("p4").Obj() // nil start time
	p5 := st.MakePod().Name("p5").StartTime(now).PodGroupName("pg1").Obj()

	pi1, _ := framework.NewPodInfo(p1)
	pi2, _ := framework.NewPodInfo(p2)
	pi3, _ := framework.NewPodInfo(p3)
	pi4, _ := framework.NewPodInfo(p4)
	pi5, _ := framework.NewPodInfo(p5)

	tests := []struct {
		name       string
		podInfos   []fwk.PodInfo
		priority   int32
		keyType    fwk.EntityKeyType
		wantErr    bool
		wantVictim *victim
	}{
		{
			name:       "empty pods slice returns error",
			podInfos:   nil,
			priority:   10,
			keyType:    fwk.PodKeyType,
			wantErr:    true,
			wantVictim: nil,
		},
		{
			name:       "empty keyType returns error",
			podInfos:   []fwk.PodInfo{pi1},
			priority:   20,
			keyType:    "",
			wantErr:    true,
			wantVictim: nil,
		},
		{
			name:     "single pod without scheduling group",
			podInfos: []fwk.PodInfo{pi1},
			priority: 20,
			keyType:  fwk.PodKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          20,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi1},
				keyType:           fwk.PodKeyType,
			},
		},
		{
			name:     "multiple pods with mixed start times (including nil)",
			podInfos: []fwk.PodInfo{pi4, pi2, pi3},
			priority: 30,
			keyType:  fwk.PodKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          30,
				earliestStartTime: &earlier,
				pods:              []fwk.PodInfo{pi4, pi2, pi3},
				keyType:           fwk.PodKeyType,
			},
		},
		{
			name:     "pod with scheduling group is constructed as pod group",
			podInfos: []fwk.PodInfo{pi5},
			priority: 40,
			keyType:  fwk.PodGroupKeyType,
			wantErr:  false,
			wantVictim: &victim{
				priority:          40,
				earliestStartTime: &now,
				pods:              []fwk.PodInfo{pi5},
				keyType:           fwk.PodGroupKeyType,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotVictim, err := NewVictim(tt.podInfos, tt.priority, tt.keyType)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewVictim() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if diff := cmp.Diff(tt.wantVictim, gotVictim, cmp.AllowUnexported(victim{}, framework.PodInfo{})); diff != "" {
				t.Errorf("Victim mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewDomainVictim(t *testing.T) {
	tests := []struct {
		name              string
		nodes             []*v1.Node
		pods              []*v1.Pod
		priority          int32
		wantErr           bool
		wantAffectedNodes sets.Set[string]
	}{
		{
			name: "pod on missing node returns error",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Node("missing-node").Obj(),
			},
			priority:          10,
			wantErr:           true,
			wantAffectedNodes: nil,
		},
		{
			name: "multiple pods on same and different nodes deduplicates affectedNodes",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Obj(),
				st.MakePod().Name("p2").Node("node1").Obj(), // same node
				st.MakePod().Name("p3").Node("node2").Obj(), // different node
			},
			priority:          20,
			wantErr:           false,
			wantAffectedNodes: sets.New("node1", "node2"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			snapshot := internalcache.NewSnapshot(nil, tt.nodes)
			cache := internalcache.New(ctx, nil, true, false /* CompositePodGroup */)
			for _, node := range tt.nodes {
				cache.AddNode(logger, node)
			}
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			var podInfos []fwk.PodInfo
			for _, p := range tt.pods {
				pi, _ := framework.NewPodInfo(p)
				podInfos = append(podInfos, pi)
			}

			dv, err := newDomainVictim(snapshot, podInfos, tt.priority, fwk.PodKeyType)
			if (err != nil) != tt.wantErr {
				t.Errorf("newDomainVictim() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			gotNodes := sets.New[string]()
			for nName := range dv.AffectedNodes() {
				gotNodes.Insert(nName)
			}

			if diff := cmp.Diff(tt.wantAffectedNodes, gotNodes); diff != "" {
				t.Errorf("AffectedNodes() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
