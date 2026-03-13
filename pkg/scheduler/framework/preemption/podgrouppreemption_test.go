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
	"context"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestWorkloadExecutor_SelectVictimsOnDomain(t *testing.T) {
	type blockingRule struct {
		nodeName        string
		capacity        int
		blockingVictims []string
	}

	currentTime := metav1.Now()

	tests := []struct {
		name           string
		nodeNames      []string
		initPods       []*v1.Pod
		podGroupName   string
		preemptor      Preemptor
		pdbs           []*policy.PodDisruptionBudget
		blockingRules  []blockingRule
		expectedPods   [][]string
		expectedStatus []*fwk.Status
	}{
		{
			name:      "Basic: Mix of no-group and single-pod-groups",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			preemptor: NewPodGroupPreemptor(
				&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "preemptor-pg"}},
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
				[]fwk.CycleState{framework.NewCycleState()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: []string{"p1"}},
				{nodeName: "node2", capacity: 1, blockingVictims: []string{"p2"}},
				{nodeName: "node3", capacity: 1, blockingVictims: []string{"p3"}},
			},
			expectedPods:   [][]string{{"p1"}}, // p1 is less important than p2 because it's not part of a pod group
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Priority: Shared group vs no group",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(0, 0)).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(midPriority).Obj(),
			},
			preemptor: NewPodGroupPreemptor(
				&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "preemptor-pg"}},
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
				[]fwk.CycleState{framework.NewCycleState()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: []string{"p1"}},
				{nodeName: "node2", capacity: 1, blockingVictims: []string{"p2"}},
				{nodeName: "node3", capacity: 1, blockingVictims: []string{"p3"}},
			},
			expectedPods:   [][]string{{"p1"}}, // p1 is less important than p2 because of later StartTime
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Shared Group: Preempt separately",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(0, 0)).Obj(),
			},
			preemptor: NewPodGroupPreemptor(
				&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "preemptor-pg"}},
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
				[]fwk.CycleState{framework.NewCycleState()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: []string{"p1"}},
				{nodeName: "node2", capacity: 1, blockingVictims: []string{"p2"}},
			},
			expectedPods:   [][]string{{"p1"}}, // p1 is less important than p2
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Complex Mixed: Shared, different, and no groups",
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(2, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").StartTime(metav1.Unix(0, 0)).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(midPriority).Obj(),
				st.MakePod().Name("p5").UID("v5").Node("node5").Priority(highPriority).PodGroupName("pg3").StartTime(metav1.Unix(0, 0)).Obj(),
			},
			preemptor: NewPodGroupPreemptor(
				&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "preemptor-pg"}},
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj(),
					st.MakePod().Name("p-2").UID("p-2").Priority(highPriority).Obj(),
					st.MakePod().Name("p-3").UID("p-3").Priority(highPriority).Obj(),
				},
				[]fwk.CycleState{framework.NewCycleState()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: []string{"p1"}},
				{nodeName: "node2", capacity: 1, blockingVictims: []string{"p2"}},
				{nodeName: "node3", capacity: 1, blockingVictims: []string{"p3"}},
				{nodeName: "node4", capacity: 1, blockingVictims: []string{"p4"}},
				{nodeName: "node5", capacity: 1, blockingVictims: []string{"p5"}},
			},
			expectedPods:   [][]string{{"p1", "p2", "p3"}},
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "PDB: Mixed groups",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-pdb").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("victim-no-pdb").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: NewPodGroupPreemptor(
				&schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Name: "preemptor-pg"}},
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
				[]fwk.CycleState{framework.NewCycleState()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: []string{"victim-pdb"}},
				{nodeName: "node2", capacity: 1, blockingVictims: []string{"victim-no-pdb"}},
			},
			expectedPods:   [][]string{{"victim-no-pdb"}},
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Workload aware: all victims have the same priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(lowPriority).StartTime(metav1.NewTime(currentTime.Add(-1 * time.Hour))).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node1").Priority(lowPriority).StartTime(metav1.NewTime(currentTime.Add(-2 * time.Hour))).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node1").Priority(lowPriority).StartTime(metav1.NewTime(currentTime.Add(-3 * time.Hour))).Obj(),
			},
			preemptor: NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj(), framework.NewCycleState()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"p1"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"p2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"p3"}, capacity: 1},
			},
			expectedPods:   [][]string{{"p1"}},
			expectedStatus: []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Obj()
			}

			// Build nodes with pods
			nodeInfos := make(map[string]fwk.NodeInfo)
			for _, node := range nodes {
				nodeInfos[node.Name] = framework.NewNodeInfo()
				nodeInfos[node.Name].SetNode(node)
			}
			for _, p := range tt.initPods {
				podInfo, _ := framework.NewPodInfo(p)
				nodeInfos[p.Spec.NodeName].AddPodInfo(podInfo)
			}

			var domainNodes []fwk.NodeInfo
			for _, name := range tt.nodeNames {
				domainNodes = append(domainNodes, nodeInfos[name])
			}

			domain := NewDomainForWorkloadPreemption(domainNodes, "test-domain")

			// Create a mock podGroupSchedulingFunc
			mockSchedulingFunc := func(ctx context.Context) *fwk.Status {
				neededSlots := len(tt.preemptor.Members())
				availableSlots := 0
				nodeMap := make(map[string]fwk.NodeInfo)
				for _, n := range domainNodes {
					nodeMap[n.Node().Name] = n
				}

				for _, rule := range tt.blockingRules {
					node, exists := nodeMap[rule.nodeName]
					if !exists {
						continue
					}

					isBlocked := false
					for _, pod := range node.GetPods() {
						if slices.Contains(rule.blockingVictims, pod.GetPod().Name) {
							isBlocked = true
							break
						}
					}

					if !isBlocked {
						availableSlots += rule.capacity
					}
				}

				if availableSlots >= neededSlots {
					return fwk.NewStatus(fwk.Success)
				}
				return fwk.NewStatus(fwk.Unschedulable)
			}

			pl := &PodGroupEvaluator{
				podGroupSchedulingFunc: mockSchedulingFunc,
			}

			gotPods, gotStatus := pl.selectVictimsOnDomain(context.Background(), tt.preemptor, domain, tt.pdbs)
			if gotStatus != nil && !gotStatus.IsSuccess() {
				t.Logf("SelectVictimsOnDomain failed: %v", gotStatus.Message())
			}

			wantStatus := tt.expectedStatus[0]
			wantCode := fwk.Success
			if wantStatus != nil {
				wantCode = wantStatus.Code()
			}

			gotCode := fwk.Success
			if gotStatus != nil {
				gotCode = gotStatus.Code()
			}

			if gotCode != wantCode {
				t.Errorf("Status mismatch. Want %v, Got %v", wantCode, gotCode)
			}

			if wantCode != fwk.Success {
				return
			}

			var gotNames []string
			for _, p := range gotPods {
				gotNames = append(gotNames, p.Name)
			}
			sort.Strings(gotNames)

			wantNames := tt.expectedPods[0]
			sort.Strings(wantNames)

			if diff := cmp.Diff(wantNames, gotNames); diff != "" {
				t.Errorf("Victims mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
