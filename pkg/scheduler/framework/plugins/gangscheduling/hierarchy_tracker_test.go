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

package gangscheduling

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type trackerOpType string

const (
	opAddPod    trackerOpType = "addPod"
	opUpdatePod trackerOpType = "updatePod"
	opDeletePod trackerOpType = "deletePod"
	opAddPG     trackerOpType = "addPG"
	opUpdatePG  trackerOpType = "updatePG"
	opDeletePG  trackerOpType = "deletePG"
	opAddCPG    trackerOpType = "addCPG"
	opUpdateCPG trackerOpType = "updateCPG"
	opDeleteCPG trackerOpType = "deleteCPG"
)

type trackerStep struct {
	op              trackerOpType
	obj             any
	oldObj          any
	wantReadyCounts map[fwk.EntityKey]int
}

func TestHierarchyTracker(t *testing.T) {
	rootCPGKey := fwk.CompositePodGroupKey("ns1", "root-cpg")
	midCPGKey := fwk.CompositePodGroupKey("ns1", "mid-cpg")
	cpg1Key := fwk.CompositePodGroupKey("ns1", "cpg1")
	cpg2Key := fwk.CompositePodGroupKey("ns1", "cpg2")

	rootCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("root-cpg").MinGroupCount(1).Obj()
	midCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("mid-cpg").ParentCompositePodGroup("root-cpg").MinGroupCount(2).Obj()
	pg1 := st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("mid-cpg").MinCount(2).Obj()
	pg2 := st.MakePodGroup().Namespace("ns1").Name("pg2").ParentCompositePodGroup("mid-cpg").MinCount(1).Obj()

	pod1A := st.MakePod().Namespace("ns1").Name("pod-1a").UID("uid-1a").PodGroupName("pg1").Obj()
	pod1B := st.MakePod().Namespace("ns1").Name("pod-1b").UID("uid-1b").PodGroupName("pg1").Obj()
	pod2A := st.MakePod().Namespace("ns1").Name("pod-2a").UID("uid-2a").PodGroupName("pg2").Obj()

	cpg1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg1").MinGroupCount(1).Obj()
	cpg2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg2").MinGroupCount(1).Obj()
	pgReParent := st.MakePodGroup().Namespace("ns1").Name("pg-reparent").ParentCompositePodGroup("cpg1").MinCount(1).Obj()
	pgReParentUpdated := st.MakePodGroup().Namespace("ns1").Name("pg-reparent").ParentCompositePodGroup("cpg2").MinCount(1).Obj()
	podReParent := st.MakePod().Namespace("ns1").Name("pod-reparent").UID("uid-reparent").PodGroupName("pg-reparent").Obj()

	tests := []struct {
		name  string
		steps []trackerStep
	}{
		{
			name: "3-level hierarchy propagation (Root CPG -> Mid CPG -> PG1, PG2)",
			steps: []trackerStep{
				{op: opAddCPG, obj: rootCPG, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opAddCPG, obj: midCPG, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 0}},
				{op: opAddPG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 0}},
				{op: opAddPG, obj: pg2, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 0}},
				// Pod 1A arrives: pg1 minCount=2 not met yet
				{op: opAddPod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 0}},
				// Pod 1B arrives: pg1 minCount=2 met -> midCPG gets 1 ready child, but midCPG minGroupCount=2 so rootCPG still 0
				{op: opAddPod, obj: pod1B, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 1}},
				// Pod 2A arrives: pg2 minCount=1 met -> midCPG gets 2 ready children -> midCPG ready -> rootCPG gets 1 ready child
				{op: opAddPod, obj: pod2A, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 1, midCPGKey: 2}},
				// Delete Pod 1B: pg1 unready -> midCPG unready -> rootCPG unready
				{op: opDeletePod, obj: pod1B, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, midCPGKey: 1}},
			},
		},
		{
			name: "Out of order arrival (Pod arrives before PG, PG arrives before CPG)",
			steps: []trackerStep{
				// Pods arrive before PG and CPG exist in tracker
				{op: opAddPod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opAddPod, obj: pod1B, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				// PG arrives with parent set to root-cpg; quorum of 2 pods is satisfied, so root-cpg gets 1 ready child
				{op: opAddPG, obj: st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("root-cpg").MinCount(2).Obj(), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 1}},
				// Root CPG arrives: ready children count remains 1
				{op: opAddCPG, obj: rootCPG, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 1}},
			},
		},
		{
			name: "Re-parenting PodGroup moves readiness count across CPGs",
			steps: []trackerStep{
				{op: opAddCPG, obj: cpg1, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 0}},
				{op: opAddCPG, obj: cpg2, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 0}},
				{op: opAddPG, obj: pgReParent, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 0}},
				{op: opAddPod, obj: podReParent, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 1, cpg2Key: 0}},
				// Update PG to point to cpg2: readiness should transfer from cpg1 to cpg2
				{op: opUpdatePG, obj: pgReParentUpdated, oldObj: pgReParent, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 1}},
				// Delete PG: cpg2 should drop to 0
				{op: opDeletePG, obj: pgReParentUpdated, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 0}},
			},
		},
		{
			name: "Basic scheduling policy evaluates readiness on first pod",
			steps: []trackerStep{
				{op: opAddCPG, obj: st.MakeCompositePodGroup().Namespace("ns1").Name("root-basic").BasicPolicy().Obj(), wantReadyCounts: map[fwk.EntityKey]int{fwk.CompositePodGroupKey("ns1", "root-basic"): 0}},
				{op: opAddPG, obj: st.MakePodGroup().Namespace("ns1").Name("child-basic").ParentCompositePodGroup("root-basic").BasicPolicy().Obj(), wantReadyCounts: map[fwk.EntityKey]int{fwk.CompositePodGroupKey("ns1", "root-basic"): 0}},
				{op: opAddPod, obj: st.MakePod().Namespace("ns1").Name("pod-basic-1").UID("uid-basic-1").PodGroupName("child-basic").Obj(), wantReadyCounts: map[fwk.EntityKey]int{fwk.CompositePodGroupKey("ns1", "root-basic"): 1}},
				{op: opDeletePod, obj: st.MakePod().Namespace("ns1").Name("pod-basic-1").UID("uid-basic-1").PodGroupName("child-basic").Obj(), wantReadyCounts: map[fwk.EntityKey]int{fwk.CompositePodGroupKey("ns1", "root-basic"): 0}},
			},
		},
		{
			name: "Nil safety guards against panics",
			steps: []trackerStep{
				{op: opAddPod, obj: (*v1.Pod)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opUpdatePod, obj: (*v1.Pod)(nil), oldObj: (*v1.Pod)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opDeletePod, obj: (*v1.Pod)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opAddPG, obj: (*schedulingv1beta1.PodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opUpdatePG, obj: (*schedulingv1beta1.PodGroup)(nil), oldObj: (*schedulingv1beta1.PodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opDeletePG, obj: (*schedulingv1beta1.PodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opAddCPG, obj: (*schedulingv1alpha3.CompositePodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opUpdateCPG, obj: (*schedulingv1alpha3.CompositePodGroup)(nil), oldObj: (*schedulingv1alpha3.CompositePodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opDeleteCPG, obj: (*schedulingv1alpha3.CompositePodGroup)(nil), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := NewHierarchyTracker()
			for stepIdx, step := range tt.steps {
				switch step.op {
				case opAddPod:
					tracker.OnPodAdd(step.obj.(*v1.Pod))
				case opUpdatePod:
					var oldP, newP *v1.Pod
					if step.oldObj != nil {
						oldP = step.oldObj.(*v1.Pod)
					}
					if step.obj != nil {
						newP = step.obj.(*v1.Pod)
					}
					tracker.OnPodUpdate(oldP, newP)
				case opDeletePod:
					tracker.OnPodDelete(step.obj.(*v1.Pod))
				case opAddPG:
					tracker.OnPodGroupAdd(step.obj.(*schedulingv1beta1.PodGroup))
				case opUpdatePG:
					var oldPG, newPG *schedulingv1beta1.PodGroup
					if step.oldObj != nil {
						oldPG = step.oldObj.(*schedulingv1beta1.PodGroup)
					}
					if step.obj != nil {
						newPG = step.obj.(*schedulingv1beta1.PodGroup)
					}
					tracker.OnPodGroupUpdate(oldPG, newPG)
				case opDeletePG:
					tracker.OnPodGroupDelete(step.obj.(*schedulingv1beta1.PodGroup))
				case opAddCPG:
					tracker.OnCompositePodGroupAdd(step.obj.(*schedulingv1alpha3.CompositePodGroup))
				case opUpdateCPG:
					var oldCPG, newCPG *schedulingv1alpha3.CompositePodGroup
					if step.oldObj != nil {
						oldCPG = step.oldObj.(*schedulingv1alpha3.CompositePodGroup)
					}
					if step.obj != nil {
						newCPG = step.obj.(*schedulingv1alpha3.CompositePodGroup)
					}
					tracker.OnCompositePodGroupUpdate(oldCPG, newCPG)
				case opDeleteCPG:
					tracker.OnCompositePodGroupDelete(step.obj.(*schedulingv1alpha3.CompositePodGroup))
				default:
					t.Fatalf("unknown step op %v at step %d", step.op, stepIdx)
				}

				for key, wantCount := range step.wantReadyCounts {
					gotCount := tracker.ReadyChildrenCount(key)
					if gotCount != wantCount {
						t.Errorf("step %d (%s): ReadyChildrenCount(%s) = %d, want %d", stepIdx, step.op, key.String(), gotCount, wantCount)
					}
				}
			}
		})
	}
}
