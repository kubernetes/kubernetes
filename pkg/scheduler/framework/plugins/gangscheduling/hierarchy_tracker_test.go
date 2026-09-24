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
	"k8s.io/klog/v2/ktesting"
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

func toGenericPodGroup(obj any) *fwk.GenericPodGroup {
	if obj == nil {
		return nil
	}
	switch o := obj.(type) {
	case *schedulingv1beta1.PodGroup:
		if o == nil {
			return nil
		}
		return fwk.NewGenericPodGroup(o)
	case *schedulingv1alpha3.CompositePodGroup:
		if o == nil {
			return nil
		}
		return fwk.NewGenericCompositePodGroup(o)
	case *fwk.GenericPodGroup:
		return o
	default:
		return nil
	}
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
	pod1ARecreated := st.MakePod().Namespace("ns1").Name("pod-1a").UID("uid-1a-recreated").PodGroupName("pg2").Obj()

	cpg1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg1").MinGroupCount(1).Obj()
	cpg2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg2").MinGroupCount(1).Obj()
	pgReParent := st.MakePodGroup().Namespace("ns1").Name("pg-reparent").ParentCompositePodGroup("cpg1").MinCount(1).Obj()
	pgReParentUpdated := st.MakePodGroup().Namespace("ns1").Name("pg-reparent").ParentCompositePodGroup("cpg2").MinCount(1).Obj()
	podReParent := st.MakePod().Namespace("ns1").Name("pod-reparent").UID("uid-reparent").PodGroupName("pg-reparent").Obj()

	cpgCycleA := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg-cycle-a").ParentCompositePodGroup("cpg-cycle-b").MinGroupCount(1).Obj()
	cpgCycleB := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg-cycle-b").ParentCompositePodGroup("cpg-cycle-a").MinGroupCount(1).Obj()
	pgCycle := st.MakePodGroup().Namespace("ns1").Name("pg-cycle").ParentCompositePodGroup("cpg-cycle-a").MinCount(1).Obj()
	podCycle := st.MakePod().Namespace("ns1").Name("pod-cycle").UID("uid-cycle").PodGroupName("pg-cycle").Obj()

	deepCPG1 := st.MakeCompositePodGroup().Namespace("ns1").Name("deep-cpg1").MinGroupCount(1).Obj()
	deepCPG2 := st.MakeCompositePodGroup().Namespace("ns1").Name("deep-cpg2").ParentCompositePodGroup("deep-cpg1").MinGroupCount(1).Obj()
	deepCPG3 := st.MakeCompositePodGroup().Namespace("ns1").Name("deep-cpg3").ParentCompositePodGroup("deep-cpg2").MinGroupCount(1).Obj()
	deepCPG4 := st.MakeCompositePodGroup().Namespace("ns1").Name("deep-cpg4").ParentCompositePodGroup("deep-cpg3").MinGroupCount(1).Obj()
	deepPG := st.MakePodGroup().Namespace("ns1").Name("deep-pg").ParentCompositePodGroup("deep-cpg4").MinCount(1).Obj()
	deepPod := st.MakePod().Namespace("ns1").Name("pod-deep").UID("uid-deep").PodGroupName("deep-pg").Obj()

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
				{op: opUpdatePG, obj: pgReParentUpdated, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0, cpg2Key: 1}},
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
			name: "Preserve active pods when PodGroup is deleted and re-created",
			steps: []trackerStep{
				{op: opAddCPG, obj: rootCPG, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0}},
				{op: opAddPG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, fwk.PodGroupKey("ns1", "pg1"): 0}},
				{op: opAddPod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, fwk.PodGroupKey("ns1", "pg1"): 1}},
				{op: opAddPod, obj: pod1B, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, fwk.PodGroupKey("ns1", "pg1"): 2}},
				// Delete PodGroup pg1: readiness retracted from parent, but activePods are retained
				{op: opDeletePG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 0, fwk.PodGroupKey("ns1", "pg1"): 2}},
				// Re-create PodGroup pg1: active pods are still present, so parent readiness is restored once mid-cpg is also satisfied or directly
				{op: opAddPG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 2}},
			},
		},
		{
			name: "Prune deleted PodGroup when all pods are removed",
			steps: []trackerStep{
				{op: opAddPG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 0}},
				{op: opAddPod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 1}},
				{op: opDeletePG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 1}},
				// When last pod is deleted, empty group is removed from tracking
				{op: opDeletePod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 0}},
			},
		},
		{
			name: "Collapsed pod delete and recreate moves the pod between groups",
			steps: []trackerStep{
				{op: opAddPG, obj: pg1, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 0}},
				{op: opAddPG, obj: pg2, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg2"): 0}},
				{op: opAddPod, obj: pod1A, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 1, fwk.PodGroupKey("ns1", "pg2"): 0}},
				// Different UID and different group: retract from pg1, count in pg2.
				{op: opUpdatePod, oldObj: pod1A, obj: pod1ARecreated, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 0, fwk.PodGroupKey("ns1", "pg2"): 1}},
				// Same pod on both sides: nothing changes.
				{op: opUpdatePod, oldObj: pod1ARecreated, obj: pod1ARecreated, wantReadyCounts: map[fwk.EntityKey]int{fwk.PodGroupKey("ns1", "pg1"): 0, fwk.PodGroupKey("ns1", "pg2"): 1}},
			},
		},
		{
			name: "Parent cycle terminates readiness propagation",
			steps: []trackerStep{
				{op: opAddCPG, obj: cpgCycleA},
				{op: opAddCPG, obj: cpgCycleB},
				{op: opAddPG, obj: pgCycle},
				{op: opAddPod, obj: podCycle, wantReadyCounts: map[fwk.EntityKey]int{
					fwk.PodGroupKey("ns1", "pg-cycle"):             1,
					fwk.CompositePodGroupKey("ns1", "cpg-cycle-a"): 2,
					fwk.CompositePodGroupKey("ns1", "cpg-cycle-b"): 1,
				}},
			},
		},
		{
			name: "Hierarchy deeper than WorkloadMaxTreeDepth stops propagating",
			steps: []trackerStep{
				{op: opAddCPG, obj: deepCPG1},
				{op: opAddCPG, obj: deepCPG2},
				{op: opAddCPG, obj: deepCPG3},
				{op: opAddCPG, obj: deepCPG4},
				{op: opAddPG, obj: deepPG},
				{op: opAddPod, obj: deepPod, wantReadyCounts: map[fwk.EntityKey]int{
					fwk.PodGroupKey("ns1", "deep-pg"):            1,
					fwk.CompositePodGroupKey("ns1", "deep-cpg4"): 1,
					fwk.CompositePodGroupKey("ns1", "deep-cpg3"): 1,
					fwk.CompositePodGroupKey("ns1", "deep-cpg2"): 1,
					fwk.CompositePodGroupKey("ns1", "deep-cpg1"): 0,
				}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(true)
			for stepIdx, step := range tt.steps {
				switch step.op {
				case opAddPod:
					tracker.AddPod(logger, step.obj.(*v1.Pod))
				case opUpdatePod:
					var oldP, newP *v1.Pod
					if step.oldObj != nil {
						oldP = step.oldObj.(*v1.Pod)
					}
					if step.obj != nil {
						newP = step.obj.(*v1.Pod)
					}
					tracker.UpdatePod(logger, oldP, newP)
				case opDeletePod:
					tracker.DeletePod(logger, step.obj.(*v1.Pod))
				case opAddPG, opAddCPG:
					tracker.AddGenericPodGroup(logger, toGenericPodGroup(step.obj))
				case opUpdatePG, opUpdateCPG:
					tracker.UpdateGenericPodGroup(logger, toGenericPodGroup(step.obj))
				case opDeletePG, opDeleteCPG:
					tracker.DeleteGenericPodGroup(logger, toGenericPodGroup(step.obj))
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

func TestHierarchyTracker_FindRootGroupReadiness(t *testing.T) {
	rootCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("root-cpg").MinGroupCount(2).Obj()
	midCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("mid-cpg").ParentCompositePodGroup("root-cpg").MinGroupCount(2).Obj()
	pg1 := st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("mid-cpg").MinCount(2).Obj()
	pg2 := st.MakePodGroup().Namespace("ns1").Name("pg2").ParentCompositePodGroup("mid-cpg").MinCount(2).Obj()
	standalonePG := st.MakePodGroup().Namespace("ns1").Name("standalone-pg").MinCount(2).Obj()
	danglingPG := st.MakePodGroup().Namespace("ns1").Name("dangling-pg").ParentCompositePodGroup("non-existent-cpg").MinCount(2).Obj()

	cycleCPG1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle1").ParentCompositePodGroup("cycle2").Obj()
	cycleCPG2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle2").ParentCompositePodGroup("cycle1").Obj()

	cpg1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg1").Obj()
	cpg2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg2").ParentCompositePodGroup("cpg1").Obj()
	cpg3 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg3").ParentCompositePodGroup("cpg2").Obj()
	cpg4 := st.MakeCompositePodGroup().Namespace("ns1").Name("cpg4").ParentCompositePodGroup("cpg3").Obj()
	deepPG := st.MakePodGroup().Namespace("ns1").Name("deep-pg").ParentCompositePodGroup("cpg4").Obj()

	tests := []struct {
		name              string
		cpgEnabled        bool
		groups            []*fwk.GenericPodGroup
		pods              []*v1.Pod
		lookupKey         fwk.EntityKey
		wantRootName      string
		wantReadyChildren int
		wantNilRoot       bool
		wantErr           string
	}{
		{
			name:       "Standalone PodGroup returns itself when CPG enabled",
			cpgEnabled: true,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(standalonePG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("standalone-pg").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("standalone-pg").Obj(),
			},
			lookupKey:         fwk.PodGroupKey("ns1", "standalone-pg"),
			wantRootName:      "standalone-pg",
			wantReadyChildren: 2,
		},
		{
			name:       "Leaf PodGroup returns root CompositePodGroup when CPG enabled",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(rootCPG),
				fwk.NewGenericCompositePodGroup(midCPG),
				fwk.NewGenericPodGroup(pg1),
				fwk.NewGenericPodGroup(pg2),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg2").Obj(),
				st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj(),
			},
			lookupKey:         fwk.PodGroupKey("ns1", "pg1"),
			wantRootName:      "root-cpg",
			wantReadyChildren: 1,
		},
		{
			name:       "Mid CompositePodGroup returns root CompositePodGroup when CPG enabled",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(rootCPG),
				fwk.NewGenericCompositePodGroup(midCPG),
				fwk.NewGenericPodGroup(pg1),
				fwk.NewGenericPodGroup(pg2),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg2").Obj(),
				st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj(),
			},
			lookupKey:         fwk.CompositePodGroupKey("ns1", "mid-cpg"),
			wantRootName:      "root-cpg",
			wantReadyChildren: 1,
		},
		{
			name:       "Root CompositePodGroup returns itself when CPG enabled",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(rootCPG),
				fwk.NewGenericCompositePodGroup(midCPG),
				fwk.NewGenericPodGroup(pg1),
				fwk.NewGenericPodGroup(pg2),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg2").Obj(),
				st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj(),
			},
			lookupKey:         fwk.CompositePodGroupKey("ns1", "root-cpg"),
			wantRootName:      "root-cpg",
			wantReadyChildren: 1,
		},
		{
			name:        "PodGroup with dangling parent returns nil without error when CPG enabled",
			cpgEnabled:  true,
			groups:      []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(danglingPG)},
			lookupKey:   fwk.PodGroupKey("ns1", "dangling-pg"),
			wantNilRoot: true,
		},
		{
			name:        "Non-existent key returns nil without error when CPG enabled",
			cpgEnabled:  true,
			lookupKey:   fwk.PodGroupKey("ns1", "missing"),
			wantNilRoot: true,
		},
		{
			name:       "Leaf PodGroup returns itself when CPG disabled",
			cpgEnabled: false,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(pg1)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
			},
			lookupKey:         fwk.PodGroupKey("ns1", "pg1"),
			wantRootName:      "pg1",
			wantReadyChildren: 2,
		},
		{
			name:       "Dangling PodGroup returns itself when CPG disabled",
			cpgEnabled: false,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(danglingPG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("dangling-pg").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("dangling-pg").Obj(),
			},
			lookupKey:         fwk.PodGroupKey("ns1", "dangling-pg"),
			wantRootName:      "dangling-pg",
			wantReadyChildren: 2,
		},
		{
			name:       "Hierarchy cycle exceeding max tree depth returns error",
			cpgEnabled: true,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericCompositePodGroup(cycleCPG1), fwk.NewGenericCompositePodGroup(cycleCPG2)},
			lookupKey:  fwk.CompositePodGroupKey("ns1", "cycle1"),
			wantErr:    "hierarchy exceeded maximum tree depth at compositepodgroup/ns1/cycle1, possibly caused by cycle or deep hierarchy",
		},
		{
			name:       "Hierarchy exceeding max tree depth returns error",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(cpg1),
				fwk.NewGenericCompositePodGroup(cpg2),
				fwk.NewGenericCompositePodGroup(cpg3),
				fwk.NewGenericCompositePodGroup(cpg4),
				fwk.NewGenericPodGroup(deepPG),
			},
			lookupKey: fwk.PodGroupKey("ns1", "deep-pg"),
			wantErr:   "hierarchy exceeded maximum tree depth at compositepodgroup/ns1/cpg1, possibly caused by cycle or deep hierarchy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(tt.cpgEnabled)
			for _, g := range tt.groups {
				tracker.AddGenericPodGroup(logger, g)
			}
			for _, pod := range tt.pods {
				tracker.AddPod(logger, pod)
			}

			rootReadiness, err := tracker.FindRootGroupReadiness(tt.lookupKey)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error %q, got nil (rootReadiness: %v)", tt.wantErr, rootReadiness)
				}
				if err.Error() != tt.wantErr {
					t.Errorf("expected error %q, got %q", tt.wantErr, err.Error())
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantNilRoot {
				if rootReadiness != nil {
					t.Errorf("expected nil rootReadiness, got %v", rootReadiness)
				}
				return
			}

			if rootReadiness == nil || rootReadiness.RootGroup == nil || rootReadiness.RootGroup.GetName() != tt.wantRootName {
				t.Errorf("expected root name %q, got %v", tt.wantRootName, rootReadiness)
			}
			if rootReadiness != nil && rootReadiness.ReadyChildren != tt.wantReadyChildren {
				t.Errorf("expected readyChildren %d, got %d", tt.wantReadyChildren, rootReadiness.ReadyChildren)
			}
		})
	}
}
