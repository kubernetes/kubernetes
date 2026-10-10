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
				{op: opAddPod, obj: pod1A},
				{op: opAddPod, obj: pod1B},
				// PG arrives with parent set to root-cpg; quorum of 2 pods is satisfied, so root-cpg gets 1 ready child
				{op: opAddPG, obj: st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("root-cpg").MinCount(2).Obj(), wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 1}},
				// Root CPG arrives: ready children count remains 1
				{op: opAddCPG, obj: rootCPG, wantReadyCounts: map[fwk.EntityKey]int{rootCPGKey: 1}},
			},
		},
		{
			name: "Re-parenting PodGroup moves readiness count across CPGs",
			steps: []trackerStep{
				{op: opAddCPG, obj: cpg1, wantReadyCounts: map[fwk.EntityKey]int{cpg1Key: 0}},
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
				{op: opDeletePod, obj: pod1A},
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
					fwk.CompositePodGroupKey("ns1", "cpg-cycle-a"): 1,
					fwk.CompositePodGroupKey("ns1", "cpg-cycle-b"): 1,
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
					group, exists := tracker.groups[key]
					if !exists {
						t.Fatalf("step %d (%s): group %s was not found in the hierarchy tracker", stepIdx, step.op, key.String())
					}
					if gotCount := group.readyCount(); gotCount != wantCount {
						t.Errorf("step %d (%s): ReadyCount(%s) = %d, want %d", stepIdx, step.op, key.String(), gotCount, wantCount)
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
	pg3 := st.MakePodGroup().Namespace("ns1").Name("pg3").ParentCompositePodGroup("root-cpg").MinCount(1).Obj()
	standalonePG := st.MakePodGroup().Namespace("ns1").Name("standalone-pg").MinCount(2).Obj()
	basicRootCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("basic-root-cpg").BasicPolicy().Obj()
	basicChildPG := st.MakePodGroup().Namespace("ns1").Name("basic-child-pg").ParentCompositePodGroup("basic-root-cpg").BasicPolicy().Obj()
	basicStandalonePG := st.MakePodGroup().Namespace("ns1").Name("basic-standalone-pg").BasicPolicy().Obj()
	pg4 := st.MakePodGroup().Namespace("ns1").Name("pg4").ParentCompositePodGroup("basic-root-cpg").MinCount(2).Obj()

	tests := []struct {
		name         string
		cpgEnabled   bool
		groups       []*fwk.GenericPodGroup
		pods         []*v1.Pod
		lookupKey    fwk.EntityKey
		wantRootName string
		wantIsReady  bool
	}{
		{
			name:       "Standalone PodGroup returns itself when CPG enabled",
			cpgEnabled: true,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(standalonePG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("standalone-pg").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("standalone-pg").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "standalone-pg"),
			wantRootName: "standalone-pg",
			wantIsReady:  true,
		},
		{
			name:       "Standalone PodGroup returns unready when below minCount",
			cpgEnabled: true,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(standalonePG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("standalone-pg").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "standalone-pg"),
			wantRootName: "standalone-pg",
			wantIsReady:  false,
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
			lookupKey:    fwk.PodGroupKey("ns1", "pg1"),
			wantRootName: "root-cpg",
			wantIsReady:  false,
		},
		{
			name:       "Leaf PodGroup returns ready root CompositePodGroup when quorum met",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(rootCPG),
				fwk.NewGenericCompositePodGroup(midCPG),
				fwk.NewGenericPodGroup(pg1),
				fwk.NewGenericPodGroup(pg2),
				fwk.NewGenericPodGroup(pg3),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p3").UID("p3").PodGroupName("pg2").Obj(),
				st.MakePod().Namespace("ns1").Name("p4").UID("p4").PodGroupName("pg2").Obj(),
				st.MakePod().Namespace("ns1").Name("p5").UID("p5").PodGroupName("pg3").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "pg1"),
			wantRootName: "root-cpg",
			wantIsReady:  true,
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
			lookupKey:    fwk.CompositePodGroupKey("ns1", "mid-cpg"),
			wantRootName: "root-cpg",
			wantIsReady:  false,
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
			lookupKey:    fwk.CompositePodGroupKey("ns1", "root-cpg"),
			wantRootName: "root-cpg",
			wantIsReady:  false,
		},
		{
			name:         "Basic standalone PodGroup is unready with 0 pods",
			cpgEnabled:   true,
			groups:       []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(basicStandalonePG)},
			lookupKey:    fwk.PodGroupKey("ns1", "basic-standalone-pg"),
			wantRootName: "basic-standalone-pg",
			wantIsReady:  false,
		},
		{
			name:       "Basic standalone PodGroup is ready with 1 pod",
			cpgEnabled: true,
			groups:     []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(basicStandalonePG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("basic-standalone-pg").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "basic-standalone-pg"),
			wantRootName: "basic-standalone-pg",
			wantIsReady:  true,
		},
		{
			name:       "Basic root CompositePodGroup is ready with 1 ready child",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(basicRootCPG),
				fwk.NewGenericPodGroup(basicChildPG),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("basic-child-pg").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "basic-child-pg"),
			wantRootName: "basic-root-cpg",
			wantIsReady:  true,
		},
		{
			name:       "Basic root CompositePodGroup is not ready with 0 ready children",
			cpgEnabled: true,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(basicRootCPG),
				fwk.NewGenericPodGroup(pg4),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg4").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "pg4"),
			wantRootName: "basic-root-cpg",
			wantIsReady:  false,
		},
		{
			name:       "Leaf PodGroup returns itself when CPG disabled",
			cpgEnabled: false,
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(rootCPG),
				fwk.NewGenericCompositePodGroup(midCPG),
				fwk.NewGenericPodGroup(pg1),
			},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj(),
			},
			lookupKey:    fwk.PodGroupKey("ns1", "pg1"),
			wantRootName: "pg1",
			wantIsReady:  true,
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
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if rootReadiness == nil {
				t.Fatalf("expected non-nil rootReadiness")
			}
			if rootReadiness.RootGroup == nil || rootReadiness.RootGroup.GetName() != tt.wantRootName {
				t.Errorf("expected root name %q, got %v", tt.wantRootName, rootReadiness)
			}
			if rootReadiness.IsReady != tt.wantIsReady {
				t.Errorf("expected isReady %v, got %v", tt.wantIsReady, rootReadiness.IsReady)
			}
		})
	}
}

func TestHierarchyTracker_FindRootGroupReadiness_MissingObjects(t *testing.T) {
	danglingCPG := st.MakeCompositePodGroup().Namespace("ns1").Name("dangling-cpg").ParentCompositePodGroup("non-existent-cpg").MinGroupCount(1).Obj()
	danglingPG := st.MakePodGroup().Namespace("ns1").Name("dangling-pg").ParentCompositePodGroup("non-existent-cpg").MinCount(2).Obj()
	pgUnderDanglingCPG := st.MakePodGroup().Namespace("ns1").Name("pg-under-dangling-cpg").ParentCompositePodGroup("dangling-cpg").MinCount(1).Obj()

	tests := []struct {
		name      string
		groups    []*fwk.GenericPodGroup
		pods      []*v1.Pod
		lookupKey fwk.EntityKey
	}{
		{
			name:      "PodGroup with dangling parent returns nil without error",
			groups:    []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(danglingPG)},
			lookupKey: fwk.PodGroupKey("ns1", "dangling-pg"),
		},
		{
			name:   "Ready PodGroup with dangling parent placeholder returns nil without error",
			groups: []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(danglingPG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("dangling-pg").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("dangling-pg").Obj(),
			},
			lookupKey: fwk.PodGroupKey("ns1", "dangling-pg"),
		},
		{
			name:      "CompositePodGroup with dangling parent returns nil without error",
			groups:    []*fwk.GenericPodGroup{fwk.NewGenericCompositePodGroup(danglingCPG)},
			lookupKey: fwk.CompositePodGroupKey("ns1", "dangling-cpg"),
		},
		{
			name: "PodGroup under CompositePodGroup with dangling parent returns nil without error",
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(danglingCPG),
				fwk.NewGenericPodGroup(pgUnderDanglingCPG),
			},
			lookupKey: fwk.PodGroupKey("ns1", "pg-under-dangling-cpg"),
		},
		{
			name:      "Non-existent PodGroup key returns nil without error",
			lookupKey: fwk.PodGroupKey("ns1", "missing-pg"),
		},
		{
			name:      "Non-existent CompositePodGroup key returns nil without error",
			lookupKey: fwk.CompositePodGroupKey("ns1", "missing-cpg"),
		},
		{
			name: "PodGroup placeholder without PodGroup object returns nil without error",
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("missing-pg").Obj(),
			},
			lookupKey: fwk.PodGroupKey("ns1", "missing-pg"),
		},
		{
			name:   "CompositePodGroup placeholder without CompositePodGroup object returns nil without error",
			groups: []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(danglingPG)},
			pods: []*v1.Pod{
				st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("dangling-pg").Obj(),
				st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("dangling-pg").Obj(),
			},
			lookupKey: fwk.CompositePodGroupKey("ns1", "non-existent-cpg"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(true)
			for _, g := range tt.groups {
				tracker.AddGenericPodGroup(logger, g)
			}
			for _, pod := range tt.pods {
				tracker.AddPod(logger, pod)
			}

			rootReadiness, err := tracker.FindRootGroupReadiness(tt.lookupKey)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if rootReadiness != nil {
				t.Fatalf("expected nil rootReadiness, got %v", rootReadiness)
			}
		})
	}
}

func TestHierarchyTracker_FindRootGroupReadiness_Errors(t *testing.T) {
	cycleCPG1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle1").ParentCompositePodGroup("cycle2").Obj()
	cycleCPG2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle2").ParentCompositePodGroup("cycle1").Obj()
	pg1 := st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("cycle1").Obj()

	tests := []struct {
		name      string
		groups    []*fwk.GenericPodGroup
		lookupKey fwk.EntityKey
	}{
		{
			name:      "CPG",
			groups:    []*fwk.GenericPodGroup{fwk.NewGenericCompositePodGroup(cycleCPG1), fwk.NewGenericCompositePodGroup(cycleCPG2)},
			lookupKey: fwk.CompositePodGroupKey("ns1", "cycle1"),
		},
		{
			name: "PodGroup connects to cycle",
			groups: []*fwk.GenericPodGroup{
				fwk.NewGenericCompositePodGroup(cycleCPG1),
				fwk.NewGenericCompositePodGroup(cycleCPG2),
				fwk.NewGenericPodGroup(pg1)},
			lookupKey: fwk.PodGroupKey("ns1", "pg1"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(true)
			for _, g := range tt.groups {
				tracker.AddGenericPodGroup(logger, g)
			}

			rootReadiness, err := tracker.FindRootGroupReadiness(tt.lookupKey)
			if err == nil {
				t.Fatalf("expected error, got nil (rootReadiness: %v)", rootReadiness)
			}
		})
	}
}

func TestHierarchyTracker_AreSameHierarchy(t *testing.T) {
	root1 := st.MakeCompositePodGroup().Namespace("ns1").Name("root1").MinGroupCount(1).Obj()
	root2 := st.MakeCompositePodGroup().Namespace("ns1").Name("root2").MinGroupCount(1).Obj()
	mid := st.MakeCompositePodGroup().Namespace("ns1").Name("mid").ParentCompositePodGroup("root1").MinGroupCount(1).Obj()
	pg1 := st.MakePodGroup().Namespace("ns1").Name("pg1").ParentCompositePodGroup("mid").MinCount(1).Obj()
	pg2 := st.MakePodGroup().Namespace("ns1").Name("pg2").ParentCompositePodGroup("root1").MinCount(1).Obj()
	pg3 := st.MakePodGroup().Namespace("ns1").Name("pg3").ParentCompositePodGroup("root2").MinCount(1).Obj()

	groups := []*fwk.GenericPodGroup{
		fwk.NewGenericCompositePodGroup(root1),
		fwk.NewGenericCompositePodGroup(root2),
		fwk.NewGenericCompositePodGroup(mid),
		fwk.NewGenericPodGroup(pg1),
		fwk.NewGenericPodGroup(pg2),
		fwk.NewGenericPodGroup(pg3),
	}

	tests := []struct {
		name       string
		cpgEnabled bool
		key1       fwk.EntityKey
		key2       fwk.EntityKey
		want       bool
	}{
		{
			name:       "two pod groups in the same hierarchy",
			cpgEnabled: true,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.PodGroupKey("ns1", "pg2"),
			want:       true,
		},
		{
			name:       "pod group and composite pod group in the same hierarchy",
			cpgEnabled: true,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.CompositePodGroupKey("ns1", "root1"),
			want:       true,
		},
		{
			name:       "two pod groups in different hierarchies",
			cpgEnabled: true,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.PodGroupKey("ns1", "pg3"),
			want:       false,
		},
		{
			name:       "missing first key returns false",
			cpgEnabled: true,
			key1:       fwk.PodGroupKey("ns1", "missing"),
			key2:       fwk.PodGroupKey("ns1", "pg1"),
			want:       false,
		},
		{
			name:       "missing second key returns false",
			cpgEnabled: true,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.PodGroupKey("ns1", "missing"),
			want:       false,
		},
		{
			name:       "CPG disabled: same pod group returns true",
			cpgEnabled: false,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.PodGroupKey("ns1", "pg1"),
			want:       true,
		},
		{
			name:       "CPG disabled: sibling pod groups return false",
			cpgEnabled: false,
			key1:       fwk.PodGroupKey("ns1", "pg1"),
			key2:       fwk.PodGroupKey("ns1", "pg2"),
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(tt.cpgEnabled)
			for _, g := range groups {
				tracker.AddGenericPodGroup(logger, g)
			}

			got, err := tracker.AreSameHierarchy(tt.key1, tt.key2)
			if err != nil {
				t.Fatalf("AreSameHierarchy(%v, %v) failed unexpectedly: %v", tt.key1, tt.key2, err)
			}
			if got != tt.want {
				t.Errorf("AreSameHierarchy(%v, %v) = %v, want %v", tt.key1, tt.key2, got, tt.want)
			}
		})
	}
}

func TestHierarchyTracker_AreSameHierarchy_Errors(t *testing.T) {
	root1 := st.MakeCompositePodGroup().Namespace("ns1").Name("root1").MinGroupCount(1).Obj()
	cycleCPG1 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle1").ParentCompositePodGroup("cycle2").MinGroupCount(1).Obj()
	cycleCPG2 := st.MakeCompositePodGroup().Namespace("ns1").Name("cycle2").ParentCompositePodGroup("cycle1").MinGroupCount(1).Obj()

	groups := []*fwk.GenericPodGroup{
		fwk.NewGenericCompositePodGroup(root1),
		fwk.NewGenericCompositePodGroup(cycleCPG1),
		fwk.NewGenericCompositePodGroup(cycleCPG2),
	}

	tests := []struct {
		name string
		key1 fwk.EntityKey
		key2 fwk.EntityKey
	}{
		{
			name: "cycle detected in first key returns error",
			key1: fwk.CompositePodGroupKey("ns1", "cycle1"),
			key2: fwk.CompositePodGroupKey("ns1", "root1"),
		},
		{
			name: "cycle detected in second key returns error",
			key1: fwk.CompositePodGroupKey("ns1", "root1"),
			key2: fwk.CompositePodGroupKey("ns1", "cycle1"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tracker := NewHierarchyTracker(true)
			for _, g := range groups {
				tracker.AddGenericPodGroup(logger, g)
			}

			same, err := tracker.AreSameHierarchy(tt.key1, tt.key2)
			if err == nil {
				t.Fatalf("AreSameHierarchy(%v, %v) expected error, got nil (same: %v)", tt.key1, tt.key2, same)
			}
		})
	}
}
