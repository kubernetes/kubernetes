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

package queue

import (
	"context"
	"testing"

	"k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/apiserver/pkg/util/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPodGroupWithParentCPGQueueing(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.CompositePodGroup, true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	q := NewTestQueue(ctx, newDefaultQueueSort())

	pgName := "test-pg"
	cpgName := "test-cpg"
	namespace := "default"

	// 1. Create and Add CPG
	cpg := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cpgName,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
		},
	}
	q.AddCompositePodGroup(cpg)

	// 2. Create and Add PG
	pg := &v1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: namespace,
		},
		Spec: v1alpha3.PodGroupSpec{
			SchedulingPolicy: v1alpha3.PodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
			ParentCompositePodGroupName: &cpgName,
		},
	}
	q.AddPodGroup(pg)

	t.Logf("After AddPodGroup, CPGs: %+v", q.cpgs)
	t.Logf("After AddPodGroup, Orphan PGs: %+v", q.orphanPGs)

	// 3. Create Pod and Add to queue
	pod := st.MakePod().Name("pod1").Namespace(namespace).UID("pod1").PodGroupName(pgName).Obj()
	
	q.Add(ctx, pod)

	pgKey := namespace + "/" + pgName
	if pgInfo, ok := q.pgs[pgKey]; ok {
		t.Logf("PG Parent: %p", pgInfo.Parent)
	}

	// Verify state
	cpgKey := namespace + "/" + cpgName
	cpgInfo, ok := q.cpgs[cpgKey]
	if !ok {
		t.Fatalf("Expected CPG to be in queue's cpgs map")
	}

	if !q.activeQ.has(cpgInfo) {
		t.Errorf("Expected CPG to be in activeQ")
	}
}

func TestMultiLevelCPGReadinessDecrease(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.CompositePodGroup, true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	q := NewTestQueue(ctx, newDefaultQueueSort())

	pgName := "test-pg"
	cpgBName := "test-cpg-b"
	cpgAName := "test-cpg-a"
	namespace := "default"

	// 1. Create and Add CPG A (Root)
	cpgA := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cpgAName,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
		},
	}
	q.AddCompositePodGroup(cpgA)

	// 2. Create and Add CPG B (Child of A)
	cpgB := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cpgBName,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
			ParentCompositePodGroupName: &cpgAName,
		},
	}
	q.AddCompositePodGroup(cpgB)

	// 3. Create and Add PG (Child of B)
	pg := &v1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: namespace,
		},
		Spec: v1alpha3.PodGroupSpec{
			SchedulingPolicy: v1alpha3.PodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
			ParentCompositePodGroupName: &cpgBName,
		},
	}
	q.AddPodGroup(pg)

	// 4. Create Pod and Add to queue
	pod := st.MakePod().Name("pod1").Namespace(namespace).UID("pod1").PodGroupName(pgName).Obj()
	q.Add(ctx, pod)

	// Verify all are ready and A is in activeQ
	cpgAKey := namespace + "/" + cpgAName
	
	cpgAInfo := q.cpgs[cpgAKey]

	if !q.activeQ.has(cpgAInfo) {
		t.Errorf("Expected CPG A to be in activeQ")
	}

	// 5. Remove PodGroup
	q.RemovePodGroup(pg)

	// Verify it is STILL in activeQ (lazy removal)
	if !q.activeQ.has(cpgAInfo) {
		t.Errorf("Expected CPG A to STILL be in activeQ (lazy removal)")
	}
}

func TestMultiLevelCPGWithHighMinGroupCount(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.CompositePodGroup, true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	q := NewTestQueue(ctx, newDefaultQueueSort())

	namespace := "default"
	rootCPGName := "root-cpg"
	childCPG1Name := "child-cpg-1"
	childCPG2Name := "child-cpg-2"
	pg1Name := "test-pg-1"
	pg2Name := "test-pg-2"

	// 1. Create and Add Root CPG (MinGroupCount: 2)
	rootCPG := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      rootCPGName,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 2,
				},
			},
		},
	}
	q.AddCompositePodGroup(rootCPG)

	// 2. Create and Add Child CPG 1 (MinGroupCount: 1)
	childCPG1 := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      childCPG1Name,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
			ParentCompositePodGroupName: &rootCPGName,
		},
	}
	q.AddCompositePodGroup(childCPG1)

	// 3. Create and Add Child CPG 2 (MinGroupCount: 1)
	childCPG2 := &v1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      childCPG2Name,
			Namespace: namespace,
		},
		Spec: v1alpha3.CompositePodGroupSpec{
			SchedulingPolicy: v1alpha3.CompositePodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangGroupSchedulingPolicy{
					MinGroupCount: 1,
				},
			},
			ParentCompositePodGroupName: &rootCPGName,
		},
	}
	q.AddCompositePodGroup(childCPG2)

	// 4. Create and Add PGs
	pg1 := &v1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pg1Name,
			Namespace: namespace,
		},
		Spec: v1alpha3.PodGroupSpec{
			SchedulingPolicy: v1alpha3.PodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
			ParentCompositePodGroupName: &childCPG1Name,
		},
	}
	q.AddPodGroup(pg1)

	pg2 := &v1alpha3.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pg2Name,
			Namespace: namespace,
		},
		Spec: v1alpha3.PodGroupSpec{
			SchedulingPolicy: v1alpha3.PodGroupSchedulingPolicy{
				Gang: &v1alpha3.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
			ParentCompositePodGroupName: &childCPG2Name,
		},
	}
	q.AddPodGroup(pg2)

	// Verify Root CPG is NOT ready
	rootKey := namespace + "/" + rootCPGName
	rootInfo := q.cpgs[rootKey]

	// 5. Add Pod to PG1
	pod1 := st.MakePod().Name("pod1").Namespace(namespace).UID("pod1").PodGroupName(pg1Name).Obj()
	q.Add(ctx, pod1)

	// 6. Add Pod to PG2
	pod2 := st.MakePod().Name("pod2").Namespace(namespace).UID("pod2").PodGroupName(pg2Name).Obj()
	q.Add(ctx, pod2)

	if !q.activeQ.has(rootInfo) {
		t.Errorf("Expected Root CPG to be in activeQ")
	}
}
