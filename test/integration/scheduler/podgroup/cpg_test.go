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

package podgroup

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// TestCPGHierarchicalScheduling verifies that readiness and feasibility propagate correctly
// across a 3-level CPG hierarchy with mixed Gang and Basic policies.
//
// Tree structure:
//
//	                    cpg-root (Gang, Min: 2)
//	            /                 |                 \
//	   cpg-sub1 (Gang, Min:2)  cpg-sub2 (Basic)    cpg-sub3 (Gang, Min:2)
//	   /    |    \              /       \             /         \
//	 pg1   pg2   pg3          pg4       pg5         pg6        pg7
//	(S)   (S)   (F)          (S)       (S)         (F)        (F)
//
// [Gang] [Gang] [Gang]     [Basic]   [Gang]      [Gang]     [Gang]
//
// (S) = Success (pods schedule according to policy)
// (F) = Fail (pods cannot schedule)
// Note: pg6 fails because its parent cpg-sub3 fails (needs 2 children but pg7 fails).
func TestCPGHierarchicalScheduling(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:         true,
		features.GangScheduling:          true,
		features.CompositePodGroup:       true,
		features.WorkloadAwarePreemption: true,
	})

	// Create a large node to hold all successful pods (5 PGs * 3 pods = 15 pods)
	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-hierarchical",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// Create a Workload object for the entire hierarchy
	workload := st.MakeWorkload().Name("workload-cpg").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").GangPolicy(2).Children(
				st.MakeCompositePodGroupTemplate().Name("sub1-3-t").GangPolicy(2).Children(
					st.MakePodGroupTemplate().Name("gang-t1").MinCount(3),
				),
				st.MakeCompositePodGroupTemplate().Name("sub2-t").BasicPolicy().Children(
					st.MakePodGroupTemplate().Name("basic-t").BasicPolicy(),
					st.MakePodGroupTemplate().Name("gang-t2").MinCount(3),
				),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	// Level 1: Root CPG
	rootCPG := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "cpg-root", Namespace: ns},
		Spec: schedulingapi.CompositePodGroupSpec{
			WorkloadRef: &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg", TemplateName: "root-t"},
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	// Level 2: cpg-sub1 (Gang, Min: 2)
	cpgSub1 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "cpg-sub1", Namespace: ns},
		Spec: schedulingapi.CompositePodGroupSpec{
			WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg", TemplateName: "sub1-3-t"},
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub1, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create cpg-sub1: %v", err)
	}

	// Level 2: cpg-sub2 (Basic)
	cpgSub2 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "cpg-sub2", Namespace: ns},
		Spec: schedulingapi.CompositePodGroupSpec{
			WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg", TemplateName: "sub2-t"},
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicGroupSchedulingPolicy{},
			},
		},
	}
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub2, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create cpg-sub2: %v", err)
	}

	// Level 2: cpg-sub3 (Gang, Min: 2)
	cpgSub3 := &schedulingapi.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "cpg-sub3", Namespace: ns},
		Spec: schedulingapi.CompositePodGroupSpec{
			WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg", TemplateName: "sub1-3-t"},
			ParentCompositePodGroupName: ptr.To("cpg-root"),
			SchedulingPolicy: schedulingapi.CompositePodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangGroupSchedulingPolicy{MinGroupCount: 2},
			},
		},
	}
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub3, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create cpg-sub3: %v", err)
	}

	// Helper function to create PodGroups
	createPG := func(name string, template string, minCount int32, parentCPG string) {
		var policy schedulingapi.PodGroupSchedulingPolicy
		if minCount > 0 {
			policy = schedulingapi.PodGroupSchedulingPolicy{
				Gang: &schedulingapi.GangSchedulingPolicy{MinCount: minCount},
			}
		} else {
			policy = schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			}
		}
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-pg", TemplateName: template},
				SchedulingPolicy:            policy,
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	// Level 3: Leaves
	createPG("pg1", "gang-t1", 3, "cpg-sub1")
	createPG("pg2", "gang-t1", 3, "cpg-sub1")
	createPG("pg3", "gang-t1", 3, "cpg-sub1")

	createPG("pg4", "basic-t", 0, "cpg-sub2")
	createPG("pg5", "gang-t2", 3, "cpg-sub2")

	createPG("pg6", "gang-t1", 3, "cpg-sub3")
	createPG("pg7", "gang-t1", 3, "cpg-sub3")

	// Helper function to create Pods
	createPods := func(pgName string, count int, reqCPU string, schedulable bool) {
		for i := 0; i < count; i++ {
			pod := st.MakePod().Namespace(ns).Name(fmt.Sprintf("%s-pod-%d", pgName, i)).
				PodGroupName(pgName).Priority(100).Obj()

			pod.Spec.Containers = []v1.Container{{
				Name:  "container",
				Image: "image",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(reqCPU),
					},
				},
			}}

			if !schedulable {
				// Make it unschedulable by requesting too much CPU
				pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("100")
			}

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
			t.Logf("Created pod %s", pod.Name)
		}
	}

	// Create Pods for each PG
	// pg1 (S), pg2 (S), pg3 (F)
	createPods("pg1", 3, "1", true)
	createPods("pg2", 3, "1", true)
	createPods("pg3", 3, "1", false)

	// pg4 (Basic, S), pg5 (Gang, S)
	createPods("pg4", 3, "1", true)
	createPods("pg5", 3, "1", true)

	// pg6 (Gang, S), pg7 (Gang, F)
	createPods("pg6", 3, "1", true)
	createPods("pg7", 3, "1", false)

	// Wait for successful pods to be scheduled
	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
		"pg4-pod-0", "pg4-pod-1", "pg4-pod-2",
		"pg5-pod-0", "pg5-pod-1", "pg5-pod-2",
	}

	for _, podName := range successPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	// Log pod placement
	pods, err := cs.CoreV1().Pods(ns).List(testCtx.Ctx, metav1.ListOptions{})
	if err != nil {
		t.Logf("Failed to list pods: %v", err)
	} else {
		for _, p := range pods.Items {
			t.Logf("Pod %s scheduled to node %q", p.Name, p.Spec.NodeName)
		}
	}

	// Verify that failing pods are NOT scheduled
	failPods := []string{
		"pg3-pod-0", "pg3-pod-1", "pg3-pod-2",
		"pg6-pod-0", "pg6-pod-1", "pg6-pod-2",
		"pg7-pod-0", "pg7-pod-1", "pg7-pod-2",
	}
	for _, podName := range failPods {
		pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod %s: %v", podName, err)
		}
		if pod.Spec.NodeName != "" {
			t.Errorf("Pod %s should not be scheduled, but scheduled to %s", podName, pod.Spec.NodeName)
		}
	}
}

func TestPodGroupDependentRequiredValidation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:   true,
		features.CompositePodGroup: true,
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "pg-validation",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	// Case 1: ParentCompositePodGroupName set and WorkloadRef set -> Should PASS
	pgPass := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg-pass", Namespace: ns},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			ParentCompositePodGroupName: ptr.To("some-cpg"),
			WorkloadRef: &schedulingapi.WorkloadReference{
				WorkloadName: "some-workload",
				TemplateName: "some-template",
			},
		},
	}

	if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pgPass, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Expected PodGroup with ParentCompositePodGroupName and WorkloadRef to be created successfully, got error: %v", err)
	}

	// Case 2: ParentCompositePodGroupName set and WorkloadRef NOT set -> Should FAIL
	pgFail := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "pg-fail", Namespace: ns},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingPolicy: schedulingapi.PodGroupSchedulingPolicy{
				Basic: &schedulingapi.BasicSchedulingPolicy{},
			},
			ParentCompositePodGroupName: ptr.To("some-cpg"),
		},
	}

	if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pgFail, metav1.CreateOptions{}); err == nil {
		t.Fatalf("Expected creation of PodGroup with ParentCompositePodGroupName and no WorkloadRef to fail, but it passed")
	}
}
