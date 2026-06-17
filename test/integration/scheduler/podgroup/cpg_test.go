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
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
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
			st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Children(
				st.MakeCompositePodGroupTemplate().Name("sub1-3-t").MinGroupCount(2).Children(
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
	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg", "root-t").MinGroupCount(2).Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	// Level 2: cpg-sub1 (Gang, Min: 2)
	cpgSub1 := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-sub1").WorkloadRef("workload-cpg", "sub1-3-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Obj()

	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub1, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create cpg-sub1: %v", err)
	}

	// Level 2: cpg-sub2 (Basic)
	cpgSub2 := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-sub2").WorkloadRef("workload-cpg", "sub2-t").BasicPolicy().ParentCompositePodGroup("cpg-root").Obj()

	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub2, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create cpg-sub2: %v", err)
	}

	// Level 2: cpg-sub3 (Gang, Min: 2)
	cpgSub3 := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-sub3").WorkloadRef("workload-cpg", "sub1-3-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Obj()

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
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
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

// TestCPGMinGroupCount verifies that a Gang CompositePodGroup schedules its children
// if at least MinGroupCount of them are fully schedulable.
//
// Tree structure:
//
//	   cpg-root (Gang, MinGroup: 2)
//	  /             |              \
//	pg1            pg2            pg3
//	(S)            (S)            (F)
//
// (S) = Success
// (F) = Fail
func TestCPGMinGroupCount(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-min-group-count",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-min").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Children(
				st.MakePodGroupTemplate().Name("gang-t").MinCount(3),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-min", "root-t").MinGroupCount(2).Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, minCount int32, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-min", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Gang: &schedulingapi.GangSchedulingPolicy{MinCount: minCount}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPG("pg1", "gang-t", 3, "cpg-root")
	createPG("pg2", "gang-t", 3, "cpg-root")
	createPG("pg3", "gang-t", 3, "cpg-root")

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

			if !schedulable && i == 0 {
				pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("100")
			}

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	createPods("pg1", 3, "1", true)
	createPods("pg2", 3, "1", true)
	createPods("pg3", 3, "1", false)

	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
	}

	for _, podName := range successPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	failPods := []string{
		"pg3-pod-0", "pg3-pod-1", "pg3-pod-2",
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

// TestCPGBasicWithGangChildren verifies that a Basic CompositePodGroup allows its ready
// child Gang groups to schedule independently of each other.
//
// Tree structure:
//
//	   cpg-root (Basic)
//	  /            \
//	pg1            pg2
//	(S)            (F)
//
// (S) = Success
// (F) = Fail
func TestCPGBasicWithGangChildren(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-basic-gang-children",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-basic").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Children(
				st.MakePodGroupTemplate().Name("gang-t").MinCount(3),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-basic", "root-t").BasicPolicy().Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, minCount int32, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-basic", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Gang: &schedulingapi.GangSchedulingPolicy{MinCount: minCount}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPG("pg1", "gang-t", 3, "cpg-root")
	createPG("pg2", "gang-t", 3, "cpg-root")

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

			if !schedulable && i == 0 {
				pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("100")
			}

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	createPods("pg1", 3, "1", true)
	createPods("pg2", 3, "1", false)

	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
	}

	for _, podName := range successPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	failPods := []string{
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
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

// TestCPGBasicWithBasicChildren verifies that a Basic CompositePodGroup with Basic child
// pod groups schedules all pods successfully since there are no gang constraints.
//
// Tree structure:
//
//	   cpg-root (Basic)
//	  /            \
//	pg1            pg2
//	(S)            (S)
//
// (S) = Success
func TestCPGBasicWithBasicChildren(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-basic-basic-children",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-basic-basic").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Children(
				st.MakePodGroupTemplate().Name("basic-t").BasicPolicy(),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-basic-basic", "root-t").BasicPolicy().Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-basic-basic", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Basic: &schedulingapi.BasicSchedulingPolicy{}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPG("pg1", "basic-t", "cpg-root")
	createPG("pg2", "basic-t", "cpg-root")

	createPods := func(pgName string, count int, reqCPU string) {
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

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	createPods("pg1", 2, "1")
	createPods("pg2", 2, "1")

	expectedPods := []string{
		"pg1-pod-0", "pg1-pod-1",
		"pg2-pod-0", "pg2-pod-1",
	}

	for _, podName := range expectedPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}
}

// TestCPGGangWithBasicChildren verifies that a Gang CompositePodGroup with Basic children
// enforces the gang scheduling policy (minGroupCount) at the root level.
//
// Tree structure:
//
//	   cpg-root (Gang, Min: 2)
//	  /          |          \
//	pg1         pg2         pg3
//	(S)         (S)         (F)
//
// (S) = Success
// (F) = Fail
func TestCPGGangWithBasicChildren(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	// Capacity matches 2 groups (6 CPUs total). pg3 pods request 10 CPUs each and will fail.
	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-gang-basic-children",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-gang-basic").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").MinGroupCount(2).Children(
				st.MakePodGroupTemplate().Name("basic-t").BasicPolicy(),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-gang-basic", "root-t").MinGroupCount(2).Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-gang-basic", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Basic: &schedulingapi.BasicSchedulingPolicy{}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPG("pg1", "basic-t", "cpg-root")
	createPG("pg2", "basic-t", "cpg-root")
	createPG("pg3", "basic-t", "cpg-root")

	createPods := func(pgName string, count int, reqCPU string) {
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

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	// Create pods. pg1 and pg2 fit, but pg3 fails. MinGroupCount=2 is satisfied.
	createPods("pg1", 3, "1")
	createPods("pg2", 3, "1")
	createPods("pg3", 3, "10")

	successPods := []string{
		"pg1-pod-0", "pg1-pod-1", "pg1-pod-2",
		"pg2-pod-0", "pg2-pod-1", "pg2-pod-2",
	}

	for _, podName := range successPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	failPods := []string{
		"pg3-pod-0", "pg3-pod-1", "pg3-pod-2",
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

// TestCPGBasicWithUnschedulableBasicChildren verifies that when a Basic CompositePodGroup has
// unschedulable Basic child pod groups, the pods remain unscheduled.
//
// Tree structure:
//
//	   cpg-root (Basic)
//	  /            \
//	pg1            pg2
//	(F)            (F)
//
// (F) = Fail
func TestCPGBasicWithUnschedulableBasicChildren(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-unschedulable-basic",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := "default"

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-basic-unschedulable").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Children(
				st.MakePodGroupTemplate().Name("basic-t").BasicPolicy(),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-basic-unschedulable", "root-t").BasicPolicy().Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-basic-unschedulable", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Basic: &schedulingapi.BasicSchedulingPolicy{}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPG("pg1", "basic-t", "cpg-root")
	createPG("pg2", "basic-t", "cpg-root")

	createPods := func(pgName string, count int, reqCPU string) {
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

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	// Requesting 50 CPUs each on a node with 20 capacity makes them unschedulable.
	createPods("pg1", 2, "50")
	createPods("pg2", 2, "50")

	expectedPods := []string{
		"pg1-pod-0", "pg1-pod-1",
		"pg2-pod-0", "pg2-pod-1",
	}

	// Wait briefly to allow scheduling cycles to attempt and fail.
	time.Sleep(2 * time.Second)

	for _, podName := range expectedPods {
		pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod %s: %v", podName, err)
		}
		if pod.Spec.NodeName != "" {
			t.Errorf("Pod %s should not be scheduled, but scheduled to %s", podName, pod.Spec.NodeName)
		}
	}
}

// TestCPGDynamicChildAddition verifies that after a CompositePodGroup (CPG) hierarchy
// successfully schedules its initial pods, dynamically adding new child groups and sub-CPGs
// behaves correctly under scheduling policies (e.g., Gang scheduling with minGroupCount).
//
// Tree structure progression:
//
// Stage 1:
//
//	cpg-root (Basic)
//	   |
//	  pg1 (Basic)
//	  (Scheduled)
//
// Stage 2 (Dynamically added pg2):
//
//	 cpg-root (Basic)
//	  /      \
//	pg1      pg2 (Basic)
//	(S)      (Scheduled)
//
// Stage 3 (Dynamically added cpg-sub tree under cpg-root):
//
//	 cpg-root (Basic)
//	  /   |  \
//	pg1  pg2  cpg-sub (Gang minGroupCount = 2)
//	(S)  (S)     |
//	           sub-pg1 (Basic)
//	           (Pending - quorum of 2 not met)
//
// Stage 4 (Dynamically added sub-pg2 under cpg-sub):
//
//	 cpg-root (Basic)
//	  /   |  \
//	pg1  pg2  cpg-sub (Gang minGroupCount = 2)
//	(S)  (S)   /     \
//	        sub-pg1  sub-pg2 (Basic)
//	        (S)      (S)
//
// (S) = Successfully scheduled
func TestCPGDynamicChildAddition(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	node := st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "20"}).Obj()

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-dynamic-child",
		scheduler.WithPodMaxBackoffSeconds(1),
		scheduler.WithPodInitialBackoffSeconds(1))
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	workload := st.MakeWorkload().Name("workload-cpg-dynamic").
		Children(
			st.MakeCompositePodGroupTemplate().Name("root-t").BasicPolicy().Children(
				st.MakePodGroupTemplate().Name("basic-t").BasicPolicy(),
				st.MakeCompositePodGroupTemplate().Name("sub-cpg-t").MinGroupCount(2).Children(
					st.MakePodGroupTemplate().Name("sub-pg-t").BasicPolicy(),
				),
			),
		).Obj()

	if _, err := cs.SchedulingV1alpha3().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	rootCPG := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-root").WorkloadRef("workload-cpg-dynamic", "root-t").BasicPolicy().Obj()
	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, rootCPG, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create root CPG: %v", err)
	}

	createPG := func(name string, template string, parentCPG string) {
		pg := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingapi.PodGroupSpec{
				WorkloadRef:                 &schedulingapi.WorkloadReference{WorkloadName: "workload-cpg-dynamic", TemplateName: template},
				SchedulingPolicy:            schedulingapi.PodGroupSchedulingPolicy{Basic: &schedulingapi.BasicSchedulingPolicy{}},
				ParentCompositePodGroupName: ptr.To(parentCPG),
			},
		}
		if _, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create PG %s: %v", name, err)
		}
	}

	createPods := func(pgName string, count int, reqCPU string) {
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

			if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create pod %s-pod-%d: %v", pgName, i, err)
			}
		}
	}

	// 1. Create first pg and its pods.
	createPG("pg1", "basic-t", "cpg-root")
	createPods("pg1", 2, "1")

	// Verify pg1 pods scheduled.
	pg1Pods := []string{"pg1-pod-0", "pg1-pod-1"}
	for _, podName := range pg1Pods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	// 2. Dynamically create second pg and its pods.
	createPG("pg2", "basic-t", "cpg-root")
	createPods("pg2", 2, "1")

	// Verify pg2 pods scheduled.
	pg2Pods := []string{"pg2-pod-0", "pg2-pod-1"}
	for _, podName := range pg2Pods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}

	// 3. Dynamically create a cpg node (cpg-sub) under cpg-root with gang scheduling policy (minGroupCount=2).
	cpgSub := st.MakeCompositePodGroup().Namespace(ns).Name("cpg-sub").WorkloadRef("workload-cpg-dynamic", "sub-cpg-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root").Obj()

	if _, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(testCtx.Ctx, cpgSub, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create sub CPG: %v", err)
	}

	// Add sub-pg1 under cpg-sub.
	createPG("sub-pg1", "sub-pg-t", "cpg-sub")
	createPods("sub-pg1", 2, "1")

	// Since cpg-sub has Gang policy with minGroupCount=2, sub-pg1 pods should not schedule yet.
	time.Sleep(2 * time.Second)
	subPg1Pods := []string{"sub-pg1-pod-0", "sub-pg1-pod-1"}
	for _, podName := range subPg1Pods {
		pod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get pod %s: %v", podName, err)
		}
		if pod.Spec.NodeName != "" {
			t.Errorf("Pod %s should not be scheduled before quorum is met, but scheduled to %s", podName, pod.Spec.NodeName)
		}
	}

	// Add sub-pg2 under cpg-sub to satisfy the minGroupCount quorum.
	createPG("sub-pg2", "sub-pg-t", "cpg-sub")
	createPods("sub-pg2", 2, "1")

	// Both sub-pg1 and sub-pg2 pods should schedule successfully now.
	allSubPods := []string{
		"sub-pg1-pod-0", "sub-pg1-pod-1",
		"sub-pg2-pod-0", "sub-pg2-pod-1",
	}
	for _, podName := range allSubPods {
		err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 30*time.Second, false,
			testutils.PodScheduled(cs, ns, podName))
		if err != nil {
			t.Errorf("Failed to wait for pod %s to be scheduled: %v", podName, err)
		}
	}
}
