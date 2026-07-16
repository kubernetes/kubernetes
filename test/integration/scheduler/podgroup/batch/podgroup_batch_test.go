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

package batch

import (
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	config "k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutil "k8s.io/kubernetes/test/integration/util"
)

type batchGetter interface {
	TotalBatchedPods() int64
}

func newDefaultComponentConfig() (*config.KubeSchedulerConfiguration, error) {
	gvk := kubeschedulerconfigv1.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(nil, &gvk, &cfg)
	if err != nil {
		return nil, err
	}

	// Clear pod topo spread defaults.
	profile := cfg.Profiles[0]
	for _, cfg := range profile.PluginConfig {
		if cfg.Name == names.PodTopologySpread {
			tps := cfg.Args.(*config.PodTopologySpreadArgs)
			tps.DefaultConstraints = []v1.TopologySpreadConstraint{}
			tps.DefaultingType = config.ListDefaulting
		}
	}

	return &cfg, nil
}

func initScheduler(t *testing.T, nsPrefix string) (*testutil.TestContext, batchGetter) {
	cfg, err := newDefaultComponentConfig()
	if err != nil {
		t.Fatalf("Error creating default component config: %v", err)
	}
	testCtx := testutil.InitTestSchedulerWithNS(t, nsPrefix,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0),
	)
	getter := testCtx.Scheduler.Profiles["default-scheduler"].(batchGetter)
	return testCtx, getter
}

func TestPodGroupBatching(t *testing.T) {
	// Enable GenericWorkload feature gate.
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload: true,
	})

	// Start scheduler once for all subtests.
	testCtx, getter := initScheduler(t, "pg-batch-shared")

	// Pre-create shared templates.
	workload := st.MakeWorkload().Name("workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("pg-tmpl").MinCount(2).Obj()).
		Obj()
	pg := st.MakePodGroup().Name("pg1").WorkloadRef("pg-tmpl", "workload").
		Priority(100).MinCount(2).Obj()

	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "StaticSignature",
			steps: []stepsframework.Step{
				{
					Name: "Create Nodes",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("static-node1").Label("forpod", "1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
						st.MakeNode().Name("static-node2").Label("forpod", "1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:            "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{workload},
				},
				{
					Name:           "Create PodGroup",
					CreatePodGroup: pg,
				},
				{
					Name: "Create Pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("p1").PodGroupName("pg1").Priority(100).
							NodeSelector(map[string]string{"forpod": "1"}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
						st.MakePod().Name("p2").PodGroupName("pg1").Priority(100).
							NodeSelector(map[string]string{"forpod": "1"}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
					},
				},
				{
					Name:                 "Wait for pods to schedule",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "DynamicSignatureUpdate",
			steps: []stepsframework.Step{
				{
					Name: "Create Tainted Nodes",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("dynamic-node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).
							Taints([]v1.Taint{{Key: "dedicated", Value: "group1", Effect: v1.TaintEffectNoSchedule}}).Obj(),
						st.MakeNode().Name("dynamic-node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).
							Taints([]v1.Taint{{Key: "dedicated", Value: "group1", Effect: v1.TaintEffectNoSchedule}}).Obj(),
					},
				},
				{
					Name:            "Create Workload",
					CreateWorkloads: []*schedulingapi.Workload{workload},
				},
				{
					Name:           "Create PodGroup",
					CreatePodGroup: pg,
				},
				{
					Name: "Create Pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("p1").PodGroupName("pg1").Priority(100).
							Tolerations([]v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpEqual, Value: "group1", Effect: v1.TaintEffectNoSchedule}}).
							Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
						st.MakePod().Name("p2").PodGroupName("pg1").Priority(100).
							Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
					},
				},
				{
					Name: "Update p2 tolerations",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p2",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.Tolerations = []v1.Toleration{{Key: "dedicated", Operator: v1.TolerationOpEqual, Value: "group1", Effect: v1.TaintEffectNoSchedule}}
						},
					},
				},
				{
					Name:                 "Wait for pods to schedule",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Create a unique namespace for this subtest.
			nsObj := framework.CreateNamespaceOrDie(testCtx.ClientSet, strings.ToLower("subtest-"+tc.name), t)
			ns := nsObj.Name

			// Clean up nodes created in this subtest upon completion.
			t.Cleanup(func() {
				err := testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(testCtx.Ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{})
				if err != nil {
					t.Errorf("Failed to clean up nodes: %v", err)
				}
			})

			// Record baseline batched pods metric
			prevBatched := getter.TotalBatchedPods()

			// Run all steps defined in the test case
			if err := stepsframework.RunSteps(testCtx, t, ns, tc.steps); err != nil {
				t.Fatalf("Steps execution failed: %v", err)
			}

			// Verify that opportunistic batching cache was utilized
			currBatched := getter.TotalBatchedPods()
			if currBatched <= prevBatched {
				t.Errorf("Expected TotalBatchedPods to increase (used opportunistic batching), but remained at %d", currBatched)
			}
		})
	}
}
