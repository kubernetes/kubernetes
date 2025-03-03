/*
Copyright 2017 The Kubernetes Authors.

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

package scoring

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"
)

// imported from testutils
var (
	runPausePod                  = testutils.RunPausePod
	createAndWaitForNodesInCache = testutils.CreateAndWaitForNodesInCache
	createNode                   = testutils.CreateNode
	createNamespacesWithLabels   = testutils.CreateNamespacesWithLabels
	runPodWithContainers         = testutils.RunPodWithContainers
	initPausePod                 = testutils.InitPausePod
	initPodWithContainers        = testutils.InitPodWithContainers
	podScheduledIn               = testutils.PodScheduledIn
	podUnschedulable             = testutils.PodUnschedulable
)

var (
	hardSpread   = v1.DoNotSchedule
	softSpread   = v1.ScheduleAnyway
	ignorePolicy = v1.NodeInclusionPolicyIgnore
	honorPolicy  = v1.NodeInclusionPolicyHonor
	taints       = []v1.Taint{{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectPreferNoSchedule}}
)

const (
	resourceGPU  = "example.com/gpu"
	pollInterval = 100 * time.Millisecond
)

// initTestSchedulerForScoringTests initializes the test environment for scheduler scoring function tests.
// It configures a scheduler configuration, enabling the specified prescore and score plugins,
// while disabling all other plugins.
// This setup ensures that only the desired plugins are active during the integration test.
func initTestSchedulerForScoringTests(t *testing.T, preScorePluginName, scorePluginName string) *testutils.TestContext {
	cc := configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				PreScore: configv1.PluginSet{
					Disabled: []configv1.Plugin{
						{Name: "*"},
					},
				},
				Score: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: scorePluginName, Weight: ptr.To[int32](1)},
					},
					Disabled: []configv1.Plugin{
						{Name: "*"},
					},
				},
			},
		}},
	}
	if preScorePluginName != "" {
		cc.Profiles[0].Plugins.PreScore.Enabled = append(cc.Profiles[0].Plugins.PreScore.Enabled, configv1.Plugin{Name: preScorePluginName})
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, cc)
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, strings.ToLower(scorePluginName), nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

func initTestSchedulerForNodeResourcesTest(t *testing.T, strategy configv1.ScoringStrategyType) *testutils.TestContext {
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{
			{
				SchedulerName: ptr.To(v1.DefaultSchedulerName),
				PluginConfig: []configv1.PluginConfig{
					{
						Name: noderesources.Name,
						Args: runtime.RawExtension{Object: &configv1.NodeResourcesFitArgs{
							ScoringStrategy: &configv1.ScoringStrategy{
								Type: strategy,
								Resources: []configv1.ResourceSpec{
									{Name: string(v1.ResourceCPU), Weight: 1},
									{Name: string(v1.ResourceMemory), Weight: 1},
									{Name: resourceGPU, Weight: 2}},
							},
						}},
					},
				},
			},
		},
	})
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, strings.ToLower(noderesources.Name), nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

// TestNodeResourcesScoring verifies that scheduler's node resources priority function
// works correctly.
func TestNodeResourcesScoring(t *testing.T) {
	tests := []struct {
		name         string
		pod          func(testCtx *testutils.TestContext) *v1.Pod
		existingPods func(testCtx *testutils.TestContext) []*v1.Pod
		nodes        []*v1.Node
		strategy     configv1.ScoringStrategyType
		// expectedNodeName is the list of node names. The pod should be scheduled on either of them.
		expectedNodeName []string
	}{
		{
			name: "with least allocated strategy, take existing sidecars into consideration",
			pod: func(testCtx *testutils.TestContext) *v1.Pod {
				return st.MakePod().Namespace(testCtx.NS.Name).Name("pod").
					Res(map[v1.ResourceName]string{
						v1.ResourceCPU:    "2",
						v1.ResourceMemory: "4G",
						resourceGPU:       "1",
					}).Obj()
			},
			existingPods: func(testCtx *testutils.TestContext) []*v1.Pod {
				return []*v1.Pod{
					st.MakePod().Namespace(testCtx.NS.Name).Name("existing-pod-1").Node("node-1").
						Res(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "4G",
							resourceGPU:       "1",
						}).
						SidecarReq(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "2G",
						}).
						Obj(),
					st.MakePod().Namespace(testCtx.NS.Name).Name("existing-pod-2").Node("node-2").
						Res(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "4G",
							resourceGPU:       "1",
						}).Obj(),
				}
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Capacity(
					map[v1.ResourceName]string{
						v1.ResourceCPU:    "8",
						v1.ResourceMemory: "16G",
						resourceGPU:       "4",
					}).Obj(),
				st.MakeNode().Name("node-2").Capacity(
					map[v1.ResourceName]string{
						v1.ResourceCPU:    "8",
						v1.ResourceMemory: "16G",
						resourceGPU:       "4",
					}).Obj(),
			},
			strategy:         configv1.LeastAllocated,
			expectedNodeName: []string{"node-2"},
		},
		{
			name: "with most allocated strategy, take existing sidecars into consideration",
			pod: func(testCtx *testutils.TestContext) *v1.Pod {
				return st.MakePod().Namespace(testCtx.NS.Name).Name("pod").
					Res(map[v1.ResourceName]string{
						v1.ResourceCPU:    "2",
						v1.ResourceMemory: "4G",
						resourceGPU:       "1",
					}).Obj()
			},
			existingPods: func(testCtx *testutils.TestContext) []*v1.Pod {
				return []*v1.Pod{
					st.MakePod().Namespace(testCtx.NS.Name).Name("existing-pod-1").Node("node-1").
						Res(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "4G",
							resourceGPU:       "1",
						}).
						SidecarReq(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "2G",
						}).
						Obj(),
					st.MakePod().Namespace(testCtx.NS.Name).Name("existing-pod-2").Node("node-2").
						Res(map[v1.ResourceName]string{
							v1.ResourceCPU:    "2",
							v1.ResourceMemory: "4G",
							resourceGPU:       "1",
						}).Obj(),
				}
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Capacity(
					map[v1.ResourceName]string{
						v1.ResourceCPU:    "8",
						v1.ResourceMemory: "16G",
						resourceGPU:       "4",
					}).Obj(),
				st.MakeNode().Name("node-2").Capacity(
					map[v1.ResourceName]string{
						v1.ResourceCPU:    "8",
						v1.ResourceMemory: "16G",
						resourceGPU:       "4",
					}).Obj(),
			},
			strategy:         configv1.MostAllocated,
			expectedNodeName: []string{"node-1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := initTestSchedulerForNodeResourcesTest(t, tt.strategy)

			for _, n := range tt.nodes {
				if _, err := createNode(testCtx.ClientSet, n); err != nil {
					t.Fatalf("failed to create node: %v", err)
				}
			}

			if err := testutils.WaitForNodesInCache(testCtx.Ctx, testCtx.Scheduler, len(tt.nodes)); err != nil {
				t.Fatalf("failed to wait for nodes in cache: %v", err)
			}

			if tt.existingPods != nil {
				for _, p := range tt.existingPods(testCtx) {
					if _, err := runPausePod(testCtx.ClientSet, p); err != nil {
						t.Fatalf("failed to create existing pod: %v", err)
					}
				}
			}

			pod, err := runPausePod(testCtx.ClientSet, tt.pod(testCtx))
			if err != nil {
				t.Fatalf("Error running pause pod: %v", err)
			}

			err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false, podScheduledIn(testCtx.ClientSet, pod.Namespace, pod.Name, tt.expectedNodeName))
			if err != nil {
				t.Errorf("Error while trying to wait for a pod to be scheduled: %v", err)
			}
		})
	}
}

// TestNodeAffinityScoring verifies that scheduler's node affinity priority function
// works correctly.
func TestNodeAffinityScoring(t *testing.T) {
	testCtx := initTestSchedulerForScoringTests(t, nodeaffinity.Name, nodeaffinity.Name)
	// Add a few nodes.
	_, err := createAndWaitForNodesInCache(testCtx, "testnode", st.MakeNode(), 4)
	if err != nil {
		t.Fatal(err)
	}
	// Add a label to one of the nodes.
	labelKey := "kubernetes.io/node-topologyKey"
	labelValue := "topologyvalue"
	labeledNode, err := createNode(testCtx.ClientSet, st.MakeNode().Name("testnode-4").Label(labelKey, labelValue).Obj())
	if err != nil {
		t.Fatalf("Cannot create labeled node: %v", err)
	}

	// Create a pod with node affinity.
	podName := "pod-with-node-affinity"
	pod, err := runPausePod(testCtx.ClientSet, initPausePod(&testutils.PausePodConfig{
		Name:      podName,
		Namespace: testCtx.NS.Name,
		Affinity: &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
					{
						Preference: v1.NodeSelectorTerm{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      labelKey,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{labelValue},
								},
							},
						},
						Weight: 20,
					},
				},
			},
		},
	}))
	if err != nil {
		t.Fatalf("Error running pause pod: %v", err)
	}
	if pod.Spec.NodeName != labeledNode.Name {
		t.Errorf("Pod %v got scheduled on an unexpected node: %v. Expected node: %v.", podName, pod.Spec.NodeName, labeledNode.Name)
	} else {
		t.Logf("Pod %v got successfully scheduled on node %v.", podName, pod.Spec.NodeName)
	}
}

// TestPodAffinityScoring verifies that scheduler's pod affinity priority function
// works correctly.
func TestPodAffinityScoring(t *testing.T) {
	labelKey := "service"
	labelValue := "S1"
	topologyKey := "node-topologykey"
	topologyValues := []string{}
	for i := 0; i < 5; i++ {
		topologyValues = append(topologyValues, fmt.Sprintf("topologyvalue%d", i))
	}
	tests := []struct {
		name         string
		pod          *testutils.PausePodConfig
		existingPods []*testutils.PausePodConfig
		nodes        []*v1.Node
		// expectedNodeName is the list of node names. The pod should be scheduled on either of them.
		expectedNodeName               []string
		enableMatchLabelKeysInAffinity bool
	}{
		{
			name: "pod affinity",
			pod: &testutils.PausePodConfig{
				Name:      "pod1",
				Namespace: "ns1",
				Affinity: &v1.Affinity{
					PodAffinity: &v1.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: v1.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      labelKey,
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{labelValue, "S3"},
											},
										},
									},
									TopologyKey: topologyKey,
								},
								Weight: 50,
							},
						},
					},
				},
			},
			existingPods: []*testutils.PausePodConfig{
				{
					Name:      "attractor-pod",
					Namespace: "ns1",
					Labels:    map[string]string{labelKey: labelValue},
					NodeName:  "node1",
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node3").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node4").Label(topologyKey, topologyValues[3]).Obj(),
				st.MakeNode().Name("node5").Label(topologyKey, topologyValues[4]).Obj(),
				st.MakeNode().Name("node6").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node7").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node8").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node9").Label(topologyKey, topologyValues[3]).Obj(),
				st.MakeNode().Name("node10").Label(topologyKey, topologyValues[4]).Obj(),
				st.MakeNode().Name("other-node1").Obj(),
				st.MakeNode().Name("other-node2").Obj(),
			},
			expectedNodeName: []string{"node1", "node6"},
		},
		{
			name: "pod affinity with namespace selector",
			pod: &testutils.PausePodConfig{
				Name:      "pod1",
				Namespace: "ns2",
				Affinity: &v1.Affinity{
					PodAffinity: &v1.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: v1.PodAffinityTerm{
									NamespaceSelector: &metav1.LabelSelector{}, // all namespaces
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      labelKey,
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{labelValue, "S3"},
											},
										},
									},
									TopologyKey: topologyKey,
								},
								Weight: 50,
							},
						},
					},
				},
			},
			existingPods: []*testutils.PausePodConfig{
				{
					Name:      "attractor-pod",
					Namespace: "ns1",
					Labels:    map[string]string{labelKey: labelValue},
					NodeName:  "node1",
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node3").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node4").Label(topologyKey, topologyValues[3]).Obj(),
				st.MakeNode().Name("node5").Label(topologyKey, topologyValues[4]).Obj(),
				st.MakeNode().Name("node6").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node7").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node8").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node9").Label(topologyKey, topologyValues[3]).Obj(),
				st.MakeNode().Name("node10").Label(topologyKey, topologyValues[4]).Obj(),
				st.MakeNode().Name("other-node1").Obj(),
				st.MakeNode().Name("other-node2").Obj(),
			},
			expectedNodeName: []string{"node1", "node6"},
		},
		{
			name: "anti affinity: matchLabelKeys is merged into LabelSelector with In operator (feature flag: enabled)",
			pod: &testutils.PausePodConfig{
				Name:      "incoming",
				Namespace: "ns1",
				Labels:    map[string]string{"foo": "", "bar": "a"},
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpExists,
											},
										},
									},
									MatchLabelKeys: []string{"bar"},
								},
								Weight: 50,
							},
						},
					},
				},
			},
			existingPods: []*testutils.PausePodConfig{
				// It matches the incoming Pod's anti affinity's labelSelector.
				// BUT, the matchLabelKeys make the existing Pod's anti affinity's labelSelector not match with this label.
				{
					NodeName:  "node1",
					Name:      "pod1",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "fuga"},
				},
				// It matches the incoming Pod's anti affinity.
				{
					NodeName:  "node2",
					Name:      "pod2",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "a"},
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
			},
			expectedNodeName:               []string{"node1"},
			enableMatchLabelKeysInAffinity: true,
		},
		{
			name: "anti affinity: mismatchLabelKeys is merged into LabelSelector with NotIn operator  (feature flag: enabled)",
			pod: &testutils.PausePodConfig{
				Name:      "incoming",
				Namespace: "ns1",
				Labels:    map[string]string{"foo": "", "bar": "a"},
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpExists,
											},
										},
									},
									MismatchLabelKeys: []string{"bar"},
								},
								Weight: 50,
							},
						},
					},
				},
			},
			existingPods: []*testutils.PausePodConfig{
				// It matches the incoming Pod's anti affinity's labelSelector.
				{
					NodeName:  "node1",
					Name:      "pod1",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "fuga"},
				},
				// It matches the incoming Pod's affinity.
				// But, the mismatchLabelKeys make the existing Pod's anti affinity's labelSelector not match with this label.
				{
					NodeName:  "node2",
					Name:      "pod2",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "a"},
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
			},
			expectedNodeName:               []string{"node2"},
			enableMatchLabelKeysInAffinity: true,
		},
		{
			name: "affinity: matchLabelKeys is merged into LabelSelector with In operator (feature flag: enabled)",
			pod: &testutils.PausePodConfig{
				Affinity: &v1.Affinity{
					PodAffinity: &v1.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								// affinity with pod3.
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpExists,
											},
										},
									},
									MatchLabelKeys: []string{"bar"},
								},
								Weight: 50,
							},
							{
								// affinity with pod1 and pod2.
								// schedule this Pod by this weaker affinity
								// if `matchLabelKeys` above isn't working correctly.
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "bar",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"hoge"},
											},
										},
									},
								},
								Weight: 10,
							},
						},
					},
				},
				Name:      "incoming",
				Namespace: "ns1",
				Labels:    map[string]string{"foo": "", "bar": "a"},
			},
			existingPods: []*testutils.PausePodConfig{
				{
					NodeName:  "node1",
					Name:      "pod1",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "hoge"},
				},
				{
					NodeName:  "node2",
					Name:      "pod2",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "hoge"},
				},
				{
					NodeName:  "node3",
					Name:      "pod3",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "a"},
				},
			},
			enableMatchLabelKeysInAffinity: true,
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node3").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node4").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node5").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node6").Label(topologyKey, topologyValues[2]).Obj(),
			},
			expectedNodeName: []string{"node3", "node6"},
		},
		{
			name: "affinity: mismatchLabelKeys is merged into LabelSelector with NotIn operator (feature flag: enabled)",
			pod: &testutils.PausePodConfig{
				Affinity: &v1.Affinity{
					PodAffinity: &v1.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
							{
								// affinity with pod3.
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpExists,
											},
										},
									},
									MismatchLabelKeys: []string{"bar"},
								},
								Weight: 50,
							},
							{
								// affinity with pod1 and pod2.
								// schedule this Pod by this weaker affinity
								// if `matchLabelKeys` above isn't working correctly.
								PodAffinityTerm: v1.PodAffinityTerm{
									TopologyKey: topologyKey,
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "bar",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"hoge"},
											},
										},
									},
								},
								Weight: 10,
							},
						},
					},
				},
				Name:      "incoming",
				Namespace: "ns1",
				Labels:    map[string]string{"foo": "", "bar": "a"},
			},
			existingPods: []*testutils.PausePodConfig{
				{
					NodeName:  "node1",
					Name:      "pod1",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "a"},
				},
				{
					NodeName:  "node2",
					Name:      "pod2",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "a"},
				},
				{
					NodeName:  "node3",
					Name:      "pod3",
					Namespace: "ns1",
					Labels:    map[string]string{"foo": "", "bar": "hoge"},
				},
			},
			enableMatchLabelKeysInAffinity: true,
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node2").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node3").Label(topologyKey, topologyValues[2]).Obj(),
				st.MakeNode().Name("node4").Label(topologyKey, topologyValues[0]).Obj(),
				st.MakeNode().Name("node5").Label(topologyKey, topologyValues[1]).Obj(),
				st.MakeNode().Name("node6").Label(topologyKey, topologyValues[2]).Obj(),
			},
			expectedNodeName: []string{"node3", "node6"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodAffinity, tt.enableMatchLabelKeysInAffinity)

			testCtx := initTestSchedulerForScoringTests(t, interpodaffinity.Name, interpodaffinity.Name)
			if err := createNamespacesWithLabels(testCtx.ClientSet, []string{"ns1", "ns2"}, map[string]string{"team": "team1"}); err != nil {
				t.Fatal(err)
			}

			for _, n := range tt.nodes {
				if _, err := createNode(testCtx.ClientSet, n); err != nil {
					t.Fatalf("failed to create node: %v", err)
				}
			}

			for _, p := range tt.existingPods {
				if _, err := runPausePod(testCtx.ClientSet, initPausePod(p)); err != nil {
					t.Fatalf("failed to create existing pod: %v", err)
				}
			}

			pod, err := runPausePod(testCtx.ClientSet, initPausePod(tt.pod))
			if err != nil {
				t.Fatalf("Error running pause pod: %v", err)
			}

			err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false, podScheduledIn(testCtx.ClientSet, pod.Namespace, pod.Name, tt.expectedNodeName))
			if err != nil {
				t.Errorf("Error while trying to wait for a pod to be scheduled: %v", err)
			}
		})
	}
}

func TestTaintTolerationScoring(t *testing.T) {
	tests := []struct {
		name           string
		podTolerations []v1.Toleration
		nodes          []*v1.Node
		// expectedNodesName is a set of nodes that the pod should potentially be scheduled on.
		// It is used to verify that the pod is scheduled on the expected nodes.
		expectedNodesName sets.Set[string]
	}{
		{
			name:           "no taints or tolerations",
			podTolerations: []v1.Toleration{},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Obj(),
				st.MakeNode().Name("node-2").Obj(),
			},
			expectedNodesName: sets.New("node-1", "node-2"),
		},
		{
			name: "pod with toleration for node's taint",
			podTolerations: []v1.Toleration{
				{
					Key:      "example-key",
					Operator: v1.TolerationOpEqual,
					Value:    "example-value",
					Effect:   v1.TaintEffectPreferNoSchedule,
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").
					Taints([]v1.Taint{
						{
							Key:    "example-key",
							Value:  "example-value",
							Effect: v1.TaintEffectPreferNoSchedule,
						},
					}).Obj(),
				st.MakeNode().Name("node-2").Obj(),
			},
			expectedNodesName: sets.New("node-1", "node-2"),
		},
		{
			name: "pod without toleration for node's taint",
			podTolerations: []v1.Toleration{
				{
					Key:      "other-key",
					Operator: v1.TolerationOpEqual,
					Value:    "other-value",
					Effect:   v1.TaintEffectPreferNoSchedule,
				},
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").
					Taints([]v1.Taint{
						{
							Key:    "example-key",
							Value:  "example-value",
							Effect: v1.TaintEffectPreferNoSchedule,
						},
					}).Obj(),
				st.MakeNode().Name("node-2").Obj(),
			},
			expectedNodesName: sets.New("node-2"),
		},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := initTestSchedulerForScoringTests(t, tainttoleration.Name, tainttoleration.Name)

			for _, n := range tt.nodes {
				if _, err := createNode(testCtx.ClientSet, n); err != nil {
					t.Fatalf("Failed to create node: %v", err)
				}
			}
			pod, err := runPausePod(testCtx.ClientSet, initPausePod(&testutils.PausePodConfig{
				Name:        fmt.Sprintf("test-pod-%v", i),
				Namespace:   testCtx.NS.Name,
				Tolerations: tt.podTolerations,
			}))
			if err != nil {
				t.Fatalf("Error running pause pod: %v", err)
			}
			defer testutils.CleanupPods(testCtx.Ctx, testCtx.ClientSet, t, []*v1.Pod{pod})
			if !tt.expectedNodesName.Has(pod.Spec.NodeName) {
				t.Errorf("Pod was not scheduled to expected node: %v", pod.Spec.NodeName)
			}
		})
	}
}

// TestImageLocalityScoring verifies that the scheduler's image locality priority function
// works correctly, i.e., the pod gets scheduled to the node where its container images are ready.
func TestImageLocalityScoring(t *testing.T) {
	testCtx := initTestSchedulerForScoringTests(t, "", imagelocality.Name)

	// Create a node with the large image.
	// We use a fake large image as the test image used by the pod, which has
	// relatively large image size.
	imageName := "fake-large-image:v1"
	nodeWithLargeImage, err := createNode(
		testCtx.ClientSet,
		st.MakeNode().Name("testnode-large-image").Images(map[string]int64{imageName: 3000 * 1024 * 1024}).Obj(),
	)
	if err != nil {
		t.Fatalf("cannot create node with a large image: %v", err)
	}

	// Add a few nodes.
	_, err = createAndWaitForNodesInCache(testCtx, "testnode", st.MakeNode(), 10)
	if err != nil {
		t.Fatal(err)
	}

	// Create a pod with containers each having the specified image.
	podName := "pod-using-large-image"
	pod, err := runPodWithContainers(testCtx.ClientSet, initPodWithContainers(testCtx.ClientSet, &testutils.PodWithContainersConfig{
		Name:       podName,
		Namespace:  testCtx.NS.Name,
		Containers: makeContainersWithImages([]string{imageName}),
	}))
	if err != nil {
		t.Fatalf("error running pod with images: %v", err)
	}
	if pod.Spec.NodeName != nodeWithLargeImage.Name {
		t.Errorf("pod %v got scheduled on an unexpected node: %v. Expected node: %v.", podName, pod.Spec.NodeName, nodeWithLargeImage.Name)
	} else {
		t.Logf("pod %v got successfully scheduled on node %v.", podName, pod.Spec.NodeName)
	}
}

// makeContainerWithImage returns a list of v1.Container objects for each given image. Duplicates of an image are ignored,
// i.e., each image is used only once.
func makeContainersWithImages(images []string) []v1.Container {
	var containers []v1.Container
	usedImages := make(map[string]struct{})

	for _, image := range images {
		if _, ok := usedImages[image]; !ok {
			containers = append(containers, v1.Container{
				Name:  strings.Replace(image, ":", "-", -1) + "-container",
				Image: image,
			})
			usedImages[image] = struct{}{}
		}
	}
	return containers
}

// TestPodTopologySpreadScoring verifies that the PodTopologySpread Score plugin works.
func TestPodTopologySpreadScoring(t *testing.T) {
	pause := imageutils.GetPauseImageName()
	taint := v1.Taint{
		Key:    "k1",
		Value:  "v1",
		Effect: v1.TaintEffectNoSchedule,
	}

	//  default nodes with labels "zone: zone-{0,1}" and "node: <node name>".
	defaultNodes := []*v1.Node{
		st.MakeNode().Name("node-0").Label("node", "node-0").Label("zone", "zone-0").Taints([]v1.Taint{taint}).Obj(),
		st.MakeNode().Name("node-1").Label("node", "node-1").Label("zone", "zone-0").Obj(),
		st.MakeNode().Name("node-2").Label("node", "node-2").Label("zone", "zone-1").Obj(),
		st.MakeNode().Name("node-3").Label("node", "node-3").Label("zone", "zone-1").Obj(),
	}

	tests := []struct {
		name                      string
		incomingPod               *v1.Pod
		existingPods              []*v1.Pod
		fits                      bool
		nodes                     []*v1.Node
		want                      []string // nodes expected to schedule onto
		enableNodeInclusionPolicy bool
		enableMatchLabelKeys      bool
	}{
		// note: naming starts at index 0
		// the symbol ~X~ means that node is infeasible
		{
			name: "place pod on a ~0~/1/2/3 cluster with MaxSkew=1, node-1 is the preferred fit",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3c").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:  true,
			nodes: defaultNodes,
			want:  []string{"node-1"},
		},
		{
			name: "combined with hardSpread constraint on a ~4~/0/1/2 cluster",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p0a").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p0b").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p0c").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p0d").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:  true,
			nodes: defaultNodes,
			want:  []string{"node-2"},
		},
		{
			// 1. to fulfil "zone" constraint, pods spread across zones as ~3~/0
			// 2. to fulfil "node" constraint, pods spread across zones as 1/~2~/0/~0~
			// node-2 and node 4 are filtered out by plugins
			name: "soft constraint with two node inclusion Constraints, zone: honor/ignore, node: honor/ignore",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				NodeSelector(map[string]string{"foo": ""}).
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1a").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p4a").Node("node-4").Label("foo", "").Container(pause).Obj(),
			},
			fits: true,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Label("node", "node-1").Label("zone", "zone-1").Label("foo", "").Obj(),
				st.MakeNode().Name("node-2").Label("node", "node-2").Label("zone", "zone-1").Label("foo", "").Taints(taints).Obj(),
				st.MakeNode().Name("node-3").Label("node", "node-3").Label("zone", "zone-2").Label("foo", "").Obj(),
				st.MakeNode().Name("node-4").Label("node", "node-4").Label("zone", "zone-2").Obj(),
			},
			want:                      []string{"node-3"},
			enableNodeInclusionPolicy: true,
		},
		{
			// 1. to fulfil "zone" constraint, pods spread across zones as ~3~/~1~
			// 2. to fulfil "node" constraint, pods spread across zones as 1/~0~/0/~0~
			// node-2 and node 4 are filtered out by plugins
			name: "soft constraint with two node inclusion Constraints, zone: ignore/ignore, node: honor/honor",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				NodeSelector(map[string]string{"foo": ""}).
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, &ignorePolicy, nil, nil).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, &honorPolicy, nil).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1a").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p4a").Node("node-4").Label("foo", "").Container(pause).Obj(),
			},
			fits: true,
			nodes: []*v1.Node{
				st.MakeNode().Name("node-1").Label("node", "node-1").Label("zone", "zone-1").Label("foo", "").Obj(),
				st.MakeNode().Name("node-2").Label("node", "node-2").Label("zone", "zone-1").Label("foo", "").Taints(taints).Obj(),
				st.MakeNode().Name("node-3").Label("node", "node-3").Label("zone", "zone-2").Label("foo", "").Obj(),
				st.MakeNode().Name("node-4").Label("node", "node-4").Label("zone", "zone-2").Obj(),
			},
			want:                      []string{"node-3"},
			enableNodeInclusionPolicy: true,
		},
		{
			name: "matchLabelKeys ignored when feature gate disabled, node-1 is the preferred fit",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, []string{"bar"}).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node-1").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3a").Node("node-3").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p3b").Node("node-3").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p3c").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:                 true,
			nodes:                defaultNodes,
			want:                 []string{"node-1"},
			enableMatchLabelKeys: false,
		},
		{
			name: "matchLabelKeys ANDed with LabelSelector when LabelSelector isn't empty, node-2 is the preferred fit",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Label("bar", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, []string{"bar"}).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node-1").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3a").Node("node-3").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p3b").Node("node-3").Label("foo", "").Label("bar", "").Container(pause).Obj(),
				st.MakePod().Name("p3c").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:                 true,
			nodes:                defaultNodes,
			want:                 []string{"node-2"},
			enableMatchLabelKeys: true,
		},
		{
			name: "matchLabelKeys ANDed with LabelSelector when LabelSelector is empty, node-1 is the preferred fit",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Obj(), nil, nil, nil, []string{"foo"}).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Name("p3c").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:                 true,
			nodes:                defaultNodes,
			want:                 []string{"node-1"},
			enableMatchLabelKeys: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeInclusionPolicyInPodTopologySpread, tt.enableNodeInclusionPolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpread, tt.enableMatchLabelKeys)

			testCtx := initTestSchedulerForScoringTests(t, podtopologyspread.Name, podtopologyspread.Name)
			cs := testCtx.ClientSet
			ns := testCtx.NS.Name

			for i := range tt.nodes {
				if _, err := createNode(cs, tt.nodes[i]); err != nil {
					t.Fatalf("Cannot create node: %v", err)
				}
			}

			// set namespace to pods
			for i := range tt.existingPods {
				tt.existingPods[i].SetNamespace(ns)
			}
			tt.incomingPod.SetNamespace(ns)

			allPods := append(tt.existingPods, tt.incomingPod)
			defer testutils.CleanupPods(testCtx.Ctx, cs, t, allPods)
			for _, pod := range tt.existingPods {
				createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Test Failed: error while creating pod during test: %v", err)
				}
				err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false,
					testutils.PodScheduled(cs, createdPod.Namespace, createdPod.Name))
				if err != nil {
					t.Errorf("Test Failed: error while waiting for pod during test: %v", err)
				}
			}

			testPod, err := cs.CoreV1().Pods(tt.incomingPod.Namespace).Create(testCtx.Ctx, tt.incomingPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Test Failed: error while creating pod during test: %v", err)
			}

			if tt.fits {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false,
					podScheduledIn(cs, testPod.Namespace, testPod.Name, tt.want))
			} else {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false,
					podUnschedulable(cs, testPod.Namespace, testPod.Name))
			}
			if err != nil {
				t.Errorf("Test Failed: %v", err)
			}
		})
	}
}

// TestDefaultPodTopologySpreadScoring verifies that the PodTopologySpread Score plugin
// with the system default spreading spreads Pods belonging to a Service.
// The setup has 300 nodes over 3 zones.
func TestDefaultPodTopologySpreadScoring(t *testing.T) {
	testCtx := initTestSchedulerForScoringTests(t, podtopologyspread.Name, podtopologyspread.Name)
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	zoneForNode := make(map[string]string)
	for i := 0; i < 300; i++ {
		nodeName := fmt.Sprintf("node-%d", i)
		zone := fmt.Sprintf("zone-%d", i%3)
		zoneForNode[nodeName] = zone
		_, err := createNode(cs, st.MakeNode().Name(nodeName).Label(v1.LabelHostname, nodeName).Label(v1.LabelTopologyZone, zone).Obj())
		if err != nil {
			t.Fatalf("Cannot create node: %v", err)
		}
	}

	serviceName := "test-service"
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: ns,
		},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{
				"service": serviceName,
			},
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt32(80),
			}},
		},
	}
	_, err := cs.CoreV1().Services(ns).Create(testCtx.Ctx, svc, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Cannot create Service: %v", err)
	}

	pause := imageutils.GetPauseImageName()
	totalPodCnt := 0
	for _, nPods := range []int{3, 9, 15} {
		// Append nPods each iteration.
		t.Run(fmt.Sprintf("%d-pods", totalPodCnt+nPods), func(t *testing.T) {
			for i := 0; i < nPods; i++ {
				p := st.MakePod().Name(fmt.Sprintf("p-%d", totalPodCnt)).Label("service", serviceName).Container(pause).Obj()
				_, err = cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Cannot create Pod: %v", err)
				}
				totalPodCnt++
			}
			var pods []v1.Pod
			// Wait for all Pods scheduled.
			err = wait.PollUntilContextTimeout(testCtx.Ctx, pollInterval, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
				podList, err := cs.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
				if err != nil {
					t.Fatalf("Cannot list pods to verify scheduling: %v", err)
				}
				for _, p := range podList.Items {
					if p.Spec.NodeName == "" {
						return false, nil
					}
				}
				pods = podList.Items
				return true, nil
			})
			// Verify zone spreading.
			zoneCnts := make(map[string]int)
			for _, p := range pods {
				zoneCnts[zoneForNode[p.Spec.NodeName]]++
			}
			maxCnt := 0
			minCnt := len(pods)
			for _, c := range zoneCnts {
				if c > maxCnt {
					maxCnt = c
				}
				if c < minCnt {
					minCnt = c
				}
			}
			if skew := maxCnt - minCnt; skew != 0 {
				t.Errorf("Zone skew is %d, should be 0", skew)
			}
		})
	}
}
