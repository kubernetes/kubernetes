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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-scheduler/config/v1beta3"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"
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

// This file tests the scheduler priority functions.
func initTestSchedulerForPriorityTest(t *testing.T, scorePluginName string) *testutils.TestContext {
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{{
			SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			Plugins: &v1beta3.Plugins{
				Score: v1beta3.PluginSet{
					Enabled: []v1beta3.Plugin{
						{Name: scorePluginName, Weight: pointer.Int32Ptr(1)},
					},
					Disabled: []v1beta3.Plugin{
						{Name: "*"},
					},
				},
			},
		}},
	})
	testCtx := testutils.InitTestSchedulerWithOptions(
		t,
		testutils.InitTestAPIServer(t, strings.ToLower(scorePluginName), nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

func initTestSchedulerForNodeResourcesTest(t *testing.T) *testutils.TestContext {
	cfg := configtesting.V1beta3ToInternalWithDefaults(t, v1beta3.KubeSchedulerConfiguration{
		Profiles: []v1beta3.KubeSchedulerProfile{
			{
				SchedulerName: pointer.StringPtr(v1.DefaultSchedulerName),
			},
			{
				SchedulerName: pointer.StringPtr("gpu-binpacking-scheduler"),
				PluginConfig: []v1beta3.PluginConfig{
					{
						Name: noderesources.Name,
						Args: runtime.RawExtension{Object: &v1beta3.NodeResourcesFitArgs{
							ScoringStrategy: &v1beta3.ScoringStrategy{
								Type: v1beta3.MostAllocated,
								Resources: []v1beta3.ResourceSpec{
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
	testutils.SyncInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)
	return testCtx
}

// TestNodeResourcesScoring verifies that scheduler's node resources priority function
// works correctly.
func TestNodeResourcesScoring(t *testing.T) {
	testCtx := initTestSchedulerForNodeResourcesTest(t)
	defer testutils.CleanupTest(t, testCtx)
	// Add a few nodes.
	_, err := createAndWaitForNodesInCache(testCtx, "testnode", st.MakeNode().Capacity(
		map[v1.ResourceName]string{
			v1.ResourceCPU:    "8",
			v1.ResourceMemory: "16G",
			resourceGPU:       "4",
		}), 2)
	if err != nil {
		t.Fatal(err)
	}
	cpuBoundPod1, err := runPausePod(testCtx.ClientSet, st.MakePod().Namespace(testCtx.NS.Name).Name("cpubound1").Req(
		map[v1.ResourceName]string{
			v1.ResourceCPU:    "2",
			v1.ResourceMemory: "4G",
			resourceGPU:       "1",
		},
	).Obj())
	if err != nil {
		t.Fatal(err)
	}
	gpuBoundPod1, err := runPausePod(testCtx.ClientSet, st.MakePod().Namespace(testCtx.NS.Name).Name("gpubound1").Req(
		map[v1.ResourceName]string{
			v1.ResourceCPU:    "1",
			v1.ResourceMemory: "2G",
			resourceGPU:       "2",
		},
	).Obj())
	if err != nil {
		t.Fatal(err)
	}
	if cpuBoundPod1.Spec.NodeName == "" || gpuBoundPod1.Spec.NodeName == "" {
		t.Fatalf("pods should have nodeName assigned, got %q and %q",
			cpuBoundPod1.Spec.NodeName, gpuBoundPod1.Spec.NodeName)
	}

	// Since both pods used the default scheduler, then they should land on two different
	// nodes because the default configuration uses LeastAllocated.
	if cpuBoundPod1.Spec.NodeName == gpuBoundPod1.Spec.NodeName {
		t.Fatalf("pods should have landed on different nodes, both scheduled on %q",
			cpuBoundPod1.Spec.NodeName)
	}

	// The following pod is using the gpu-binpacking-scheduler profile, which gives a higher weight to
	// GPU-based binpacking, and so it should land on the node with higher GPU utilization.
	cpuBoundPod2, err := runPausePod(testCtx.ClientSet, st.MakePod().Namespace(testCtx.NS.Name).Name("cpubound2").SchedulerName("gpu-binpacking-scheduler").Req(
		map[v1.ResourceName]string{
			v1.ResourceCPU:    "2",
			v1.ResourceMemory: "4G",
			resourceGPU:       "1",
		},
	).Obj())
	if err != nil {
		t.Fatal(err)
	}
	if cpuBoundPod2.Spec.NodeName != gpuBoundPod1.Spec.NodeName {
		t.Errorf("pods should have landed on the same node")
	}
}

// TestNodeAffinityScoring verifies that scheduler's node affinity priority function
// works correctly.
func TestNodeAffinityScoring(t *testing.T) {
	testCtx := initTestSchedulerForPriorityTest(t, nodeaffinity.Name)
	defer testutils.CleanupTest(t, testCtx)
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
	topologyValue := "topologyvalue"
	tests := []struct {
		name      string
		podConfig *testutils.PausePodConfig
	}{
		{
			name: "pod affinity",
			podConfig: &testutils.PausePodConfig{
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
		},
		{
			name: "pod affinity with namespace selector",
			podConfig: &testutils.PausePodConfig{
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
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := initTestSchedulerForPriorityTest(t, interpodaffinity.Name)
			defer testutils.CleanupTest(t, testCtx)
			// Add a few nodes.
			nodesInTopology, err := createAndWaitForNodesInCache(testCtx, "in-topology", st.MakeNode().Label(topologyKey, topologyValue), 5)
			if err != nil {
				t.Fatal(err)
			}
			if err := createNamespacesWithLabels(testCtx.ClientSet, []string{"ns1", "ns2"}, map[string]string{"team": "team1"}); err != nil {
				t.Fatal(err)
			}
			// Add a pod with a label and wait for it to schedule.
			_, err = runPausePod(testCtx.ClientSet, initPausePod(&testutils.PausePodConfig{
				Name:      "attractor-pod",
				Namespace: "ns1",
				Labels:    map[string]string{labelKey: labelValue},
			}))
			if err != nil {
				t.Fatalf("Error running the attractor pod: %v", err)
			}
			// Add a few more nodes without the topology label.
			_, err = createAndWaitForNodesInCache(testCtx, "other-node", st.MakeNode(), 5)
			if err != nil {
				t.Fatal(err)
			}
			// Add a new pod with affinity to the attractor pod.
			pod, err := runPausePod(testCtx.ClientSet, initPausePod(tt.podConfig))
			if err != nil {
				t.Fatalf("Error running pause pod: %v", err)
			}
			// The new pod must be scheduled on one of the nodes with the same topology
			// key-value as the attractor pod.
			for _, node := range nodesInTopology {
				if node.Name == pod.Spec.NodeName {
					t.Logf("Pod %v got successfully scheduled on node %v.", tt.podConfig.Name, pod.Spec.NodeName)
					return
				}
			}
			t.Errorf("Pod %v got scheduled on an unexpected node: %v.", tt.podConfig.Name, pod.Spec.NodeName)
		})
	}
}

// TestImageLocalityScoring verifies that the scheduler's image locality priority function
// works correctly, i.e., the pod gets scheduled to the node where its container images are ready.
func TestImageLocalityScoring(t *testing.T) {
	testCtx := initTestSchedulerForPriorityTest(t, imagelocality.Name)
	defer testutils.CleanupTest(t, testCtx)

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
		name                       string
		incomingPod                *v1.Pod
		existingPods               []*v1.Pod
		fits                       bool
		nodes                      []*v1.Node
		want                       []string // nodes expected to schedule onto
		enableNodeInclustionPolicy bool
	}{
		// note: naming starts at index 0
		// the symbol ~X~ means that node is infeasible
		{
			name: "place pod on a ~0~/1/2/3 cluster with MaxSkew=1, node-1 is the preferred fit",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil).
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
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil).
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil).
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
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil).
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
			want:                       []string{"node-3"},
			enableNodeInclustionPolicy: true,
		},
		{
			// 1. to fulfil "zone" constraint, pods spread across zones as ~3~/~1~
			// 2. to fulfil "node" constraint, pods spread across zones as 1/~0~/0/~0~
			// node-2 and node 4 are filtered out by plugins
			name: "soft constraint with two node inclusion Constraints, zone: ignore/ignore, node: honor/honor",
			incomingPod: st.MakePod().Name("p").Label("foo", "").Container(pause).
				NodeSelector(map[string]string{"foo": ""}).
				SpreadConstraint(1, "zone", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, &ignorePolicy, nil).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, &honorPolicy).
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
			want:                       []string{"node-3"},
			enableNodeInclustionPolicy: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeInclusionPolicyInPodTopologySpread, tt.enableNodeInclustionPolicy)()

			testCtx := initTestSchedulerForPriorityTest(t, podtopologyspread.Name)
			defer testutils.CleanupTest(t, testCtx)
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
			defer testutils.CleanupPods(cs, t, allPods)
			for _, pod := range tt.existingPods {
				createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Test Failed: error while creating pod during test: %v", err)
				}
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, testutils.PodScheduled(cs, createdPod.Namespace, createdPod.Name))
				if err != nil {
					t.Errorf("Test Failed: error while waiting for pod during test: %v", err)
				}
			}

			testPod, err := cs.CoreV1().Pods(tt.incomingPod.Namespace).Create(context.TODO(), tt.incomingPod, metav1.CreateOptions{})
			if err != nil && !apierrors.IsInvalid(err) {
				t.Fatalf("Test Failed: error while creating pod during test: %v", err)
			}

			if tt.fits {
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podScheduledIn(cs, testPod.Namespace, testPod.Name, tt.want))
			} else {
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podUnschedulable(cs, testPod.Namespace, testPod.Name))
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
	testCtx := initTestSchedulerForPriorityTest(t, podtopologyspread.Name)
	t.Cleanup(func() {
		testutils.CleanupTest(t, testCtx)
	})
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
				TargetPort: intstr.FromInt(80),
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
			err = wait.Poll(pollInterval, wait.ForeverTestTimeout, func() (bool, error) {
				podList, err := cs.CoreV1().Pods(ns).List(testCtx.Ctx, metav1.ListOptions{})
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
