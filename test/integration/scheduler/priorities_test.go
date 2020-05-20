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

package scheduler

import (
	"context"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// This file tests the scheduler priority functions.

// TestNodeAffinity verifies that scheduler's node affinity priority function
// works correctly.
func TestNodeAffinity(t *testing.T) {
	testCtx := initTest(t, "node-affinity")
	defer testutils.CleanupTest(t, testCtx)
	// Add a few nodes.
	nodes, err := createNodes(testCtx.ClientSet, "testnode", nil, 5)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	// Add a label to one of the nodes.
	labeledNode := nodes[1]
	labelKey := "kubernetes.io/node-topologyKey"
	labelValue := "topologyvalue"
	labels := map[string]string{
		labelKey: labelValue,
	}
	if err = utils.AddLabelsToNode(testCtx.ClientSet, labeledNode.Name, labels); err != nil {
		t.Fatalf("Cannot add labels to node: %v", err)
	}
	if err = waitForNodeLabels(testCtx.ClientSet, labeledNode.Name, labels); err != nil {
		t.Fatalf("Adding labels to node didn't succeed: %v", err)
	}
	// Create a pod with node affinity.
	podName := "pod-with-node-affinity"
	pod, err := runPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{
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

// TestPodAffinity verifies that scheduler's pod affinity priority function
// works correctly.
func TestPodAffinity(t *testing.T) {
	testCtx := initTest(t, "pod-affinity")
	defer testutils.CleanupTest(t, testCtx)
	// Add a few nodes.
	nodesInTopology, err := createNodes(testCtx.ClientSet, "in-topology", nil, 5)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	topologyKey := "node-topologykey"
	topologyValue := "topologyvalue"
	nodeLabels := map[string]string{
		topologyKey: topologyValue,
	}
	for _, node := range nodesInTopology {
		// Add topology key to all the nodes.
		if err = utils.AddLabelsToNode(testCtx.ClientSet, node.Name, nodeLabels); err != nil {
			t.Fatalf("Cannot add labels to node %v: %v", node.Name, err)
		}
		if err = waitForNodeLabels(testCtx.ClientSet, node.Name, nodeLabels); err != nil {
			t.Fatalf("Adding labels to node %v didn't succeed: %v", node.Name, err)
		}
	}
	// Add a pod with a label and wait for it to schedule.
	labelKey := "service"
	labelValue := "S1"
	_, err = runPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{
		Name:      "attractor-pod",
		Namespace: testCtx.NS.Name,
		Labels:    map[string]string{labelKey: labelValue},
	}))
	if err != nil {
		t.Fatalf("Error running the attractor pod: %v", err)
	}
	// Add a few more nodes without the topology label.
	_, err = createNodes(testCtx.ClientSet, "other-node", nil, 5)
	if err != nil {
		t.Fatalf("Cannot create the second set of nodes: %v", err)
	}
	// Add a new pod with affinity to the attractor pod.
	podName := "pod-with-podaffinity"
	pod, err := runPausePod(testCtx.ClientSet, initPausePod(&pausePodConfig{
		Name:      podName,
		Namespace: testCtx.NS.Name,
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
									{
										Key:      labelKey,
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"S2"},
									}, {
										Key:      labelKey,
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
							TopologyKey: topologyKey,
							Namespaces:  []string{testCtx.NS.Name},
						},
						Weight: 50,
					},
				},
			},
		},
	}))
	if err != nil {
		t.Fatalf("Error running pause pod: %v", err)
	}
	// The new pod must be scheduled on one of the nodes with the same topology
	// key-value as the attractor pod.
	for _, node := range nodesInTopology {
		if node.Name == pod.Spec.NodeName {
			t.Logf("Pod %v got successfully scheduled on node %v.", podName, pod.Spec.NodeName)
			return
		}
	}
	t.Errorf("Pod %v got scheduled on an unexpected node: %v.", podName, pod.Spec.NodeName)
}

// TestImageLocality verifies that the scheduler's image locality priority function
// works correctly, i.e., the pod gets scheduled to the node where its container images are ready.
func TestImageLocality(t *testing.T) {
	testCtx := initTest(t, "image-locality")
	defer testutils.CleanupTest(t, testCtx)

	// We use a fake large image as the test image used by the pod, which has relatively large image size.
	image := v1.ContainerImage{
		Names: []string{
			"fake-large-image:v1",
		},
		SizeBytes: 3000 * 1024 * 1024,
	}

	// Create a node with the large image.
	nodeWithLargeImage, err := createNodeWithImages(testCtx.ClientSet, "testnode-large-image", nil, []v1.ContainerImage{image})
	if err != nil {
		t.Fatalf("cannot create node with a large image: %v", err)
	}

	// Add a few nodes.
	_, err = createNodes(testCtx.ClientSet, "testnode", nil, 10)
	if err != nil {
		t.Fatalf("cannot create nodes: %v", err)
	}

	// Create a pod with containers each having the specified image.
	podName := "pod-using-large-image"
	pod, err := runPodWithContainers(testCtx.ClientSet, initPodWithContainers(testCtx.ClientSet, &podWithContainersConfig{
		Name:       podName,
		Namespace:  testCtx.NS.Name,
		Containers: makeContainersWithImages(image.Names),
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

// TestEvenPodsSpreadPriority verifies that EvenPodsSpread priority functions well.
func TestEvenPodsSpreadPriority(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvenPodsSpread, true)()

	testCtx := initTest(t, "eps-priority")
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name
	defer testutils.CleanupTest(t, testCtx)
	// Add 4 nodes.
	nodes, err := createNodes(cs, "node", nil, 4)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	for i, node := range nodes {
		// Apply labels "zone: zone-{0,1}" and "node: <node name>" to each node.
		labels := map[string]string{
			"zone": fmt.Sprintf("zone-%d", i/2),
			"node": node.Name,
		}
		if err = utils.AddLabelsToNode(cs, node.Name, labels); err != nil {
			t.Fatalf("Cannot add labels to node: %v", err)
		}
		if err = waitForNodeLabels(cs, node.Name, labels); err != nil {
			t.Fatalf("Adding labels to node failed: %v", err)
		}
	}

	// Taint the 0th node
	taint := v1.Taint{
		Key:    "k1",
		Value:  "v1",
		Effect: v1.TaintEffectNoSchedule,
	}
	if err = testutils.AddTaintToNode(cs, nodes[0].Name, taint); err != nil {
		t.Fatalf("Adding taint to node failed: %v", err)
	}
	if err = testutils.WaitForNodeTaints(cs, nodes[0], []v1.Taint{taint}); err != nil {
		t.Fatalf("Taint not seen on node: %v", err)
	}

	pause := imageutils.GetPauseImageName()
	tests := []struct {
		name         string
		incomingPod  *v1.Pod
		existingPods []*v1.Pod
		fits         bool
		want         []string // nodes expected to schedule onto
	}{
		// note: naming starts at index 0
		// the symbol ~X~ means that node is infeasible
		{
			name: "place pod on a ~0~/1/2/3 cluster with MaxSkew=1, node-1 is the preferred fit",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p1").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3c").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits: true,
			want: []string{"node-1"},
		},
		{
			name: "combined with hardSpread constraint on a ~4~/0/1/2 cluster",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", softSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p0a").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p0b").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p0c").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p0d").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p2").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits: true,
			want: []string{"node-2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
