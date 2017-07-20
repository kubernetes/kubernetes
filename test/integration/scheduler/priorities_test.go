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
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testutils "k8s.io/kubernetes/test/utils"
)

// This file tests the scheduler priority functions.

// TestNodeAffinity verifies that scheduler's node affinity priority function
// works correctly.
func TestNodeAffinity(t *testing.T) {
	context := initTest(t, "node-affinity")
	defer cleanupTest(t, context)
	// Add a few nodes.
	nodes, err := createNodes(context.clientSet, "testnode", nil, 5)
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
	if err = testutils.AddLabelsToNode(context.clientSet, labeledNode.Name, labels); err != nil {
		t.Fatalf("Cannot add labels to node: %v", err)
	}
	if err = waitForNodeLabels(context.clientSet, labeledNode.Name, labels); err != nil {
		t.Fatalf("Adding labels to node didn't succeed: %v", err)
	}
	// Create a pod with node affinity.
	podName := "pod-with-node-affinity"
	pod, err := runPausePod(context.clientSet, &pausePodConfig{
		Name:      podName,
		Namespace: context.ns.Name,
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
	})
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
	context := initTest(t, "pod-affinity")
	defer cleanupTest(t, context)
	// Add a few nodes.
	nodesInTopology, err := createNodes(context.clientSet, "in-topology", nil, 5)
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
		if err = testutils.AddLabelsToNode(context.clientSet, node.Name, nodeLabels); err != nil {
			t.Fatalf("Cannot add labels to node %v: %v", node.Name, err)
		}
		if err = waitForNodeLabels(context.clientSet, node.Name, nodeLabels); err != nil {
			t.Fatalf("Adding labels to node %v didn't succeed: %v", node.Name, err)
		}
	}
	// Add a pod with a label and wait for it to schedule.
	labelKey := "service"
	labelValue := "S1"
	_, err = runPausePod(context.clientSet, &pausePodConfig{
		Name:      "attractor-pod",
		Namespace: context.ns.Name,
		Labels:    map[string]string{labelKey: labelValue},
	})
	if err != nil {
		t.Fatalf("Error running the attractor pod: %v", err)
	}
	// Add a few more nodes without the topology label.
	_, err = createNodes(context.clientSet, "other-node", nil, 5)
	if err != nil {
		t.Fatalf("Cannot create the second set of nodes: %v", err)
	}
	// Add a new pod with affinity to the attractor pod.
	podName := "pod-with-podaffinity"
	pod, err := runPausePod(context.clientSet, &pausePodConfig{
		Name:      podName,
		Namespace: context.ns.Name,
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
							Namespaces:  []string{context.ns.Name},
						},
						Weight: 50,
					},
				},
			},
		},
	})
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
