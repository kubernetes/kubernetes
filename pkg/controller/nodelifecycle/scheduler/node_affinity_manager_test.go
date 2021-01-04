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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

func createPodWithRequiredDuringExecution(name string, key string, op v1.NodeSelectorOperator, values ...string) *v1.Pod {
	pod := testutil.NewPod(name, "")
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingRequiredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      key,
								Operator: op,
								Values:   values,
							},
						},
					},
				},
			},
		},
	}
	return pod
}

func createNodeWithLabels(name string, labels map[string]string) *v1.Node {
	node := testutil.NewNode("node1")
	node.Labels = labels
	return node
}

func TestUpdateNodeLabels(t *testing.T) {
	k := "test-label"
	v := "test-value"
	testCases := []struct {
		description string
		pods        []*v1.Pod
		prevNode    *v1.Node
		newNode     *v1.Node
		deletedPods map[string]bool
	}{
		{
			description: "evict pod",
			pods:        []*v1.Pod{createPodWithRequiredDuringExecution("pod1", k, v1.NodeSelectorOpIn, v)},
			prevNode: createNodeWithLabels("node1", map[string]string{
				k: v,
			}),
			newNode: createNodeWithLabels("node1", map[string]string{
				k: v + "changed",
			}),
			deletedPods: map[string]bool{
				"pod1": true,
			},
		},
		{
			description: "pod should survive",
			pods:        []*v1.Pod{createPodWithRequiredDuringExecution("pod1", k, v1.NodeSelectorOpExists)},
			prevNode: createNodeWithLabels("node1", map[string]string{
				k: v,
			}),
			newNode: createNodeWithLabels("node1", map[string]string{
				k: v + "changed",
			}),
		},
		{
			description: "all pods should be evicted",
			pods: []*v1.Pod{
				createPodWithRequiredDuringExecution("pod1", k, v1.NodeSelectorOpIn, "15"),
				createPodWithRequiredDuringExecution("pod2", k, v1.NodeSelectorOpGt, "5"),
				createPodWithRequiredDuringExecution("pod3", "to-be-deleted-label", v1.NodeSelectorOpExists),
			},
			prevNode: createNodeWithLabels("node1", map[string]string{
				k:                     "10",
				"to-be-deleted-label": v,
			}),
			newNode: createNodeWithLabels("node1", map[string]string{
				k: "0",
			}),
			deletedPods: map[string]bool{
				"pod1": true,
				"pod2": true,
				"pod3": true,
			},
		},
		{
			description: "pod1 should survive, pod2 should be evicted",
			pods: []*v1.Pod{
				createPodWithRequiredDuringExecution("pod1", k, v1.NodeSelectorOpLt, "15"),
				createPodWithRequiredDuringExecution("pod2", k, v1.NodeSelectorOpGt, "5"),
			},
			prevNode: createNodeWithLabels("node1", map[string]string{
				k: "5",
			}),
			newNode: createNodeWithLabels("node1", map[string]string{
				k: "0",
			}),
			deletedPods: map[string]bool{
				"pod2": true,
			},
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			stopCh := make(chan struct{})
			fakeClientset := fake.NewSimpleClientset()
			holder := &nodeHolder{}
			holder.setNode(item.prevNode)

			controller := NewNodeAffinityManager(fakeClientset, holder.getNode, func(nodeName string) ([]*v1.Pod, error) {
				return item.pods, nil
			})
			go controller.Run(stopCh)

			fakeClientset.ClearActions()
			time.Sleep(timeForControllerToProgress)
			holder.setNode(item.newNode)
			controller.NodeUpdated(item.prevNode, item.newNode)
			// wait a bit
			time.Sleep(timeForControllerToProgress)

			deleted := map[string]bool{}
			for _, action := range fakeClientset.Actions() {
				deleteAction, ok := action.(clienttesting.DeleteActionImpl)
				if !ok {
					t.Logf("Unexpected delete action with verb %v. Skipped.", action.GetVerb())
					continue
				}
				if deleteAction.GetResource().Resource != "pods" {
					t.Logf("Unexpected deleted resource %v. Skipped.", action.GetResource().Resource)
					continue
				}
				if shouldBeDeleted, ok := item.deletedPods[deleteAction.GetName()]; ok && !shouldBeDeleted {
					t.Errorf("Pod %v should not be deleted", deleteAction.GetName())
				}
				deleted[deleteAction.GetName()] = true
			}
			for podName, shouldBeDeleted := range item.deletedPods {
				if !shouldBeDeleted {
					continue
				}
				if _, ok := deleted[podName]; !ok {
					t.Errorf("%v: Pod %v should be deleted", item.description, podName)
				}
			}
			close(stopCh)
		})
	}
}
