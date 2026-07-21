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

package node

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func TestGraphPopulator_CoalescedDeleteAndRecreate(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	graph := NewGraph()

	g := &graphPopulator{
		graph:     graph,
		podQueue:  newRateLimitingQueue("test_node_authorizer_pod_populator"),
		podLister: corev1listers.NewPodLister(indexer),
	}

	// Step 1: Pod 1 (UID 1) on node1 with secret1 is in the graph.
	pod1 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-x",
			Namespace: "default",
			UID:       types.UID("uid-1"),
		},
		Spec: corev1.PodSpec{
			NodeName: "node1",
			Volumes: []corev1.Volume{
				{
					Name: "secret-vol-1",
					VolumeSource: corev1.VolumeSource{
						Secret: &corev1.SecretVolumeSource{
							SecretName: "secret1",
						},
					},
				},
			},
		},
	}
	graph.AddPod(pod1)

	// Verify graph has pod1 state
	podVert, ok := graph.getVertexRLocked(podVertexType, "default", "pod-x")
	if !ok {
		t.Fatalf("expected pod-x vertex")
	}
	node1Vert, ok := graph.getVertexRLocked(nodeVertexType, "", "node1")
	if !ok {
		t.Fatalf("expected node1 vertex")
	}
	secret1Vert, ok := graph.getVertexRLocked(secretVertexType, "default", "secret1")
	if !ok {
		t.Fatalf("expected secret1 vertex")
	}
	if !graph.graph.HasEdgeFromTo(podVert, node1Vert) || !graph.graph.HasEdgeFromTo(secret1Vert, podVert) {
		t.Fatalf("expected initial graph edges for pod1")
	}

	// Step 2: Simulate pod1 deletion event and pod2 creation event queued together.
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-x",
			Namespace: "default",
			UID:       types.UID("uid-2"),
		},
		Spec: corev1.PodSpec{
			NodeName: "node2",
			Volumes: []corev1.Volume{
				{
					Name: "secret-vol-2",
					VolumeSource: corev1.VolumeSource{
						Secret: &corev1.SecretVolumeSource{
							SecretName: "secret2",
						},
					},
				},
			},
		},
	}

	if err := indexer.Add(pod2); err != nil {
		t.Fatalf("failed to add pod2 to indexer: %v", err)
	}

	// Enqueue delete for pod1, then enqueue add for pod2 (coalesces into 1 item in workqueue)
	g.deletePod(pod1)
	g.addPod(pod2)

	if g.podQueue.Len() != 1 {
		t.Fatalf("expected queue length 1 due to event coalescing, got %d", g.podQueue.Len())
	}

	// Step 3: Run processNextWorkItem once.
	processed := processNextWorkItem(g.podQueue, g.processPodKey)
	if !processed {
		t.Fatalf("expected processNextWorkItem to return true")
	}
	if g.podQueue.Len() != 0 {
		t.Fatalf("expected queue to be empty after processing, got %d", g.podQueue.Len())
	}

	// Step 4: Assert graph state.
	graph.lock.RLock()
	defer graph.lock.RUnlock()

	podVert, ok = graph.getVertexRLocked(podVertexType, "default", "pod-x")
	if !ok {
		t.Fatalf("expected pod-x vertex after processing")
	}
	node2Vert, ok := graph.getVertexRLocked(nodeVertexType, "", "node2")
	if !ok {
		t.Fatalf("expected node2 vertex after processing")
	}
	secret2Vert, ok := graph.getVertexRLocked(secretVertexType, "default", "secret2")
	if !ok {
		t.Fatalf("expected secret2 vertex after processing")
	}

	if !graph.graph.HasEdgeFromTo(podVert, node2Vert) {
		t.Errorf("expected edge from pod-x to node2")
	}
	if !graph.graph.HasEdgeFromTo(secret2Vert, podVert) {
		t.Errorf("expected edge from secret2 to pod-x")
	}

	if _, secret1Exists := graph.getVertexRLocked(secretVertexType, "default", "secret1"); secret1Exists {
		t.Errorf("expected secret1 vertex to be purged from graph")
	}
	if node1Vert, ok := graph.getVertexRLocked(nodeVertexType, "", "node1"); ok {
		if graph.graph.HasEdgeFromTo(podVert, node1Vert) {
			t.Errorf("expected pod-x to NOT have edge to node1")
		}
	}
}
