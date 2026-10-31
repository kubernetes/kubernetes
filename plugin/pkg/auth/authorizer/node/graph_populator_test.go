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
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1listers "k8s.io/client-go/listers/core/v1"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
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

func TestGraphPopulator_UpdatePodVolumes(t *testing.T) {
	secretVolume := func(volName, secretName string) corev1.Volume {
		return corev1.Volume{
			Name: volName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{SecretName: secretName},
			},
		}
	}
	configMapVolume := func(volName, cmName string) corev1.Volume {
		return corev1.Volume{
			Name: volName,
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: cmName},
				},
			},
		}
	}
	projectedVolume := func(volName, secretName, cmName string) corev1.Volume {
		return corev1.Volume{
			Name: volName,
			VolumeSource: corev1.VolumeSource{
				Projected: &corev1.ProjectedVolumeSource{
					Sources: []corev1.VolumeProjection{
						{
							Secret: &corev1.SecretProjection{
								LocalObjectReference: corev1.LocalObjectReference{Name: secretName},
							},
						},
						{
							ConfigMap: &corev1.ConfigMapProjection{
								LocalObjectReference: corev1.LocalObjectReference{Name: cmName},
							},
						},
					},
				},
			},
		}
	}

	testCases := []struct {
		name                   string
		initialVolumes         []corev1.Volume
		updatedVolumes         []corev1.Volume
		incrementGeneration    bool
		expectEnqueued         bool
		expectedSecretEdges    []string
		absentSecrets          []string
		expectedConfigMapEdges []string
		absentConfigMaps       []string
	}{
		{
			name: "unchanged generation does not enqueue pod",
			initialVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
			},
			updatedVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
			},
			incrementGeneration:    false,
			expectEnqueued:         false,
			expectedSecretEdges:    []string{"secret1"},
			expectedConfigMapEdges: []string{"cm1"},
		},
		{
			name: "incremented generation with unchanged volumes does not enqueue pod",
			initialVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
			},
			updatedVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
			},
			incrementGeneration:    true,
			expectEnqueued:         false,
			expectedSecretEdges:    []string{"secret1"},
			expectedConfigMapEdges: []string{"cm1"},
		},
		{
			name: "adding secret and projected volumes adds new graph edges",
			initialVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
			},
			updatedVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
				secretVolume("secret-vol-2", "secret2"),
				projectedVolume("projected-vol", "projected-secret", "projected-cm"),
			},
			incrementGeneration:    true,
			expectEnqueued:         true,
			expectedSecretEdges:    []string{"secret1", "secret2", "projected-secret"},
			expectedConfigMapEdges: []string{"cm1", "projected-cm"},
		},
		{
			name: "removing secret and projected volumes removes their graph edges",
			initialVolumes: []corev1.Volume{
				secretVolume("secret-vol-1", "secret1"),
				configMapVolume("cm-vol-1", "cm1"),
				secretVolume("secret-vol-2", "secret2"),
				projectedVolume("projected-vol", "projected-secret", "projected-cm"),
			},
			updatedVolumes: []corev1.Volume{
				configMapVolume("cm-vol-1", "cm1"),
				secretVolume("secret-vol-2", "secret2"),
			},
			incrementGeneration:    true,
			expectEnqueued:         true,
			expectedSecretEdges:    []string{"secret2"},
			absentSecrets:          []string{"secret1", "projected-secret"},
			expectedConfigMapEdges: []string{"cm1"},
			absentConfigMaps:       []string{"projected-cm"},
		},
		{
			name: "replacing volume with same volume count updates graph edges",
			initialVolumes: []corev1.Volume{
				configMapVolume("cm-vol-1", "cm1"),
				secretVolume("secret-vol-2", "secret2"),
			},
			updatedVolumes: []corev1.Volume{
				configMapVolume("cm-vol-1", "cm1"),
				secretVolume("secret-vol-3", "secret3"),
			},
			incrementGeneration:    true,
			expectEnqueued:         true,
			expectedSecretEdges:    []string{"secret3"},
			absentSecrets:          []string{"secret2"},
			expectedConfigMapEdges: []string{"cm1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			graph := NewGraph()
			g := &graphPopulator{
				graph:     graph,
				podQueue:  newRateLimitingQueue("test_node_authorizer_pod_volumes"),
				podLister: corev1listers.NewPodLister(indexer),
			}

			oldPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pod-dynamic-vols",
					Namespace:  "default",
					UID:        types.UID("uid-1"),
					Generation: 1,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
					Volumes:  tc.initialVolumes,
				},
			}
			if err := indexer.Add(oldPod); err != nil {
				t.Fatalf("failed to add oldPod to indexer: %v", err)
			}
			g.addPod(oldPod)
			if !processNextWorkItem(g.podQueue, g.processPodKey) {
				t.Fatalf("expected initial pod add to be processed")
			}

			newPod := oldPod.DeepCopy()
			if tc.incrementGeneration {
				newPod.Generation++
			}
			newPod.Spec.Volumes = tc.updatedVolumes
			if err := indexer.Update(newPod); err != nil {
				t.Fatalf("failed to update newPod in indexer: %v", err)
			}
			g.updatePod(oldPod, newPod)

			if tc.expectEnqueued {
				if g.podQueue.Len() != 1 {
					t.Fatalf("expected queue length 1, got %d", g.podQueue.Len())
				}
				if !processNextWorkItem(g.podQueue, g.processPodKey) {
					t.Fatalf("expected processNextWorkItem to return true")
				}
			} else if g.podQueue.Len() != 0 {
				t.Fatalf("expected queue length 0, got %d", g.podQueue.Len())
			}

			podVert, ok := graph.getVertexRLocked(podVertexType, "default", "pod-dynamic-vols")
			if !ok {
				t.Fatalf("expected pod vertex")
			}
			for _, secretName := range tc.expectedSecretEdges {
				vert, exists := graph.getVertexRLocked(secretVertexType, "default", secretName)
				if !exists || !graph.graph.HasEdgeFromTo(vert, podVert) {
					t.Errorf("expected edge from secret %q to pod", secretName)
				}
			}
			for _, secretName := range tc.absentSecrets {
				if _, exists := graph.getVertexRLocked(secretVertexType, "default", secretName); exists {
					t.Errorf("expected secret %q vertex to be removed from graph", secretName)
				}
			}
			for _, cmName := range tc.expectedConfigMapEdges {
				vert, exists := graph.getVertexRLocked(configMapVertexType, "default", cmName)
				if !exists || !graph.graph.HasEdgeFromTo(vert, podVert) {
					t.Errorf("expected edge from configmap %q to pod", cmName)
				}
			}
			for _, cmName := range tc.absentConfigMaps {
				if _, exists := graph.getVertexRLocked(configMapVertexType, "default", cmName); exists {
					t.Errorf("expected configmap %q vertex to be removed from graph", cmName)
				}
			}
		})
	}
}

func TestGraphPopulatorPodCheckpointAssignment(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	graph := NewGraph()
	g := &graphPopulator{
		graph:            graph,
		checkpointQueue:  newRateLimitingQueue("test_node_authorizer_checkpoint_populator"),
		checkpointLister: nodelisters.NewPodCheckpointLister(indexer),
	}
	defer g.checkpointQueue.ShutDown()
	featuresForTest := utilfeature.DefaultFeatureGate.DeepCopy()
	featuregatetesting.SetFeatureGateDuringTest(t, featuresForTest, features.PodLevelCheckpointRestore, true)
	authz := NewAuthorizer(graph, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())
	authz.features = featuresForTest
	process := func(allowedNode string) {
		t.Helper()
		if g.checkpointQueue.Len() != 1 {
			t.Fatalf("queued checkpoints = %d, want 1", g.checkpointQueue.Len())
		}
		key, _ := g.checkpointQueue.Get()
		defer g.checkpointQueue.Done(key)
		if err := g.processCheckpointKey(key); err != nil {
			t.Fatal(err)
		}
		for _, nodeName := range []string{"node0", "node1"} {
			attrs := authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "system:node:" + nodeName, Groups: []string{"system:nodes"}},
				ResourceRequest: true, Verb: "get", APIGroup: "node.k8s.io", Resource: "podcheckpoints", Namespace: "ns", Name: "checkpoint"}
			decision, reason, err := authz.Authorize(context.Background(), attrs)
			if err != nil || (decision == authorizer.DecisionAllow) != (nodeName == allowedNode) {
				t.Fatalf("node %s decision = %v (%s), error = %v, assigned node = %q", nodeName, decision, reason, err, allowedNode)
			}
		}
	}
	checkpoint := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "old"}}
	if err := indexer.Add(checkpoint); err != nil {
		t.Fatal(err)
	}
	g.addCheckpoint(checkpoint)
	process("")

	assigned := checkpoint.DeepCopy()
	assigned.Status.NodeName = new("node0")
	if err := indexer.Update(assigned); err != nil {
		t.Fatal(err)
	}
	g.updateCheckpoint(checkpoint, assigned)
	process("node0")

	// A coalesced delete/recreate must use the new object from the informer.
	replacement := assigned.DeepCopy()
	replacement.UID = "new"
	replacement.Status.NodeName = new("node1")
	if err := indexer.Update(replacement); err != nil {
		t.Fatal(err)
	}
	g.deleteCheckpoint(cache.DeletedFinalStateUnknown{Obj: assigned})
	g.addCheckpoint(replacement)
	process("node1")

	unassigned := replacement.DeepCopy()
	unassigned.Status.NodeName = nil
	if err := indexer.Update(unassigned); err != nil {
		t.Fatal(err)
	}
	g.updateCheckpoint(replacement, unassigned)
	process("")

	if err := indexer.Delete(unassigned); err != nil {
		t.Fatal(err)
	}
	g.deleteCheckpoint(unassigned)
	process("")
}
