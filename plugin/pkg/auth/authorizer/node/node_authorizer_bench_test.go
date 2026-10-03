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
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// BenchmarkHasPathFromPVSecrets isolates graph lookup cost, including the read
// lock, without concurrent graph mutations. It does not measure lock contention.
func BenchmarkHasPathFromPVSecrets(b *testing.B) {
	for _, pvCount := range []int{4, 1000, 10000, 100000, 400000} {
		// Keep large cases representative of multiple volumes per node, rather
		// than requiring hundreds of thousands of distinct Kubernetes nodes.
		nodeCount := min(pvCount, 1000)
		b.Run(fmt.Sprintf("PVs=%d/Nodes=%d", pvCount, nodeCount), func(b *testing.B) {
			for _, shared := range []bool{false, true} {
				topology, secretName := "UniqueSecrets", "secret-0"
				if shared {
					topology, secretName = "SharedSecret", "shared-secret"
				}
				b.Run(topology, func(b *testing.B) {
					authz := newPVSecretBenchmarkAuthorizer(pvCount, nodeCount, shared)
					for _, tc := range []struct {
						name         string
						nodeName     string
						resourceType vertexType
						namespace    string
						resourceName string
						wantAllowed  bool
					}{
						{"SecretAllowed", "node-0", secretVertexType, "ns", secretName, true},
						{"SecretUnrelatedNode", "unrelated-node", secretVertexType, "ns", secretName, false},
						{"PVAllowed", "node-0", pvVertexType, "", "pv-0", true},
					} {
						b.Run(tc.name, func(b *testing.B) {
							// An absent vertex would fail before traversal and invalidate
							// the comparison with an existing but unrelated node.
							authz.graph.lock.RLock()
							_, nodeExists := authz.graph.getVertexRLocked(nodeVertexType, "", tc.nodeName)
							_, resourceExists := authz.graph.getVertexRLocked(tc.resourceType, tc.namespace, tc.resourceName)
							authz.graph.lock.RUnlock()
							if !nodeExists {
								b.Fatalf("node %q is missing from the benchmark graph", tc.nodeName)
							}
							if !resourceExists {
								b.Fatalf("resource %s/%s is missing from the benchmark graph", tc.namespace, tc.resourceName)
							}
							b.ReportAllocs()
							for b.Loop() {
								allowed, err := authz.hasPathFrom(tc.nodeName, tc.resourceType, tc.namespace, tc.resourceName)
								if allowed != tc.wantAllowed || (err != nil) == tc.wantAllowed {
									b.Fatalf("hasPathFrom() = (%t, %v), want allowed=%t", allowed, err, tc.wantAllowed)
								}
							}
						})
					}
				})
			}
		})
	}
}

func newPVSecretBenchmarkAuthorizer(pvCount, nodeCount int, shared bool) *NodeAuthorizer {
	g := NewGraph()
	for i := range pvCount {
		secretName := fmt.Sprintf("secret-%d", i)
		if shared {
			secretName = "shared-secret"
		}
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(&corev1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pv-%d", i)},
			Spec: corev1.PersistentVolumeSpec{
				ClaimRef: &corev1.ObjectReference{Namespace: "ns", Name: claimName},
				PersistentVolumeSource: corev1.PersistentVolumeSource{
					CSI: &corev1.CSIPersistentVolumeSource{
						Driver:       "csi.example.com",
						VolumeHandle: fmt.Sprintf("volume-%d", i),
						NodeStageSecretRef: &corev1.SecretReference{
							Namespace: "ns",
							Name:      secretName,
						},
					},
				},
			},
		})
		g.AddPod(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pod-%d", i)},
			Spec: corev1.PodSpec{
				NodeName: fmt.Sprintf("node-%d", i%nodeCount),
				Volumes: []corev1.Volume{{
					Name: "data",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claimName},
					},
				}},
			},
		})
	}
	// Register the unrelated node through the same graph population path, without
	// giving it a reference to any of the volumes or secrets under test.
	g.AddPod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "unrelated-pod"},
		Spec:       corev1.PodSpec{NodeName: "unrelated-node"},
	})
	return &NodeAuthorizer{graph: g}
}
