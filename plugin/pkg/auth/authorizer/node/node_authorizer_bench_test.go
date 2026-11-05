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

// BenchmarkHasPathFromPVSecrets measures secret lookups for a secret referenced
// by PVs, either a distinct secret per pair of PVs or one secret shared by all
// of them. It isolates graph lookup cost and does not measure lock contention.
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
					b.Run("SecretAllowed", func(b *testing.B) {
						runHasPathFromBench(b, authz, "node-0", secretVertexType, "ns", secretName, true)
					})
					b.Run("SecretUnrelatedNode", func(b *testing.B) {
						runHasPathFromBench(b, authz, "unrelated-node", secretVertexType, "ns", secretName, false)
					})
				})
			}
		})
	}
}

// BenchmarkHighDegreeSecretAndPVC measures read and write cost when both the
// shared secret (250 PVs) and every PVC (250 pods) are above the index threshold.
func BenchmarkHighDegreeSecretAndPVC(b *testing.B) {
	const pvCount, podsPerPVC, nodeCount = 250, 250, 50
	b.Run(fmt.Sprintf("PVs=%d/PodsPerPVC=%d/Nodes=%d", pvCount, podsPerPVC, nodeCount), func(b *testing.B) {
		authz := newHighDegreeSecretAndPVCAuthorizer(pvCount, podsPerPVC, nodeCount)

		b.Run("Read", func(b *testing.B) {
			b.Run("SecretAllowed", func(b *testing.B) {
				runHasPathFromBench(b, authz, "node-0", secretVertexType, "ns", "shared-secret", true)
			})
			b.Run("SecretUnrelatedNode", func(b *testing.B) {
				runHasPathFromBench(b, authz, "unrelated-node", secretVertexType, "ns", "shared-secret", false)
			})
		})

		b.Run("Write", func(b *testing.B) {
			b.Run("AddDeletePod", func(b *testing.B) {
				extraPod := makeBenchmarkPod("pod-extra", "node-0", "pvc-0")
				b.ReportAllocs()
				for b.Loop() {
					authz.graph.AddPod(extraPod)
					authz.graph.DeletePod("pod-extra", "ns")
				}
			})

			b.Run("AddDeletePV_WithMountedPods", func(b *testing.B) {
				// Pre-populate pvc-extra with podsPerPVC running pods so adding/removing
				// pv-extra exercises the downstream walk through a high-degree PVC.
				for j := range podsPerPVC {
					authz.graph.AddPod(makeBenchmarkPod(fmt.Sprintf("pod-extra-%d", j), fmt.Sprintf("node-%d", j%nodeCount), "pvc-extra"))
				}
				extraPV := makeBenchmarkPV("pv-extra", "pvc-extra", "shared-secret")
				b.ReportAllocs()
				for b.Loop() {
					authz.graph.AddPV(extraPV)
					authz.graph.DeletePV("pv-extra")
				}
			})

			b.Run("UpdateExistingPV", func(b *testing.B) {
				existingPV := makeBenchmarkPV("pv-0", "pvc-0", "shared-secret")
				b.ReportAllocs()
				for b.Loop() {
					authz.graph.AddPV(existingPV)
				}
			})
		})
	})
}

// BenchmarkNormalPodWithUniquePVCs measures the write cost for a normal pod
// mounting one PVC bound to one PV with one secret, where the secret is either
// unique to that PV or shared with enough other PVs to be indexed.
func BenchmarkNormalPodWithUniquePVCs(b *testing.B) {
	for _, sharedSecrets := range []bool{false, true} {
		secretMode := "UniqueSecrets"
		if sharedSecrets {
			secretMode = "SharedSecrets"
		}
		b.Run(fmt.Sprintf("PVCs=1/SecretsPerPV=1/%s", secretMode), func(b *testing.B) {
			g := NewGraph()
			// Pre-populate 250 background pods/PVs so shared resources (like SA or
			// shared PV secrets) are above the 200-edge index threshold.
			for i := range 250 {
				bgClaim := fmt.Sprintf("bg-pvc-%d", i)
				bgSecret := fmt.Sprintf("bg-pv-secret-%d", i)
				if sharedSecrets {
					bgSecret = "shared-pv-secret"
				}
				g.AddPV(makeBenchmarkPV(fmt.Sprintf("bg-pv-%d", i), bgClaim, bgSecret))
				g.AddPod(makeBenchmarkPod(fmt.Sprintf("bg-pod-%d", i), fmt.Sprintf("node-%d", i%50), bgClaim))
			}

			targetSecret := "target-pv-secret"
			if sharedSecrets {
				targetSecret = "shared-pv-secret"
			}
			targetPV := makeBenchmarkPV("target-pv", "target-pvc", targetSecret)
			g.AddPV(targetPV)
			targetPod := makeBenchmarkPod("target-pod", "node-0", "target-pvc")

			b.Run("AddDeletePod", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					g.AddPod(targetPod)
					g.DeletePod("target-pod", "ns")
				}
			})

			b.Run("AddDeletePV", func(b *testing.B) {
				g.AddPod(targetPod)
				b.Cleanup(func() { g.DeletePod("target-pod", "ns") })
				b.ReportAllocs()
				for b.Loop() {
					g.AddPV(targetPV)
					g.DeletePV(targetPV.Name)
				}
				g.AddPV(targetPV)
			})
		})
	}
}

func runHasPathFromBench(b *testing.B, authz *NodeAuthorizer, nodeName string, resourceType vertexType, namespace, resourceName string, wantAllowed bool) {
	b.Helper()
	// An absent vertex would fail before traversal and invalidate
	// the comparison with an existing but unrelated node.
	authz.graph.lock.RLock()
	_, nodeExists := authz.graph.getVertexRLocked(nodeVertexType, "", nodeName)
	_, resourceExists := authz.graph.getVertexRLocked(resourceType, namespace, resourceName)
	authz.graph.lock.RUnlock()
	if !nodeExists {
		b.Fatalf("node %q is missing from the benchmark graph", nodeName)
	}
	if !resourceExists {
		b.Fatalf("resource %s/%s is missing from the benchmark graph", namespace, resourceName)
	}
	b.ReportAllocs()
	for b.Loop() {
		allowed, err := authz.hasPathFrom(nodeName, resourceType, namespace, resourceName)
		if allowed != wantAllowed || (err != nil) == wantAllowed {
			b.Fatalf("hasPathFrom() = (%t, %v), want allowed=%t", allowed, err, wantAllowed)
		}
	}
}

// makeBenchmarkPod returns a pod on nodeName in namespace "ns". If claimName
// is non-empty, the pod mounts that PVC.
func makeBenchmarkPod(name, nodeName, claimName string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		Spec: corev1.PodSpec{
			NodeName:           nodeName,
			ServiceAccountName: "default",
		},
	}
	if claimName != "" {
		pod.Spec.Volumes = []corev1.Volume{{
			Name: "data",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claimName},
			},
		}}
	}
	return pod
}

// makeBenchmarkPV returns a CSI PV bound to claimName in namespace "ns". If
// secretName is non-empty, it is referenced as the node stage secret.
func makeBenchmarkPV(pvName, claimName, secretName string) *corev1.PersistentVolume {
	csi := &corev1.CSIPersistentVolumeSource{
		Driver:       "csi.example.com",
		VolumeHandle: pvName,
	}
	if secretName != "" {
		csi.NodeStageSecretRef = &corev1.SecretReference{Namespace: "ns", Name: secretName}
	}
	return &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: pvName},
		Spec: corev1.PersistentVolumeSpec{
			ClaimRef:               &corev1.ObjectReference{Namespace: "ns", Name: claimName},
			PersistentVolumeSource: corev1.PersistentVolumeSource{CSI: csi},
		},
	}
}

// newHighDegreeSecretAndPVCAuthorizer builds a graph where one shared secret is
// referenced by pvCount PVs and every PVC is mounted by podsPerPVC pods.
func newHighDegreeSecretAndPVCAuthorizer(pvCount, podsPerPVC, nodeCount int) *NodeAuthorizer {
	g := NewGraph()
	for i := range pvCount {
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(makeBenchmarkPV(fmt.Sprintf("pv-%d", i), claimName, "shared-secret"))
		for j := range podsPerPVC {
			g.AddPod(makeBenchmarkPod(fmt.Sprintf("pod-%d-%d", i, j), fmt.Sprintf("node-%d", j%nodeCount), claimName))
		}
	}
	g.AddPod(makeBenchmarkPod("unrelated-pod", "unrelated-node", ""))
	return &NodeAuthorizer{graph: g}
}

// newPVSecretBenchmarkAuthorizer builds a graph of pvCount PVs each mounted by
// one pod, referencing either one shared secret or one secret per two PVs.
func newPVSecretBenchmarkAuthorizer(pvCount, nodeCount int, shared bool) *NodeAuthorizer {
	g := NewGraph()
	for i := range pvCount {
		secretName := fmt.Sprintf("secret-%d", i/2)
		if shared {
			secretName = "shared-secret"
		}
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(makeBenchmarkPV(fmt.Sprintf("pv-%d", i), claimName, secretName))
		g.AddPod(makeBenchmarkPod(fmt.Sprintf("pod-%d", i), fmt.Sprintf("node-%d", i%nodeCount), claimName))
	}
	// Register the unrelated node through the same graph population path, without
	// giving it a reference to any of the volumes or secrets under test.
	g.AddPod(makeBenchmarkPod("unrelated-pod", "unrelated-node", ""))
	return &NodeAuthorizer{graph: g}
}
