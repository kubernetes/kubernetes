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
						{"PVUnrelatedNode", "unrelated-node", pvVertexType, "", "pv-0", false},
						{"PVCAllowed", "node-0", pvcVertexType, "ns", "pvc-0", true},
						{"PVCUnrelatedNode", "unrelated-node", pvcVertexType, "ns", "pvc-0", false},
					} {
						b.Run(tc.name, func(b *testing.B) {
							runHasPathFromBench(b, authz, tc.nodeName, tc.resourceType, tc.namespace, tc.resourceName, tc.wantAllowed)
						})
					}
				})
			}
		})
	}
}

// BenchmarkHighDegreeSecretAndPVC benchmarks both read (hasPathFrom) and write
// (AddPod/DeletePod, AddPV/DeletePV) costs when both the Secret (P PVs) and
// each bound PVC (M pods) have high out-degrees.
func BenchmarkHighDegreeSecretAndPVC(b *testing.B) {
	for _, shape := range []struct {
		name       string
		pvCount    int
		podsPerPVC int
		nodeCount  int
	}{
		// Both Secret (250 >= 200) and each PVC (250 >= 200) have authoritative indexes.
		{name: "PVs=250/PodsPerPVC=250/Nodes=50", pvCount: 250, podsPerPVC: 250, nodeCount: 50},
		// Secret is below threshold (100 < 200), but each PVC (250 >= 200) has an authoritative index.
		{name: "PVs=100/PodsPerPVC=250/Nodes=50", pvCount: 100, podsPerPVC: 250, nodeCount: 50},
		// Secret is at the exact threshold boundary (199 PVs before adding the 200th PV), each PVC has 250 pods.
		{name: "PVs=199/PodsPerPVC=250/Nodes=50", pvCount: 199, podsPerPVC: 250, nodeCount: 50},
	} {
		b.Run(shape.name, func(b *testing.B) {
			authz := newHighDegreeSecretAndPVCAuthorizer(shape.pvCount, shape.podsPerPVC, shape.nodeCount)

			b.Run("Read", func(b *testing.B) {
				for _, tc := range []struct {
					name         string
					nodeName     string
					resourceType vertexType
					namespace    string
					resourceName string
					wantAllowed  bool
				}{
					{"SecretAllowed", "node-0", secretVertexType, "ns", "shared-secret", true},
					{"SecretUnrelatedNode", "unrelated-node", secretVertexType, "ns", "shared-secret", false},
					{"PVCAllowed", "node-0", pvcVertexType, "ns", "pvc-0", true},
					{"PVCUnrelatedNode", "unrelated-node", pvcVertexType, "ns", "pvc-0", false},
					{"PVAllowed", "node-0", pvVertexType, "", "pv-0", true},
					{"PVUnrelatedNode", "unrelated-node", pvVertexType, "", "pv-0", false},
				} {
					b.Run(tc.name, func(b *testing.B) {
						runHasPathFromBench(b, authz, tc.nodeName, tc.resourceType, tc.namespace, tc.resourceName, tc.wantAllowed)
					})
				}
			})

			b.Run("Write", func(b *testing.B) {
				b.Run("AddDeletePod", func(b *testing.B) {
					extraPod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod-extra"},
						Spec: corev1.PodSpec{
							NodeName:           "node-0",
							ServiceAccountName: "default",
							Volumes: []corev1.Volume{{
								Name: "data",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "pvc-0"},
								},
							}},
						},
					}
					b.ReportAllocs()
					for b.Loop() {
						authz.graph.AddPod(extraPod)
						authz.graph.DeletePod("pod-extra", "ns")
					}
				})

				b.Run("AddDeletePV_WithMountedPods", func(b *testing.B) {
					// Pre-populate pvc-extra with podsPerPVC running pods so adding/removing
					// pv-extra exercises the downstream walk through a high-degree PVC.
					for j := range shape.podsPerPVC {
						authz.graph.AddPod(&corev1.Pod{
							ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pod-extra-%d", j)},
							Spec: corev1.PodSpec{
								NodeName:           fmt.Sprintf("node-%d", j%shape.nodeCount),
								ServiceAccountName: "default",
								Volumes: []corev1.Volume{{
									Name: "data",
									VolumeSource: corev1.VolumeSource{
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "pvc-extra"},
									},
								}},
							},
						})
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
}

// BenchmarkNormalPodWithUniquePVCs measures the write and read cost for a
// normal pod mounting 1-2 unique PVCs, each bound to 1 PV with 0-2 secrets.
func BenchmarkNormalPodWithUniquePVCs(b *testing.B) {
	for _, pvcsPerPod := range []int{1, 2} {
		for _, secretsPerPV := range []int{0, 1, 2} {
			for _, sharedSecrets := range []bool{false, true} {
				if secretsPerPV == 0 && sharedSecrets {
					continue
				}
				secretMode := "UniqueSecrets"
				if sharedSecrets {
					secretMode = "SharedSecrets"
				}
				name := fmt.Sprintf("PVCs=%d/SecretsPerPV=%d/%s", pvcsPerPod, secretsPerPV, secretMode)
				b.Run(name, func(b *testing.B) {
					g := NewGraph()
					// Pre-populate 250 background pods/PVs so shared resources (like SA or
					// shared PV secrets) are above the 200-edge index threshold.
					for i := range 250 {
						bgClaim := fmt.Sprintf("bg-pvc-%d", i)
						var bgSecrets []string
						for s := range secretsPerPV {
							if sharedSecrets {
								bgSecrets = append(bgSecrets, fmt.Sprintf("shared-pv-secret-%d", s))
							} else {
								bgSecrets = append(bgSecrets, fmt.Sprintf("bg-pv-secret-%d-%d", i, s))
							}
						}
						g.AddPV(makeBenchmarkPVWithSecrets(fmt.Sprintf("bg-pv-%d", i), bgClaim, bgSecrets))
						g.AddPod(&corev1.Pod{
							ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("bg-pod-%d", i)},
							Spec: corev1.PodSpec{
								NodeName:           fmt.Sprintf("node-%d", i%50),
								ServiceAccountName: "default",
								Volumes: []corev1.Volume{{
									Name: "vol-0",
									VolumeSource: corev1.VolumeSource{
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: bgClaim},
									},
								}},
							},
						})
					}

					var volumes []corev1.Volume
					var targetPVs []*corev1.PersistentVolume
					for v := range pvcsPerPod {
						claimName := fmt.Sprintf("target-pvc-%d", v)
						pvName := fmt.Sprintf("target-pv-%d", v)
						var secretNames []string
						for s := range secretsPerPV {
							if sharedSecrets {
								secretNames = append(secretNames, fmt.Sprintf("shared-pv-secret-%d", s))
							} else {
								secretNames = append(secretNames, fmt.Sprintf("target-pv-secret-%d-%d", v, s))
							}
						}
						pvObj := makeBenchmarkPVWithSecrets(pvName, claimName, secretNames)
						targetPVs = append(targetPVs, pvObj)
						g.AddPV(pvObj)
						volumes = append(volumes, corev1.Volume{
							Name: fmt.Sprintf("vol-%d", v),
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claimName},
							},
						})
					}

					targetPod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "target-pod"},
						Spec: corev1.PodSpec{
							NodeName:           "node-0",
							ServiceAccountName: "default",
							Volumes:            volumes,
						},
					}

					b.Run("AddDeletePod", func(b *testing.B) {
						b.ReportAllocs()
						for b.Loop() {
							g.AddPod(targetPod)
							g.DeletePod("target-pod", "ns")
						}
					})

					b.Run("UpdatePod", func(b *testing.B) {
						g.AddPod(targetPod)
						b.Cleanup(func() { g.DeletePod("target-pod", "ns") })
						b.ReportAllocs()
						for b.Loop() {
							g.AddPod(targetPod)
						}
					})

					b.Run("AddDeletePV", func(b *testing.B) {
						g.AddPod(targetPod)
						b.Cleanup(func() { g.DeletePod("target-pod", "ns") })
						b.ReportAllocs()
						for b.Loop() {
							for _, pvObj := range targetPVs {
								g.AddPV(pvObj)
								g.DeletePV(pvObj.Name)
							}
						}
						for _, pvObj := range targetPVs {
							g.AddPV(pvObj)
						}
					})
				})
			}
		}
	}
}

// BenchmarkMixedDirectAndPVSecret benchmarks a Secret whose degree crosses the
// index threshold via a mix of direct Pod volume mounts and PV references.
func BenchmarkMixedDirectAndPVSecret(b *testing.B) {
	g := NewGraph()
	nodeCount := 100
	// 120 direct pods + 120 PVs = 240 outgoing edges on shared-secret (>= 200).
	for i := range 120 {
		g.AddPod(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("direct-pod-%d", i)},
			Spec: corev1.PodSpec{
				NodeName:           fmt.Sprintf("node-%d", i%nodeCount),
				ServiceAccountName: "default",
				Volumes: []corev1.Volume{{
					Name: "secret-vol",
					VolumeSource: corev1.VolumeSource{
						Secret: &corev1.SecretVolumeSource{SecretName: "shared-secret"},
					},
				}},
			},
		})
	}
	for i := range 120 {
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(makeBenchmarkPV(fmt.Sprintf("pv-%d", i), claimName, "shared-secret"))
		g.AddPod(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pv-pod-%d", i)},
			Spec: corev1.PodSpec{
				NodeName:           fmt.Sprintf("node-%d", i%nodeCount),
				ServiceAccountName: "default",
				Volumes: []corev1.Volume{{
					Name: "data",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claimName},
					},
				}},
			},
		})
	}
	g.AddPod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "unrelated-pod"},
		Spec: corev1.PodSpec{
			NodeName:           "unrelated-node",
			ServiceAccountName: "default",
		},
	})
	authz := &NodeAuthorizer{graph: g}

	for _, tc := range []struct {
		name         string
		nodeName     string
		resourceType vertexType
		namespace    string
		resourceName string
		wantAllowed  bool
	}{
		{"SecretAllowed", "node-0", secretVertexType, "ns", "shared-secret", true},
		{"SecretUnrelatedNode", "unrelated-node", secretVertexType, "ns", "shared-secret", false},
	} {
		b.Run(tc.name, func(b *testing.B) {
			runHasPathFromBench(b, authz, tc.nodeName, tc.resourceType, tc.namespace, tc.resourceName, tc.wantAllowed)
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

func makeBenchmarkPV(pvName, claimName, secretName string) *corev1.PersistentVolume {
	if secretName == "" {
		return makeBenchmarkPVWithSecrets(pvName, claimName, nil)
	}
	return makeBenchmarkPVWithSecrets(pvName, claimName, []string{secretName})
}

func makeBenchmarkPVWithSecrets(pvName, claimName string, secretNames []string) *corev1.PersistentVolume {
	csi := &corev1.CSIPersistentVolumeSource{
		Driver:       "csi.example.com",
		VolumeHandle: pvName,
	}
	if len(secretNames) >= 1 {
		csi.NodeStageSecretRef = &corev1.SecretReference{
			Namespace: "ns",
			Name:      secretNames[0],
		}
	}
	if len(secretNames) >= 2 {
		csi.NodePublishSecretRef = &corev1.SecretReference{
			Namespace: "ns",
			Name:      secretNames[1],
		}
	}
	return &corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: pvName},
		Spec: corev1.PersistentVolumeSpec{
			ClaimRef:               &corev1.ObjectReference{Namespace: "ns", Name: claimName},
			PersistentVolumeSource: corev1.PersistentVolumeSource{CSI: csi},
		},
	}
}

func newHighDegreeSecretAndPVCAuthorizer(pvCount, podsPerPVC, nodeCount int) *NodeAuthorizer {
	g := NewGraph()
	for i := range pvCount {
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(makeBenchmarkPV(fmt.Sprintf("pv-%d", i), claimName, "shared-secret"))
		for j := range podsPerPVC {
			g.AddPod(&corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pod-%d-%d", i, j)},
				Spec: corev1.PodSpec{
					NodeName:           fmt.Sprintf("node-%d", j%nodeCount),
					ServiceAccountName: "default",
					Volumes: []corev1.Volume{{
						Name: "data",
						VolumeSource: corev1.VolumeSource{
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: claimName},
						},
					}},
				},
			})
		}
	}
	g.AddPod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "unrelated-pod"},
		Spec: corev1.PodSpec{
			NodeName:           "unrelated-node",
			ServiceAccountName: "default",
		},
	})
	return &NodeAuthorizer{graph: g}
}

func newPVSecretBenchmarkAuthorizer(pvCount, nodeCount int, shared bool) *NodeAuthorizer {
	g := NewGraph()
	for i := range pvCount {
		secretName := fmt.Sprintf("secret-%d", i/2)
		if shared {
			secretName = "shared-secret"
		}
		claimName := fmt.Sprintf("pvc-%d", i)
		g.AddPV(makeBenchmarkPV(fmt.Sprintf("pv-%d", i), claimName, secretName))
		g.AddPod(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: fmt.Sprintf("pod-%d", i)},
			Spec: corev1.PodSpec{
				NodeName:           fmt.Sprintf("node-%d", i%nodeCount),
				ServiceAccountName: "default",
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
		Spec: corev1.PodSpec{
			NodeName:           "unrelated-node",
			ServiceAccountName: "default",
		},
	})
	return &NodeAuthorizer{graph: g}
}
