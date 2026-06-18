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

package csimock

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	e2efeature "k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume storage capacity scoring", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-capacity-scoring")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	// createCSIStorageCapacity creates a CSIStorageCapacity object for a specific node.
	createCSIStorageCapacity := func(ctx context.Context, cs clientset.Interface, ns, scName, nodeName string, capacityStr string) *storagev1.CSIStorageCapacity {
		capacityQuantity := resource.MustParse(capacityStr)
		capacity := &storagev1.CSIStorageCapacity{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "fake-capacity-",
			},
			StorageClassName: scName,
			NodeTopology: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					v1.LabelHostname: nodeName,
				},
			},
			Capacity: &capacityQuantity,
		}
		createdCapacity, err := cs.StorageV1().CSIStorageCapacities(ns).Create(ctx, capacity, metav1.CreateOptions{})
		framework.ExpectNoError(err, "create CSIStorageCapacity for node %s", nodeName)
		ginkgo.DeferCleanup(framework.IgnoreNotFound(cs.StorageV1().CSIStorageCapacities(ns).Delete), createdCapacity.Name, metav1.DeleteOptions{})
		return createdCapacity
	}

	// These tests cover StorageCapacityScoring with dynamic provisioning only.
	//
	// The mock CSI driver is deployed on one node. To keep the tests deterministic,
	// capacities are configured so that scoring prefers that node.
	ginkgo.Context("storage capacity scoring", e2efeature.StorageCapacityScoring, func() {
		var (
			yes = true
		)

		verifyPodScheduledToNode := func(ctx context.Context, pod *v1.Pod, claim *v1.PersistentVolumeClaim, expectedNodeName, expectedMessage string) {
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
			framework.ExpectNoError(err, "pod should be running")

			pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get pod")
			gomega.Expect(pod.Spec.NodeName).To(gomega.Equal(expectedNodeName), expectedMessage)

			claim, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Get(ctx, claim.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get PVC")
			gomega.Expect(claim.Annotations).To(gomega.HaveKeyWithValue("volume.kubernetes.io/selected-node", expectedNodeName))
		}

		tests := []struct {
			name                string
			scNamePrefix        string
			minNodes            int
			driverCapacity      string
			otherNodeCapacity   string
			allNodesCapacity    string
			expectUnschedulable bool
			recreatePod         bool
			expectedNodeMessage string
		}{
			{
				name:                "should prefer a node with the largest available space by default",
				scNamePrefix:        "mock-csi-capacity-scoring-max-",
				minNodes:            2,
				driverCapacity:      "100Gi",
				otherNodeCapacity:   "10Gi",
				expectedNodeMessage: "pod should be scheduled to the node with the largest available space",
			},
			{
				name:                "should prefer a node with the maximum allocatable when configured",
				scNamePrefix:        "mock-csi-capacity-scoring-max-explicit-",
				minNodes:            2,
				driverCapacity:      "200Gi",
				otherNodeCapacity:   "20Gi",
				expectedNodeMessage: "pod should be scheduled to the node with the maximum allocatable",
			},
			{
				name:                "should fail to place pod if no node meets the requested size",
				scNamePrefix:        "mock-csi-capacity-scoring-nospace-",
				allNodesCapacity:    "1Mi",
				expectUnschedulable: true,
			},
			{
				name:                "should schedule the recreated pod to the same expected node",
				scNamePrefix:        "mock-csi-capacity-scoring-recreate-",
				minNodes:            2,
				driverCapacity:      "100Gi",
				otherNodeCapacity:   "10Gi",
				recreatePod:         true,
				expectedNodeMessage: "pod should be scheduled to the node with the largest available space",
			},
		}

		for _, t := range tests {
			test := t
			f.It(test.name,
				framework.WithFeatureGate(features.StorageCapacityScoring),
				func(ctx context.Context) {
					nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
					framework.ExpectNoError(err, "get schedulable nodes")
					if len(nodes.Items) < test.minNodes {
						ginkgo.Skip(fmt.Sprintf("need at least %d schedulable nodes", test.minNodes))
					}

					scName := test.scNamePrefix + f.UniqueName
					m.init(ctx, testParameters{
						registerDriver:  true,
						scName:          scName,
						storageCapacity: &yes,
						lateBinding:     true,
						enableTopology:  true,
						disableAttach:   true,
					})
					ginkgo.DeferCleanup(m.cleanup)

					driverNodeName := m.config.ClientNodeSelection.Name
					for _, node := range nodes.Items {
						capacity := test.allNodesCapacity
						if capacity == "" {
							if node.Name == driverNodeName {
								capacity = test.driverCapacity
							} else {
								capacity = test.otherNodeCapacity
							}
						}
						createCSIStorageCapacity(ctx, f.ClientSet, f.Namespace.Name, scName, node.Name, capacity)
					}

					// Allow the scheduler to sync CSIStorageCapacity objects.
					syncDelay := 5 * time.Second
					time.Sleep(syncDelay)

					sc, claim, pod := m.createPod(ctx, pvcReference)
					gomega.Expect(sc.Name).To(gomega.Equal(scName), "pre-selected storage class name not used")

					if test.expectUnschedulable {
						err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
						framework.ExpectNoError(err, "pod should be unschedulable due to insufficient storage capacity")
						return
					}

					verifyPodScheduledToNode(ctx, pod, claim, driverNodeName, test.expectedNodeMessage)

					if !test.recreatePod {
						return
					}

					ginkgo.By("Deleting the first pod")
					err = e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
					framework.ExpectNoError(err, "delete first pod")
					// Remove from cleanup lists to avoid double cleanup.
					m.pods = m.pods[:len(m.pods)-1]

					ginkgo.By("Deleting the first PVC")
					err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
					framework.ExpectNoError(err, "delete first PVC")
					// Remove from cleanup lists to avoid double cleanup.
					m.pvcs = m.pvcs[:len(m.pvcs)-1]

					ginkgo.By("Creating a second pod with a new PVC")
					sc, claim, pod = m.createPod(ctx, pvcReference)
					gomega.Expect(sc.Name).To(gomega.Equal(scName), "pre-selected storage class name not used")

					verifyPodScheduledToNode(ctx, pod, claim, driverNodeName, "recreated pod should be scheduled to the same expected node")
				})
		}
	})

})
