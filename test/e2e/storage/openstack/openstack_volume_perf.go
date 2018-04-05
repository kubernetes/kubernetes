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

package openstack

import (
	"fmt"
	"os"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storageV1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	utils "k8s.io/kubernetes/test/e2e/storage/utils"
)

// volume performace constants
const (
	SCSIUnitsAvailablePerNode = 55
	CreateOp                  = "CreateOp"
	AttachOp                  = "AttachOp"
	DetachOp                  = "DetachOp"
	DeleteOp                  = "DeleteOp"
)

var _ = utils.SIGDescribe("osp-performance [Feature:openstack]", func() {
	f := framework.NewDefaultFramework("osp-performance")

	var (
		client           clientset.Interface
		namespace        string
		nodeSelectorList []*NodeSelector
		volumeCount      int
		volumesPerPod    int
		iterations       int
		policyName       string
		datastoreName    string
	)

	BeforeEach(func() {
		var err error
		framework.SkipUnlessProviderIs("openstack")
		client = f.ClientSet
		namespace = f.Namespace.Name

		// Read the environment variables
		volumeCountStr := os.Getenv("OSP_PERF_VOLUME_COUNT")
		Expect(volumeCountStr).NotTo(BeEmpty(), "ENV OSP_PERF_VOLUME_COUNT is not set")
		volumeCount, err = strconv.Atoi(volumeCountStr)
		Expect(err).NotTo(HaveOccurred(), "Error Parsing VCP_PERF_VOLUME_COUNT")

		volumesPerPodStr := os.Getenv("OSP_PERF_VOLUME_PER_POD")
		Expect(volumesPerPodStr).NotTo(BeEmpty(), "ENV OSP_PERF_VOLUME_PER_POD is not set")
		volumesPerPod, err = strconv.Atoi(volumesPerPodStr)
		Expect(err).NotTo(HaveOccurred(), "Error Parsing OSP_PERF_VOLUME_PER_POD")

		iterationsStr := os.Getenv("OSP_PERF_ITERATIONS")
		Expect(iterationsStr).NotTo(BeEmpty(), "ENV OSP_PERF_ITERATIONS is not set")
		iterations, err = strconv.Atoi(iterationsStr)
		Expect(err).NotTo(HaveOccurred(), "Error Parsing OSP_PERF_ITERATIONS")

		policyName = os.Getenv("OPENSTACK_SPBM_GOLD_POLICY")
		datastoreName = os.Getenv("OPENSTACK_DATASTORE")
		Expect(policyName).NotTo(BeEmpty(), "ENV OPENSTACK_SPBM_GOLD_POLICY is not set")
		Expect(datastoreName).NotTo(BeEmpty(), "ENV OPENSTACK_DATASTORE is not set")

		nodes := framework.GetReadySchedulableNodesOrDie(client)
		Expect(len(nodes.Items)).To(BeNumerically(">=", 1), "Requires at least %d nodes (not %d)", 2, len(nodes.Items))

		msg := fmt.Sprintf("Cannot attach %d volumes to %d nodes. Maximum volumes that can be attached on %d nodes is %d", volumeCount, len(nodes.Items), len(nodes.Items), SCSIUnitsAvailablePerNode*len(nodes.Items))
		Expect(volumeCount).To(BeNumerically("<=", SCSIUnitsAvailablePerNode*len(nodes.Items)), msg)

		msg = fmt.Sprintf("Cannot attach %d volumes per pod. Maximum volumes that can be attached per pod is %d", volumesPerPod, SCSIUnitsAvailablePerNode)
		Expect(volumesPerPod).To(BeNumerically("<=", SCSIUnitsAvailablePerNode), msg)

		nodeSelectorList = createNodeLabels(client, namespace, nodes)
	})

	It("osp performance tests", func() {
		scList := getTestStorageClasses(client, policyName, datastoreName)
		defer func(scList []*storageV1.StorageClass) {
			for _, sc := range scList {
				client.StorageV1().StorageClasses().Delete(sc.Name, nil)
			}
		}(scList)

		sumLatency := make(map[string]float64)
		for i := 0; i < iterations; i++ {
			latency := invokeOSVolumeLifeCyclePerformance(f, client, namespace, scList, volumesPerPod, volumeCount, nodeSelectorList)
			for key, val := range latency {
				sumLatency[key] += val
			}
		}

		iterations64 := float64(iterations)
		framework.Logf("Average latency for below operations")
		framework.Logf("Creating %d PVCs and waiting for bound phase: %v seconds", volumeCount, sumLatency[CreateOp]/iterations64)
		framework.Logf("Creating %v Pod: %v seconds", volumeCount/volumesPerPod, sumLatency[AttachOp]/iterations64)
		framework.Logf("Deleting %v Pod and waiting for disk to be detached: %v seconds", volumeCount/volumesPerPod, sumLatency[DetachOp]/iterations64)
		framework.Logf("Deleting %v PVCs: %v seconds", volumeCount, sumLatency[DeleteOp]/iterations64)

	})
})

// invokeVolumeLifeCyclePerformance peforms full volume life cycle management and records latency for each operation
func invokeOSVolumeLifeCyclePerformance(f *framework.Framework, client clientset.Interface, namespace string, sc []*storageV1.StorageClass, volumesPerPod int, volumeCount int, nodeSelectorList []*NodeSelector) (latency map[string]float64) {
	var (
		totalpvclaims [][]*v1.PersistentVolumeClaim
		totalpos      [][]*v1.PersistentVolume
		totalpods     []*v1.Pod
	)
	nodeVolumeMap := make(map[types.NodeName][]string)
	latency = make(map[string]float64)
	numPods := volumeCount / volumesPerPod

	By(fmt.Sprintf("Creating %d PVCs", volumeCount))
	start := time.Now()
	for i := 0; i < numPods; i++ {
		var pvclaims []*v1.PersistentVolumeClaim
		for j := 0; j < volumesPerPod; j++ {
			currsc := sc[((i*numPods)+j)%len(sc)]
			pvclaim, err := framework.CreatePVC(client, namespace, getOpenstackClaimSpecWithStorageClassAnnotation(namespace, "2Gi", currsc))
			Expect(err).NotTo(HaveOccurred())
			pvclaims = append(pvclaims, pvclaim)
		}
		totalpvclaims = append(totalpvclaims, pvclaims)
	}
	for _, pvclaims := range totalpvclaims {
		persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())
		totalpos = append(totalpos, persistentvolumes)
	}
	elapsed := time.Since(start)
	latency[CreateOp] = elapsed.Seconds()

	By("Creating pod to attach PVs to the node")
	start = time.Now()
	for i, pvclaims := range totalpvclaims {
		nodeSelector := nodeSelectorList[i%len(nodeSelectorList)]
		pod, err := framework.CreatePod(client, namespace, map[string]string{nodeSelector.labelKey: nodeSelector.labelValue}, pvclaims, false, "")
		Expect(err).NotTo(HaveOccurred())
		totalpods = append(totalpods, pod)

		defer framework.DeletePodWithWait(f, client, pod)
	}
	elapsed = time.Since(start)
	latency[AttachOp] = elapsed.Seconds()

	// Verify access to the volumes
	osp, id, err := getOpenstack(client)
	Expect(err).NotTo(HaveOccurred())

	for i, pod := range totalpods {
		verifyOpenstackVolumesAccessible(client, pod, totalpos[i], id, osp)
	}

	By("Deleting pods")
	start = time.Now()
	for _, pod := range totalpods {
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())
	}
	elapsed = time.Since(start)
	latency[DetachOp] = elapsed.Seconds()

	for i, pod := range totalpods {
		for _, pv := range totalpos[i] {
			nodeName := types.NodeName(pod.Spec.NodeName)
			nodeVolumeMap[nodeName] = append(nodeVolumeMap[nodeName], pv.Spec.Cinder.VolumeID)
		}
	}

	err = waitForOpenstackDisksToDetach(client, id, osp, nodeVolumeMap)
	Expect(err).NotTo(HaveOccurred())

	By("Deleting the PVCs")
	start = time.Now()
	for _, pvclaims := range totalpvclaims {
		for _, pvc := range pvclaims {
			err = framework.DeletePersistentVolumeClaim(client, pvc.Name, namespace)
			Expect(err).NotTo(HaveOccurred())
		}
	}
	elapsed = time.Since(start)
	latency[DeleteOp] = elapsed.Seconds()

	return latency
}

func createNodeLabels(client clientset.Interface, namespace string, nodes *v1.NodeList) []*NodeSelector {
	var nodeSelectorList []*NodeSelector
	for i, node := range nodes.Items {
		labelVal := "openstack_e2e_" + strconv.Itoa(i)
		nodeSelector := &NodeSelector{
			labelKey:   NodeLabelKey,
			labelValue: labelVal,
		}
		nodeSelectorList = append(nodeSelectorList, nodeSelector)
		framework.AddOrUpdateLabelOnNode(client, node.Name, NodeLabelKey, labelVal)
	}
	return nodeSelectorList
}
