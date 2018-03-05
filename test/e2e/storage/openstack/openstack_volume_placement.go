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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	openstack "k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Volume Placement", func() {
	f := framework.NewDefaultFramework("volume-placement")
	var (
		c                  clientset.Interface
		ns                 string
		osp                *openstack.OpenStack
		volumeIDs          []string
		id                 string
		node1Name          string
		node1KeyValueLabel map[string]string
		node2Name          string
		node2KeyValueLabel map[string]string
		isNodeLabeled      bool
		err                error
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		if !isNodeLabeled {
			node1Name, node1KeyValueLabel, node2Name, node2KeyValueLabel = testSetupVolumePlacement(c, ns)
			isNodeLabeled = true
		}
		By("creating vmdk")
		osp, id, err = getOpenstack(c)
		Expect(err).NotTo(HaveOccurred())
		volumeID, err := createOpenstackVolume(osp)
		Expect(err).NotTo(HaveOccurred())
		volumeIDs = append(volumeIDs, volumeID)
	})

	AfterEach(func() {
		for _, volumeID := range volumeIDs {
			osp.DeleteVolume(volumeID)
		}
		volumeIDs = nil
	})

	framework.AddCleanupAction(func() {
		// Cleanup actions will be called even when the tests are skipped and leaves namespace unset.
		if len(ns) > 0 {
			if len(node1KeyValueLabel) > 0 {
				framework.RemoveLabelOffNode(c, node1Name, "openstack_e2e_label")
			}
			if len(node2KeyValueLabel) > 0 {
				framework.RemoveLabelOffNode(c, node2Name, "openstack_e2e_label")
			}
		}
	})

	It("should create and delete pod with the same volume source on the same worker node", func() {
		var volumeFiles []string
		pod := createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)

		newEmptyFileName := fmt.Sprintf("/mnt/volume1/%v_1.txt", ns)
		volumeFiles = append(volumeFiles, newEmptyFileName)
		createAndVerifyFilesOnVolume(ns, pod.Name, []string{newEmptyFileName}, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)

		By(fmt.Sprintf("Creating pod on the same node: %v", node1Name))
		pod = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)

		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		newEmptyFileName = fmt.Sprintf("/mnt/volume1/%v_2.txt", ns)
		volumeFiles = append(volumeFiles, newEmptyFileName)
		createAndVerifyFilesOnVolume(ns, pod.Name, []string{newEmptyFileName}, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)
	})

	It("should create and delete pod with the same volume source attach/detach to different worker nodes", func() {
		var volumeFiles []string
		pod := createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)
		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		newEmptyFileName := fmt.Sprintf("/mnt/volume1/%v_1.txt", ns)
		volumeFiles = append(volumeFiles, newEmptyFileName)
		createAndVerifyFilesOnVolume(ns, pod.Name, []string{newEmptyFileName}, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)

		By(fmt.Sprintf("Creating pod on the another node: %v", node2Name))
		pod = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node2Name, node2KeyValueLabel, volumeIDs)

		newEmptyFileName = fmt.Sprintf("/mnt/volume1/%v_2.txt", ns)
		volumeFiles = append(volumeFiles, newEmptyFileName)
		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		createAndVerifyFilesOnVolume(ns, pod.Name, []string{newEmptyFileName}, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node2Name, volumeIDs)
	})

	It("should create and delete pod with multiple volumes from same datastore", func() {
		By("creating another vmdk")
		volumeID, err := createOpenstackVolume(osp)
		Expect(err).NotTo(HaveOccurred())
		volumeIDs = append(volumeIDs, volumeID)

		By(fmt.Sprintf("Creating pod on the node: %v with volume: %v and volume: %v", node1Name, volumeIDs[0], volumeIDs[1]))
		pod := createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)
		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		volumeFiles := []string{
			fmt.Sprintf("/mnt/volume1/%v_1.txt", ns),
			fmt.Sprintf("/mnt/volume2/%v_1.txt", ns),
		}
		createAndVerifyFilesOnVolume(ns, pod.Name, volumeFiles, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)
		By(fmt.Sprintf("Creating pod on the node: %v with volume :%v and volume: %v", node1Name, volumeIDs[0], volumeIDs[1]))
		pod = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)
		newEmptyFilesNames := []string{
			fmt.Sprintf("/mnt/volume1/%v_2.txt", ns),
			fmt.Sprintf("/mnt/volume2/%v_2.txt", ns),
		}
		volumeFiles = append(volumeFiles, newEmptyFilesNames[0])
		volumeFiles = append(volumeFiles, newEmptyFilesNames[1])
		createAndVerifyFilesOnVolume(ns, pod.Name, newEmptyFilesNames, volumeFiles)
	})

	It("should create and delete pod with multiple volumes from different datastore", func() {
		By("creating another vmdk on non default shared datastore")
		volumeID, err := createOpenstackVolume(osp)
		Expect(err).NotTo(HaveOccurred())
		volumeIDs = append(volumeIDs, volumeID)

		By(fmt.Sprintf("Creating pod on the node: %v with volume :%v  and volume: %v", node1Name, volumeIDs[0], volumeIDs[1]))
		pod := createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)

		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		volumeFiles := []string{
			fmt.Sprintf("/mnt/volume1/%v_1.txt", ns),
			fmt.Sprintf("/mnt/volume2/%v_1.txt", ns),
		}
		createAndVerifyFilesOnVolume(ns, pod.Name, volumeFiles, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)

		By(fmt.Sprintf("Creating pod on the node: %v with volume :%v  and volume: %v", node1Name, volumeIDs[0], volumeIDs[1]))
		pod = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, volumeIDs)
		// Create empty files on the mounted volumes on the pod to verify volume is writable
		// Verify newly and previously created files present on the volume mounted on the pod
		newEmptyFileNames := []string{
			fmt.Sprintf("/mnt/volume1/%v_2.txt", ns),
			fmt.Sprintf("/mnt/volume2/%v_2.txt", ns),
		}
		volumeFiles = append(volumeFiles, newEmptyFileNames[0])
		volumeFiles = append(volumeFiles, newEmptyFileNames[1])
		createAndVerifyFilesOnVolume(ns, pod.Name, newEmptyFileNames, volumeFiles)
		deletePodAndWaitForVolumeToDetach(f, c, pod, id, osp, node1Name, volumeIDs)
	})

	It("test back to back pod creation and deletion with different volume sources on the same worker node", func() {
		var (
			podA              *v1.Pod
			podB              *v1.Pod
			testvolumeIDsPodA []string
			testvolumeIDsPodB []string
			podAFiles         []string
			podBFiles         []string
		)

		defer func() {
			By("clean up undeleted pods")
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, podA), "defer: Failed to delete pod ", podA.Name)
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, podB), "defer: Failed to delete pod ", podB.Name)
			By(fmt.Sprintf("wait for volumes to be detached from the node: %v", node1Name))
			for _, volumeID := range volumeIDs {
				framework.ExpectNoError(waitForOpenstackDiskToDetach(c, id, osp, volumeID, types.NodeName(node1Name)))
			}
		}()

		testvolumeIDsPodA = append(testvolumeIDsPodA, volumeIDs[0])
		// Create another VMDK Volume
		By("creating another vmdk")
		volumeID, err := createOpenstackVolume(osp)
		Expect(err).NotTo(HaveOccurred())
		volumeIDs = append(volumeIDs, volumeID)
		testvolumeIDsPodB = append(testvolumeIDsPodA, volumeID)

		for index := 0; index < 5; index++ {
			By(fmt.Sprintf("Creating pod-A on the node: %v with volume: %v", node1Name, testvolumeIDsPodA[0]))
			podA = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, testvolumeIDsPodA)

			By(fmt.Sprintf("Creating pod-B on the node: %v with volume: %v", node1Name, testvolumeIDsPodB[0]))
			podB = createPodWithVolumeAndNodeSelector(c, ns, id, osp, node1Name, node1KeyValueLabel, testvolumeIDsPodB)

			podAFileName := fmt.Sprintf("/mnt/volume1/podA_%v_%v.txt", ns, index+1)
			podBFileName := fmt.Sprintf("/mnt/volume1/podB_%v_%v.txt", ns, index+1)
			podAFiles = append(podAFiles, podAFileName)
			podBFiles = append(podBFiles, podBFileName)

			// Create empty files on the mounted volumes on the pod to verify volume is writable
			By("Creating empty file on volume mounted on pod-A")
			framework.CreateEmptyFileOnPod(ns, podA.Name, podAFileName)

			By("Creating empty file volume mounted on pod-B")
			framework.CreateEmptyFileOnPod(ns, podB.Name, podBFileName)

			// Verify newly and previously created files present on the volume mounted on the pod
			By("Verify newly Created file and previously created files present on volume mounted on pod-A")
			verifyFilesExistOnOpenstackVolume(ns, podA.Name, podAFiles)
			By("Verify newly Created file and previously created files present on volume mounted on pod-B")
			verifyFilesExistOnOpenstackVolume(ns, podB.Name, podBFiles)

			By("Deleting pod-A")
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, podA), "Failed to delete pod ", podA.Name)
			By("Deleting pod-B")
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, podB), "Failed to delete pod ", podB.Name)
		}
	})
})

func testSetupVolumePlacement(client clientset.Interface, namespace string) (node1Name string, node1KeyValueLabel map[string]string, node2Name string, node2KeyValueLabel map[string]string) {
	nodes := framework.GetReadySchedulableNodesOrDie(client)
	if len(nodes.Items) < 2 {
		framework.Skipf("Requires at least %d nodes (not %d)", 2, len(nodes.Items))
	}
	node1Name = nodes.Items[0].Name
	node2Name = nodes.Items[1].Name
	node1LabelValue := "openstack_e2e_" + string(uuid.NewUUID())
	node1KeyValueLabel = make(map[string]string)
	node1KeyValueLabel["openstack_e2e_label"] = node1LabelValue
	framework.AddOrUpdateLabelOnNode(client, node1Name, "openstack_e2e_label", node1LabelValue)

	node2LabelValue := "openstack_e2e_" + string(uuid.NewUUID())
	node2KeyValueLabel = make(map[string]string)
	node2KeyValueLabel["openstack_e2e_label"] = node2LabelValue
	framework.AddOrUpdateLabelOnNode(client, node2Name, "openstack_e2e_label", node2LabelValue)
	return node1Name, node1KeyValueLabel, node2Name, node2KeyValueLabel
}

func createPodWithVolumeAndNodeSelector(client clientset.Interface, namespace string, id string, osp *openstack.OpenStack, nodeName string, nodeKeyValueLabel map[string]string, volumeIDs []string) *v1.Pod {
	var pod *v1.Pod
	var err error
	By(fmt.Sprintf("Creating pod on the node: %v", nodeName))
	podspec := getOpenstackPodSpecWithVolumeIDs(volumeIDs, nodeKeyValueLabel, nil)

	pod, err = client.CoreV1().Pods(namespace).Create(podspec)
	Expect(err).NotTo(HaveOccurred())
	By("Waiting for pod to be ready")
	Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

	By(fmt.Sprintf("Verify volume is attached to the node:%v", nodeName))
	for _, volumeID := range volumeIDs {
		isAttached, err := verifyOpenstackDiskAttached(client, osp, id, volumeID, types.NodeName(nodeName))
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "disk:"+volumeID+" is not attached with the node")
	}
	return pod
}

func createAndVerifyFilesOnVolume(namespace string, podname string, newEmptyfilesToCreate []string, filesToCheck []string) {
	// Create empty files on the mounted volumes on the pod to verify volume is writable
	By(fmt.Sprintf("Creating empty file on volume mounted on: %v", podname))
	createEmptyFilesOnOpenstackVolume(namespace, podname, newEmptyfilesToCreate)

	// Verify newly and previously created files present on the volume mounted on the pod
	By(fmt.Sprintf("Verify newly Created file and previously created files present on volume mounted on: %v", podname))
	verifyFilesExistOnOpenstackVolume(namespace, podname, filesToCheck)
}

func deletePodAndWaitForVolumeToDetach(f *framework.Framework, c clientset.Interface, pod *v1.Pod, id string, osp *openstack.OpenStack, nodeName string, volumeIDs []string) {
	By("Deleting pod")
	framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod), "Failed to delete pod ", pod.Name)

	By("Waiting for volume to be detached from the node")
	for _, volumeID := range volumeIDs {
		framework.ExpectNoError(waitForOpenstackDiskToDetach(c, id, osp, volumeID, types.NodeName(nodeName)))
	}
}
