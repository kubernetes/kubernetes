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

package e2e

import (
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("vsphere-volume-placement", func() {
	f := framework.NewDefaultFramework("vsphere-volume-placement")
	var (
		c                  clientset.Interface
		ns                 string
		volumePath         string
		node1Name          string
		node1LabelValue    string
		node1KeyValueLabel map[string]string

		node2Name          string
		node2LabelValue    string
		node2KeyValueLabel map[string]string

		isNodeLabeled bool
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		if !isNodeLabeled {
			nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			if len(nodeList.Items) != 0 {
				node1Name = nodeList.Items[0].Name
				node2Name = nodeList.Items[1].Name
			} else {
				framework.Failf("Unable to find ready and schedulable Node")
			}
			node1LabelValue = "vsphere_e2e_" + string(uuid.NewUUID())
			node1KeyValueLabel = make(map[string]string)
			node1KeyValueLabel["vsphere_e2e_label"] = node1LabelValue
			framework.AddOrUpdateLabelOnNode(c, node1Name, "vsphere_e2e_label", node1LabelValue)

			node2LabelValue = "vsphere_e2e_" + string(uuid.NewUUID())
			node2KeyValueLabel = make(map[string]string)
			node2KeyValueLabel["vsphere_e2e_label"] = node2LabelValue
			framework.AddOrUpdateLabelOnNode(c, node2Name, "vsphere_e2e_label", node2LabelValue)

		}

	})

	AddCleanupAction(func() {
		By("Running clean up actions")
		if len(node1LabelValue) > 0 {
			framework.RemoveLabelOffNode(c, node1Name, "vsphere_e2e_label")
		}
		if len(node2LabelValue) > 0 {
			framework.RemoveLabelOffNode(c, node2Name, "vsphere_e2e_label")
		}
	})

	/*
		Test Steps
		-----------
		1. Find node with the status available and ready for scheduling.
		2. Add label to the node. - (vsphere_e2e_label: Random UUID)
		3. Create VMDK volume
		4. Create POD Spec with volume path of the vmdk and NodeSelector set to label assigned in the Step 2.
		5. Create POD and wait for POD to become ready.
		6. Verify volume is attached to the node labeled in the Step 2.
		7. Delete POD.
		8. Wait for volume to be detached from the node.
		9. Repeat Step 5 to 7 and make sure back to back pod creation with the same volume is working as expected.
		10. Delete VMDK volume
	*/
	framework.KubeDescribe("Test Back-to-back pod creation/deletion with the same volume source on the same worker node", func() {
		var volumeoptions vsphere.VolumeOptions
		It("should provision pod on the node with matching label", func() {
			By("creating vmdk")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumeoptions.CapacityKB = 2097152
			volumeoptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
			volumeoptions.DiskFormat = "thin"

			volumePath, err = vsp.CreateVolume(&volumeoptions)
			Expect(err).NotTo(HaveOccurred())

			By("Creating pod on the node: " + node1Name)
			pod := getPodSpec(volumePath, node1KeyValueLabel, nil)

			pod, err = c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred())
			By("Waiting for pod to be ready")
			Expect(f.WaitForPodRunning(pod.Name)).To(Succeed())

			By("Verify volume is attached to the node: " + node1Name)
			isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, types.NodeName(node1Name))
			Expect(err).NotTo(HaveOccurred())
			Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

			By("Deleting pod")
			err = c.CoreV1().Pods(ns).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for volume to be detached from the node")
			waitForVSphereDiskToDetach(vsp, volumePath, types.NodeName(node1Name))

			By("Creating pod on the same node: " + node1Name)
			pod = getPodSpec(volumePath, node1KeyValueLabel, nil)
			pod, err = c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for pod to be running")
			Expect(f.WaitForPodRunning(pod.Name)).To(Succeed())

			By("Verify volume is attached to the node: " + node1Name)
			isAttached, err = verifyVSphereDiskAttached(vsp, volumePath, types.NodeName(node1Name))
			Expect(err).NotTo(HaveOccurred())
			Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

			By("Deleting pod")
			err = c.CoreV1().Pods(ns).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for volume to be detached from the node: " + node1Name)
			waitForVSphereDiskToDetach(vsp, volumePath, types.NodeName(node1Name))

			By("Deleting vmdk")
			if len(volumePath) > 0 {
				vsp.DeleteVolume(volumePath)
			}
		})
	})

	/*
		Test Steps
		-----------
		1. Find two nodes with the status available and ready for scheduling.
		2. Add labels to the both nodes. - (vsphere_e2e_label: Random UUID)
		3. Create VMDK volume
		4. Create POD Spec with volume path of the vmdk and NodeSelector set to node1's label.
		5. Create POD and wait for POD to become ready.
		6. Verify volume is attached to the node.
		7. Delete POD.
		8. Wait for volume to be detached from the node1.
		9. Create POD Spec with volume path of the vmdk and NodeSelector set to node2's label.
		10. Create POD and wait for POD to become ready.
		11. Verify volume is attached to the node2.
		12. Delete POD.
		13. Delete VMDK volume.
	*/
	framework.KubeDescribe("Test Back-to-back pod creation/deletion with the same volume source attach/detach to different worker nodes", func() {
		var volumeoptions vsphere.VolumeOptions
		It("should provision pod on the node with matching label", func() {
			By("creating vmdk")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumeoptions.CapacityKB = 2097152
			volumeoptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
			volumeoptions.DiskFormat = "thin"

			volumePath, err = vsp.CreateVolume(&volumeoptions)
			Expect(err).NotTo(HaveOccurred())

			By("Creating pod on the node: " + node1Name)
			pod := getPodSpec(volumePath, node1KeyValueLabel, nil)

			pod, err = c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred())
			By("Waiting for pod to be ready")
			Expect(f.WaitForPodRunning(pod.Name)).To(Succeed())

			By("Verify volume is attached to the node: " + node1Name)
			isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, types.NodeName(node1Name))
			Expect(err).NotTo(HaveOccurred())
			Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

			By("Deleting pod")
			err = c.CoreV1().Pods(ns).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for volume to be detached from the node")
			waitForVSphereDiskToDetach(vsp, volumePath, types.NodeName(node1Name))

			By("Creating pod on the another node: " + node2Name)
			pod = getPodSpec(volumePath, node2KeyValueLabel, nil)
			pod, err = c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for pod to be running")
			Expect(f.WaitForPodRunning(pod.Name)).To(Succeed())

			By("Verify volume is attached to the node: " + node2Name)
			isAttached, err = verifyVSphereDiskAttached(vsp, volumePath, types.NodeName(node2Name))
			Expect(err).NotTo(HaveOccurred())
			Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

			By("Deleting pod")
			err = c.CoreV1().Pods(ns).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for volume to be detached from the node: " + node2Name)
			waitForVSphereDiskToDetach(vsp, volumePath, types.NodeName(node2Name))

			By("Deleting vmdk")
			if len(volumePath) > 0 {
				vsp.DeleteVolume(volumePath)
			}
		})
	})
})

func getPodSpec(volumePath string, keyValuelabel map[string]string, commands []string) *v1.Pod {
	if commands == nil || len(commands) == 0 {
		commands = make([]string, 3)
		commands[0] = "/bin/sh"
		commands[1] = "-c"
		commands[2] = "while true ; do sleep 2 ; done "
	}
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "vsphere-e2e-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "vsphere-e2e-container-" + string(uuid.NewUUID()),
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: commands,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "vsphere-volume",
							MountPath: "/mnt/vsphere-volume",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "vsphere-volume",
					VolumeSource: v1.VolumeSource{
						VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
							VolumePath: volumePath,
							FSType:     "ext4",
						},
					},
				},
			},
		},
	}

	if keyValuelabel != nil {
		pod.Spec.NodeSelector = keyValuelabel
	}
	return pod
}
