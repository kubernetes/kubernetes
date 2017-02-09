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

var _ = framework.KubeDescribe("Volume Placement [Feature:Volume]", func() {
	f := framework.NewDefaultFramework("volume-placement")
	var (
		c                  clientset.Interface
		ns                 string
		vsp                *vsphere.VSphere
		volumePath         string
		node1Name          string
		node1LabelValue    string
		node1KeyValueLabel map[string]string

		node2Name          string
		node2LabelValue    string
		node2KeyValueLabel map[string]string

		isNodeLabeled bool
	)

	/*
		Steps
		1. Create VMDK volume
		2. Find two nodes with the status available and ready for scheduling.
		3. Add labels to the both nodes. - (vsphere_e2e_label: Random UUID)

	*/

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		By("creating vmdk")
		vsp, err := vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())

		volumePath, err = createVSphereVolume(vsp, nil)
		Expect(err).NotTo(HaveOccurred())

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

	/*
		Steps
		1. Remove labels assigned to node 1 and node 2
		2. Delete VMDK volume
	*/

	AddCleanupAction(func() {
		if len(node1LabelValue) > 0 {
			framework.RemoveLabelOffNode(c, node1Name, "vsphere_e2e_label")
		}
		if len(node2LabelValue) > 0 {
			framework.RemoveLabelOffNode(c, node2Name, "vsphere_e2e_label")
		}
		if len(volumePath) > 0 {
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())
			vsp.DeleteVolume(volumePath)
		}
	})

	framework.KubeDescribe("provision pod on node with matching labels", func() {

		/*
			Steps

			1. Create POD Spec with volume path of the vmdk and NodeSelector set to label assigned to node1.
			2. Create POD and wait for POD to become ready.
			3. Verify volume is attached to the node1.
			4. Delete POD.
			5. Wait for volume to be detached from the node1.
			6. Repeat Step 1 to 5 and make sure back to back pod creation on same worker node with the same volume is working as expected.

		*/

		It("should create and delete pod with the same volume source on the same worker node", func() {
			pod := createPodWithVolumeAndNodeSelector(c, ns, vsp, node1Name, node1KeyValueLabel, volumePath)
			deletePodAndWaitForVolumeToDetach(c, ns, vsp, node1Name, pod, volumePath)

			By("Creating pod on the same node: " + node1Name)
			pod = createPodWithVolumeAndNodeSelector(c, ns, vsp, node1Name, node1KeyValueLabel, volumePath)
			deletePodAndWaitForVolumeToDetach(c, ns, vsp, node1Name, pod, volumePath)
		})

		/*
			Steps

			1. Create POD Spec with volume path of the vmdk and NodeSelector set to node1's label.
			2. Create POD and wait for POD to become ready.
			3. Verify volume is attached to the node1.
			4. Delete POD.
			5. Wait for volume to be detached from the node1.
			6. Create POD Spec with volume path of the vmdk and NodeSelector set to node2's label.
			7. Create POD and wait for POD to become ready.
			8. Verify volume is attached to the node2.
			9. Delete POD.
		*/

		It("should create and delete pod with the same volume source attach/detach to different worker nodes", func() {
			pod := createPodWithVolumeAndNodeSelector(c, ns, vsp, node1Name, node1KeyValueLabel, volumePath)
			deletePodAndWaitForVolumeToDetach(c, ns, vsp, node1Name, pod, volumePath)

			By("Creating pod on the another node: " + node2Name)
			pod = createPodWithVolumeAndNodeSelector(c, ns, vsp, node2Name, node2KeyValueLabel, volumePath)
			deletePodAndWaitForVolumeToDetach(c, ns, vsp, node2Name, pod, volumePath)
		})

	})
})

func createPodWithVolumeAndNodeSelector(client clientset.Interface, namespace string, vsp *vsphere.VSphere, nodeName string, nodeKeyValueLabel map[string]string, volumePath string) *v1.Pod {
	var pod *v1.Pod
	var err error
	By("Creating pod on the node: " + nodeName)
	podspec := getPodSpec(volumePath, nodeKeyValueLabel, nil)

	pod, err = client.CoreV1().Pods(namespace).Create(podspec)
	Expect(err).NotTo(HaveOccurred())
	By("Waiting for pod to be ready")
	Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

	By("Verify volume is attached to the node: " + nodeName)
	isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, types.NodeName(nodeName))
	Expect(err).NotTo(HaveOccurred())
	Expect(isAttached).To(BeTrue(), "disk is not attached with the node")
	return pod
}
func deletePodAndWaitForVolumeToDetach(client clientset.Interface, namespace string, vsp *vsphere.VSphere, nodeName string, pod *v1.Pod, volumePath string) {
	var err error
	By("Deleting pod")
	err = client.CoreV1().Pods(namespace).Delete(pod.Name, nil)
	Expect(err).NotTo(HaveOccurred())

	By("Waiting for volume to be detached from the node")
	waitForVSphereDiskToDetach(vsp, volumePath, types.NodeName(nodeName))
}

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
