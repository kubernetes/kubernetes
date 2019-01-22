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

package vsphere

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
	Test to verify volume remains attached after kubelet restart on master node
	For the number of schedulable nodes,
	1. Create a volume with default volume options
	2. Create a Pod
	3. Verify the volume is attached
	4. Restart the kubelet on master node
	5. Verify again that the volume is attached
	6. Delete the pod and wait for the volume to be detached
	7. Delete the volume
*/
var _ = utils.SIGDescribe("Volume Attach Verify [Feature:vsphere][Serial][Disruptive]", func() {
	f := framework.NewDefaultFramework("restart-master")

	const labelKey = "vsphere_e2e_label"
	var (
		client                clientset.Interface
		namespace             string
		volumePaths           []string
		pods                  []*v1.Pod
		numNodes              int
		nodeKeyValueLabelList []map[string]string
		nodeNameList          []string
		nodeInfo              *NodeInfo
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))

		nodes := framework.GetReadySchedulableNodesOrDie(client)
		numNodes = len(nodes.Items)
		if numNodes < 2 {
			framework.Skipf("Requires at least %d nodes (not %d)", 2, len(nodes.Items))
		}
		nodeInfo = TestContext.NodeMapper.GetNodeInfo(nodes.Items[0].Name)
		for i := 0; i < numNodes; i++ {
			nodeName := nodes.Items[i].Name
			nodeNameList = append(nodeNameList, nodeName)
			nodeLabelValue := "vsphere_e2e_" + string(uuid.NewUUID())
			nodeKeyValueLabel := make(map[string]string)
			nodeKeyValueLabel[labelKey] = nodeLabelValue
			nodeKeyValueLabelList = append(nodeKeyValueLabelList, nodeKeyValueLabel)
			framework.AddOrUpdateLabelOnNode(client, nodeName, labelKey, nodeLabelValue)
		}
	})

	It("verify volume remains attached after master kubelet restart", func() {
		// Create pod on each node
		for i := 0; i < numNodes; i++ {
			By(fmt.Sprintf("%d: Creating a test vsphere volume", i))
			volumePath, err := nodeInfo.VSphere.CreateVolume(&VolumeOptions{}, nodeInfo.DataCenterRef)
			Expect(err).NotTo(HaveOccurred())
			volumePaths = append(volumePaths, volumePath)

			By(fmt.Sprintf("Creating pod %d on node %v", i, nodeNameList[i]))
			podspec := getVSpherePodSpecWithVolumePaths([]string{volumePath}, nodeKeyValueLabelList[i], nil)
			pod, err := client.CoreV1().Pods(namespace).Create(podspec)
			Expect(err).NotTo(HaveOccurred())
			defer framework.DeletePodWithWait(f, client, pod)

			By("Waiting for pod to be ready")
			Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

			pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())

			pods = append(pods, pod)

			nodeName := pod.Spec.NodeName
			By(fmt.Sprintf("Verify volume %s is attached to the node %s", volumePath, nodeName))
			expectVolumeToBeAttached(nodeName, volumePath)
		}

		By("Restarting kubelet on master node")
		masterAddress := framework.GetMasterHost() + ":22"
		err := framework.RestartKubelet(masterAddress)
		Expect(err).NotTo(HaveOccurred(), "Unable to restart kubelet on master node")

		By("Verifying the kubelet on master node is up")
		err = framework.WaitForKubeletUp(masterAddress)
		Expect(err).NotTo(HaveOccurred())

		for i, pod := range pods {
			volumePath := volumePaths[i]
			nodeName := pod.Spec.NodeName

			By(fmt.Sprintf("After master restart, verify volume %v is attached to the node %v", volumePath, nodeName))
			expectVolumeToBeAttached(nodeName, volumePath)

			By(fmt.Sprintf("Deleting pod on node %s", nodeName))
			err = framework.DeletePodWithWait(f, client, pod)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("Waiting for volume %s to be detached from the node %s", volumePath, nodeName))
			err = waitForVSphereDiskToDetach(volumePath, nodeName)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("Deleting volume %s", volumePath))
			err = nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
			Expect(err).NotTo(HaveOccurred())
		}
	})
})
