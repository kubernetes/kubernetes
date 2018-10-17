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
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/govmomi/object"
	vimtypes "github.com/vmware/govmomi/vim25/types"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
	Test to verify volume status after node power off:
	1. Verify the pod got provisioned on a different node with volume attached to it
	2. Verify the volume is detached from the powered off node
*/
var _ = utils.SIGDescribe("Node Poweroff [Feature:vsphere] [Slow] [Disruptive]", func() {
	f := framework.NewDefaultFramework("node-poweroff")
	var (
		client    clientset.Interface
		namespace string
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(nodeList.Items).NotTo(BeEmpty(), "Unable to find ready and schedulable Node")
		Expect(len(nodeList.Items) > 1).To(BeTrue(), "At least 2 nodes are required for this test")
	})

	/*
		Steps:
		1. Create a StorageClass
		2. Create a PVC with the StorageClass
		3. Create a Deployment with 1 replica, using the PVC
		4. Verify the pod got provisioned on a node
		5. Verify the volume is attached to the node
		6. Power off the node where pod got provisioned
		7. Verify the pod got provisioned on a different node
		8. Verify the volume is attached to the new node
		9. Verify the volume is detached from the old node
		10. Delete the Deployment and wait for the volume to be detached
		11. Delete the PVC
		12. Delete the StorageClass
	*/
	It("verify volume status after node power off", func() {
		By("Creating a Storage Class")
		storageClassSpec := getVSphereStorageClassSpec("test-sc", nil)
		storageclass, err := client.StorageV1().StorageClasses().Create(storageClassSpec)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create storage class with err: %v", err))
		defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

		By("Creating PVC using the Storage Class")
		pvclaimSpec := getVSphereClaimSpecWithStorageClass(namespace, "1Gi", storageclass)
		pvclaim, err := framework.CreatePVC(client, namespace, pvclaimSpec)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create PVC with err: %v", err))
		defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

		By("Waiting for PVC to be in bound phase")
		pvclaims := []*v1.PersistentVolumeClaim{pvclaim}
		pvs, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to wait until PVC phase set to bound: %v", err))
		volumePath := pvs[0].Spec.VsphereVolume.VolumePath

		By("Creating a Deployment")
		deployment, err := framework.CreateDeployment(client, int32(1), map[string]string{"test": "app"}, nil, namespace, pvclaims, "")
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create Deployment with err: %v", err))
		defer client.ExtensionsV1beta1().Deployments(namespace).Delete(deployment.Name, &metav1.DeleteOptions{})

		By("Get pod from the deployement")
		podList, err := framework.GetPodsForDeployment(client, deployment)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to get pod from the deployement with err: %v", err))
		Expect(podList.Items).NotTo(BeEmpty())
		pod := podList.Items[0]
		node1 := pod.Spec.NodeName

		By(fmt.Sprintf("Verify disk is attached to the node: %v", node1))
		isAttached, err := diskIsAttached(volumePath, node1)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "Disk is not attached to the node")

		By(fmt.Sprintf("Power off the node: %v", node1))

		nodeInfo := TestContext.NodeMapper.GetNodeInfo(node1)
		vm := object.NewVirtualMachine(nodeInfo.VSphere.Client.Client, nodeInfo.VirtualMachineRef)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		_, err = vm.PowerOff(ctx)
		Expect(err).NotTo(HaveOccurred())
		defer vm.PowerOn(ctx)

		err = vm.WaitForPowerState(ctx, vimtypes.VirtualMachinePowerStatePoweredOff)
		Expect(err).NotTo(HaveOccurred(), "Unable to power off the node")

		// Waiting for the pod to be failed over to a different node
		node2, err := waitForPodToFailover(client, deployment, node1)
		Expect(err).NotTo(HaveOccurred(), "Pod did not fail over to a different node")

		By(fmt.Sprintf("Waiting for disk to be attached to the new node: %v", node2))
		err = waitForVSphereDiskToAttach(volumePath, node2)
		Expect(err).NotTo(HaveOccurred(), "Disk is not attached to the node")

		By(fmt.Sprintf("Waiting for disk to be detached from the previous node: %v", node1))
		err = waitForVSphereDiskToDetach(volumePath, node1)
		Expect(err).NotTo(HaveOccurred(), "Disk is not detached from the node")

		By(fmt.Sprintf("Power on the previous node: %v", node1))
		vm.PowerOn(ctx)
		err = vm.WaitForPowerState(ctx, vimtypes.VirtualMachinePowerStatePoweredOn)
		Expect(err).NotTo(HaveOccurred(), "Unable to power on the node")
	})
})

// Wait until the pod failed over to a different node, or time out after 3 minutes
func waitForPodToFailover(client clientset.Interface, deployment *apps.Deployment, oldNode string) (string, error) {
	var (
		err      error
		newNode  string
		timeout  = 3 * time.Minute
		pollTime = 10 * time.Second
	)

	err = wait.Poll(pollTime, timeout, func() (bool, error) {
		newNode, err = getNodeForDeployment(client, deployment)
		if err != nil {
			return true, err
		}

		if newNode != oldNode {
			framework.Logf("The pod has been failed over from %q to %q", oldNode, newNode)
			return true, nil
		}

		framework.Logf("Waiting for pod to be failed over from %q", oldNode)
		return false, nil
	})

	if err != nil {
		if err == wait.ErrWaitTimeout {
			framework.Logf("Time out after waiting for %v", timeout)
		}
		framework.Logf("Pod did not fail over from %q with error: %v", oldNode, err)
		return "", err
	}

	return getNodeForDeployment(client, deployment)
}

// getNodeForDeployment returns node name for the Deployment
func getNodeForDeployment(client clientset.Interface, deployment *apps.Deployment) (string, error) {
	podList, err := framework.GetPodsForDeployment(client, deployment)
	if err != nil {
		return "", err
	}
	return podList.Items[0].Spec.NodeName, nil
}
