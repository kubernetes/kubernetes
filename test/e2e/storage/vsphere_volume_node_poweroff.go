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

package storage

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

/*
	Test to verify volume status after node power off:
	1. Verify the pod got provisioned on a different node with volume attached to it
	2. Verify the volume is detached from the powered off node
*/
var _ = SIGDescribe("Node Poweroff [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("node-poweroff")
	var (
		client    clientset.Interface
		namespace string
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))
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
		By("Creating s Storage Class")
		//scParameters := make(map[string]string)
		//scParameters["diskformat"] = "thin"
		storageClassSpec := getVSphereStorageClassSpec("test-sc", nil)
		storageclass, err := client.StorageV1().StorageClasses().Create(storageClassSpec)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create storage class with err: %v", err))
		defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

		By("Creating PVC using the Storage Class")
		pvclaimSpec := getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass)
		//pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(pvclaimSpec)
		pvclaim, err := framework.CreatePVC(client, namespace, pvclaimSpec)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create PVC with err: %v", err))
		//defer client.CoreV1().PersistentVolumeClaims(namespace).Delete(pvclaimSpec.Name, nil)
		defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

		By("Waiting for claim to be in bound phase")
		pvclaims := []*v1.PersistentVolumeClaim{pvclaim}
		pvs, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to wait until pvc phase set to bound: %v", err))
		volumePath := pvs[0].Spec.VsphereVolume.VolumePath

		By("Creating a Deployment")
		deployment, err := framework.CreateDeployment(client, int32(1), map[string]string{"test": "app"}, namespace, pvclaims, "")
		Expect(err).NotTo(HaveOccurred())
		defer client.Extensions().Deployments(namespace).Delete(deployment.Name, &metav1.DeleteOptions{})

		By("Get pod from the deployement")
		podList, err := framework.GetPodsForDeployment(client, deployment)
		clientPod := podList.Items[0]
		node1 := types.NodeName(clientPod.Spec.NodeName)

		By("Verify disk is attached to the node")
		vsp, err := vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
		isAttached, err := vsp.DiskIsAttached(volumePath, node1)
		//isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, node1)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "Disk is not attached with the node")

		// By("Power off the node where pod got provisioned")
		// govMoMiClient, err := vsphere.GetgovmomiClient(nil)
		// Expect(err).NotTo(HaveOccurred())

		// f := find.NewFinder(govMoMiClient.Client, true)
		// ctx, _ := context.WithCancel(context.Background())

		// workingDir := os.Getenv("VSPHERE_WORKING_DIR")
		// Expect(workingDir).NotTo(BeEmpty())

		// vmPath := workingDir + string(node1)
		// vm, err := f.VirtualMachine(ctx, vmPath)
		// Expect(err).NotTo(HaveOccurred())

		// _, err = vm.PowerOff(ctx)
		// Expect(err).NotTo(HaveOccurred())

		// err = vm.WaitForPowerState(ctx, vimtypes.VirtualMachinePowerStatePoweredOff)
		// Expect(err).NotTo(HaveOccurred(), "Unable to power off the node")

		// By("Get pod from the deployement") // Is this correct?
		// podList, err = framework.GetPodsForDeployment(client, deployment)
		// clientPod = podList.Items[0]
		// node2 := types.NodeName(clientPod.Spec.NodeName)

		// By(fmt.Sprintf("Verify disk is attached to the current node: %v", node2))
		// isAttached, err = verifyVSphereDiskAttached(vsp, volumePath, node2)
		// Expect(err).NotTo(HaveOccurred())
		// Expect(isAttached).To(BeTrue(), "Disk is not attached with the node")

		// By(fmt.Sprintf("Waiting for disk to be detached from the previous node: %v", node1))
		// err = waitForVSphereDiskToDetach(vsp, volumePath, node1)
		// Expect(err).NotTo(HaveOccurred(), "Disk is not detached from the node")

		// By("Power on the previous node")
		// vm.PowerOn(ctx)
		// err = vm.WaitForPowerState(ctx, vimtypes.VirtualMachinePowerStatePoweredOn)
		// Expect(err).NotTo(HaveOccurred(), "Unable to power on the node")
	})
})
