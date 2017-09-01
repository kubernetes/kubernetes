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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	k8stype "k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

/*
	Test to verify fstype specified in storage-class is being honored after volume creation.

	Steps
	1. Create StorageClass with fstype set to valid type (default case included).
	2. Create PVC which uses the StorageClass created in step 1.
	3. Wait for PV to be provisioned.
	4. Wait for PVC's status to become Bound.
	5. Create pod using PVC on specific node.
	6. Wait for Disk to be attached to the node.
	7. Execute command in the pod to get fstype.
	8. Delete pod and Wait for Volume Disk to be detached from the Node.
	9. Delete PVC, PV and Storage Class.
*/

var _ = SIGDescribe("vsphere Volume fstype", func() {
	f := framework.NewDefaultFramework("volume-fstype")
	var (
		client    clientset.Interface
		namespace string
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodeList.Items)).NotTo(BeZero(), "Unable to find ready and schedulable Node")
	})

	It("verify fstype - ext3 formatted volume", func() {
		By("Invoking Test for fstype: ext3")
		invokeTestForFstype(f, client, namespace, "ext3", "ext3")
	})

	It("verify disk format type - default value should be ext4", func() {
		By("Invoking Test for fstype: Default Value")
		invokeTestForFstype(f, client, namespace, "", "ext4")
	})
})

func invokeTestForFstype(f *framework.Framework, client clientset.Interface, namespace string, fstype string, expectedContent string) {
	framework.Logf("Invoking Test for fstype: %s", fstype)
	scParameters := make(map[string]string)
	scParameters["fstype"] = fstype

	By("Creating Storage Class With Fstype")
	storageclass, err := client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec("fstype", scParameters))
	Expect(err).NotTo(HaveOccurred())
	defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

	By("Creating PVC using the Storage Class")
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass))
	Expect(err).NotTo(HaveOccurred())
	defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	By("Waiting for claim to be in bound phase")
	persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims)
	Expect(err).NotTo(HaveOccurred())

	By("Creating pod to attach PV to the node")
	// Create pod to attach Volume to Node
	pod, err := framework.CreatePod(client, namespace, pvclaims, false, "")
	Expect(err).NotTo(HaveOccurred())

	// Asserts: Right disk is attached to the pod
	vsp, err := vsphere.GetVSphere()
	Expect(err).NotTo(HaveOccurred())
	By("Verify the volume is accessible and available in the pod")
	verifyVSphereVolumesAccessible(pod, persistentvolumes, vsp)

	_, err = framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/cat", "/mnt/test/fstype"}, expectedContent, time.Minute)
	Expect(err).NotTo(HaveOccurred())

	By("Deleting pod")
	framework.DeletePodWithWait(f, client, pod)

	By("Waiting for volumes to be detached from the node")
	waitForVSphereDiskToDetach(vsp, persistentvolumes[0].Spec.VsphereVolume.VolumePath, k8stype.NodeName(pod.Spec.NodeName))
}
