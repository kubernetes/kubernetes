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
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	Ext4FSType    = "ext4"
	Ext3FSType    = "ext3"
	InvalidFSType = "ext10"
	ExecCommand   = "/bin/df -T /mnt/volume1 | /bin/awk 'FNR == 2 {print $2}' > /mnt/volume1/fstype && while true ; do sleep 2 ; done"
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

	Test to verify if an invalid fstype specified in storage class fails pod creation.

	Steps
	1. Create StorageClass with inavlid.
	2. Create PVC which uses the StorageClass created in step 1.
	3. Wait for PV to be provisioned.
	4. Wait for PVC's status to become Bound.
	5. Create pod using PVC.
	6. Verify if the pod creation fails.
	7. Verify if the MountVolume.MountDevice fails because it is unable to find the file system executable file on the node.
*/

var _ = utils.SIGDescribe("Volume FStype [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("volume-fstype")
	var (
		client    clientset.Interface
		namespace string
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		Expect(GetReadySchedulableNodeInfos()).NotTo(BeEmpty())
	})

	It("verify fstype - ext3 formatted volume", func() {
		By("Invoking Test for fstype: ext3")
		invokeTestForFstype(f, client, namespace, Ext3FSType, Ext3FSType)
	})

	It("verify fstype - default value should be ext4", func() {
		By("Invoking Test for fstype: Default Value - ext4")
		invokeTestForFstype(f, client, namespace, "", Ext4FSType)
	})

	It("verify invalid fstype", func() {
		By("Invoking Test for fstype: invalid Value")
		invokeTestForInvalidFstype(f, client, namespace, InvalidFSType)
	})
})

func invokeTestForFstype(f *framework.Framework, client clientset.Interface, namespace string, fstype string, expectedContent string) {
	framework.Logf("Invoking Test for fstype: %s", fstype)
	scParameters := make(map[string]string)
	scParameters["fstype"] = fstype

	// Create Persistent Volume
	By("Creating Storage Class With Fstype")
	pvclaim, persistentvolumes := createVolume(client, namespace, scParameters)

	// Create Pod and verify the persistent volume is accessible
	pod := createPodAndVerifyVolumeAccessible(client, namespace, pvclaim, persistentvolumes)
	_, err := framework.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/cat", "/mnt/volume1/fstype"}, expectedContent, time.Minute)
	Expect(err).NotTo(HaveOccurred())

	// Detach and delete volume
	detachVolume(f, client, pod, persistentvolumes[0].Spec.VsphereVolume.VolumePath)
	err = framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)
	Expect(err).To(BeNil())
}

func invokeTestForInvalidFstype(f *framework.Framework, client clientset.Interface, namespace string, fstype string) {
	scParameters := make(map[string]string)
	scParameters["fstype"] = fstype

	// Create Persistent Volume
	By("Creating Storage Class With Invalid Fstype")
	pvclaim, persistentvolumes := createVolume(client, namespace, scParameters)

	By("Creating pod to attach PV to the node")
	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	// Create pod to attach Volume to Node
	pod, err := framework.CreatePod(client, namespace, nil, pvclaims, false, ExecCommand)
	Expect(err).To(HaveOccurred())

	eventList, err := client.CoreV1().Events(namespace).List(metav1.ListOptions{})

	// Detach and delete volume
	detachVolume(f, client, pod, persistentvolumes[0].Spec.VsphereVolume.VolumePath)
	err = framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)
	Expect(err).To(BeNil())

	Expect(eventList.Items).NotTo(BeEmpty())
	errorMsg := `MountVolume.MountDevice failed for volume "` + persistentvolumes[0].Name + `" : executable file not found`
	isFound := false
	for _, item := range eventList.Items {
		if strings.Contains(item.Message, errorMsg) {
			isFound = true
		}
	}
	Expect(isFound).To(BeTrue(), "Unable to verify MountVolume.MountDevice failure")
}

func createVolume(client clientset.Interface, namespace string, scParameters map[string]string) (*v1.PersistentVolumeClaim, []*v1.PersistentVolume) {
	storageclass, err := client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec("fstype", scParameters))
	Expect(err).NotTo(HaveOccurred())
	defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

	By("Creating PVC using the Storage Class")
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(getVSphereClaimSpecWithStorageClass(namespace, "2Gi", storageclass))
	Expect(err).NotTo(HaveOccurred())

	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	By("Waiting for claim to be in bound phase")
	persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())
	return pvclaim, persistentvolumes
}

func createPodAndVerifyVolumeAccessible(client clientset.Interface, namespace string, pvclaim *v1.PersistentVolumeClaim, persistentvolumes []*v1.PersistentVolume) *v1.Pod {
	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	By("Creating pod to attach PV to the node")
	// Create pod to attach Volume to Node
	pod, err := framework.CreatePod(client, namespace, nil, pvclaims, false, ExecCommand)
	Expect(err).NotTo(HaveOccurred())

	// Asserts: Right disk is attached to the pod
	By("Verify the volume is accessible and available in the pod")
	verifyVSphereVolumesAccessible(client, pod, persistentvolumes)
	return pod
}

// detachVolume delete the volume passed in the argument and wait until volume is detached from the node,
func detachVolume(f *framework.Framework, client clientset.Interface, pod *v1.Pod, volPath string) {
	pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).To(BeNil())
	nodeName := pod.Spec.NodeName
	By("Deleting pod")
	framework.DeletePodWithWait(f, client, pod)

	By("Waiting for volumes to be detached from the node")
	waitForVSphereDiskToDetach(volPath, nodeName)
}
