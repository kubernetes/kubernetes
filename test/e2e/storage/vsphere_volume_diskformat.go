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
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stype "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

/*
	Test to verify diskformat specified in storage-class is being honored while volume creation.
	Valid and supported options are eagerzeroedthick, zeroedthick and thin

	Steps
	1. Create StorageClass with diskformat set to valid type
	2. Create PVC which uses the StorageClass created in step 1.
	3. Wait for PV to be provisioned.
	4. Wait for PVC's status to become Bound
	5. Create pod using PVC on specific node.
	6. Wait for Disk to be attached to the node.
	7. Get node VM's devices and find PV's Volume Disk.
	8. Get Backing Info of the Volume Disk and obtain EagerlyScrub and ThinProvisioned
	9. Based on the value of EagerlyScrub and ThinProvisioned, verify diskformat is correct.
	10. Delete pod and Wait for Volume Disk to be detached from the Node.
	11. Delete PVC, PV and Storage Class
*/

var _ = framework.KubeDescribe("Volume Disk Format [Volumes]", func() {
	f := framework.NewDefaultFramework("volume-disk-format")
	var (
		client            clientset.Interface
		namespace         string
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) != 0 {
			nodeName = nodeList.Items[0].Name
		} else {
			framework.Failf("Unable to find ready and schedulable Node")
		}
		if !isNodeLabeled {
			nodeLabelValue := "vsphere_e2e_" + string(uuid.NewUUID())
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel["vsphere_e2e_label"] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(client, nodeName, "vsphere_e2e_label", nodeLabelValue)
			isNodeLabeled = true
		}
	})
	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(client, nodeName, "vsphere_e2e_label")
		}
	})

	It("verify disk format type - eagerzeroedthick is honored for dynamically provisioned pv using storageclass", func() {
		By("Invoking Test for diskformat: eagerzeroedthick")
		invokeTest(f, client, namespace, nodeName, nodeKeyValueLabel, "eagerzeroedthick")
	})
	It("verify disk format type - zeroedthick is honored for dynamically provisioned pv using storageclass", func() {
		By("Invoking Test for diskformat: zeroedthick")
		invokeTest(f, client, namespace, nodeName, nodeKeyValueLabel, "zeroedthick")
	})
	It("verify disk format type - thin is honored for dynamically provisioned pv using storageclass", func() {
		By("Invoking Test for diskformat: thin")
		invokeTest(f, client, namespace, nodeName, nodeKeyValueLabel, "thin")
	})
})

func invokeTest(f *framework.Framework, client clientset.Interface, namespace string, nodeName string, nodeKeyValueLabel map[string]string, diskFormat string) {

	framework.Logf("Invoking Test for DiskFomat: %s", diskFormat)
	scParameters := make(map[string]string)
	scParameters["diskformat"] = diskFormat

	By("Creating Storage Class With DiskFormat")
	storageClassSpec := getVSphereStorageClassSpec("thinsc", scParameters)
	storageclass, err := client.StorageV1().StorageClasses().Create(storageClassSpec)
	Expect(err).NotTo(HaveOccurred())

	defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

	By("Creating PVC using the Storage Class")
	pvclaimSpec := getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass)
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(pvclaimSpec)
	Expect(err).NotTo(HaveOccurred())

	defer func() {
		client.CoreV1().PersistentVolumeClaims(namespace).Delete(pvclaimSpec.Name, nil)
	}()

	By("Waiting for claim to be in bound phase")
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, pvclaim.Namespace, pvclaim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	// Get new copy of the claim
	pvclaim, err = client.CoreV1().PersistentVolumeClaims(pvclaim.Namespace).Get(pvclaim.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(pvclaim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	/*
		PV is required to be attached to the Node. so that using govmomi API we can grab Disk's Backing Info
		to check EagerlyScrub and ThinProvisioned property
	*/
	By("Creating pod to attach PV to the node")
	// Create pod to attach Volume to Node
	podSpec := getVSpherePodSpecWithClaim(pvclaim.Name, nodeKeyValueLabel, "while true ; do sleep 2 ; done")
	pod, err := client.CoreV1().Pods(namespace).Create(podSpec)
	Expect(err).NotTo(HaveOccurred())

	vsp, err := vsphere.GetVSphere()
	Expect(err).NotTo(HaveOccurred())
	verifyVSphereDiskAttached(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(nodeName))

	By("Waiting for pod to be running")
	Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())
	Expect(verifyDiskFormat(nodeName, pv.Spec.VsphereVolume.VolumePath, diskFormat)).To(BeTrue(), "DiskFormat Verification Failed")

	var volumePaths []string
	volumePaths = append(volumePaths, pv.Spec.VsphereVolume.VolumePath)

	By("Delete pod and wait for volume to be detached from node")
	deletePodAndWaitForVolumeToDetach(f, client, pod, vsp, nodeName, volumePaths)

}

func verifyDiskFormat(nodeName string, pvVolumePath string, diskFormat string) bool {
	By("Verifing disk format")
	eagerlyScrub := false
	thinProvisioned := false
	diskFound := false
	pvvmdkfileName := filepath.Base(pvVolumePath) + filepath.Ext(pvVolumePath)

	govMoMiClient, err := vsphere.GetgovmomiClient(nil)
	Expect(err).NotTo(HaveOccurred())

	f := find.NewFinder(govMoMiClient.Client, true)
	ctx, _ := context.WithCancel(context.Background())
	vm, err := f.VirtualMachine(ctx, os.Getenv("VSPHERE_WORKING_DIR")+nodeName)
	Expect(err).NotTo(HaveOccurred())

	vmDevices, err := vm.Device(ctx)
	Expect(err).NotTo(HaveOccurred())

	disks := vmDevices.SelectByType((*types.VirtualDisk)(nil))

	for _, disk := range disks {
		backing := disk.GetVirtualDevice().Backing.(*types.VirtualDiskFlatVer2BackingInfo)
		backingFileName := filepath.Base(backing.FileName) + filepath.Ext(backing.FileName)
		if backingFileName == pvvmdkfileName {
			diskFound = true
			if backing.EagerlyScrub != nil {
				eagerlyScrub = *backing.EagerlyScrub
			}
			if backing.ThinProvisioned != nil {
				thinProvisioned = *backing.ThinProvisioned
			}
			break
		}
	}

	Expect(diskFound).To(BeTrue(), "Failed to find disk")
	isDiskFormatCorrect := false
	if diskFormat == "eagerzeroedthick" {
		if eagerlyScrub == true && thinProvisioned == false {
			isDiskFormatCorrect = true
		}
	} else if diskFormat == "zeroedthick" {
		if eagerlyScrub == false && thinProvisioned == false {
			isDiskFormatCorrect = true
		}
	} else if diskFormat == "thin" {
		if eagerlyScrub == false && thinProvisioned == true {
			isDiskFormatCorrect = true
		}
	}
	return isDiskFormatCorrect
}
