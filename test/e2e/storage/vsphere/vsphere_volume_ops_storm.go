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
	"os"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
	Test to perform Disk Ops storm.

	Steps
    	1. Create storage class for thin Provisioning.
    	2. Create 30 PVCs using above storage class in annotation, requesting 2 GB files.
    	3. Wait until all disks are ready and all PVs and PVCs get bind. (CreateVolume storm)
    	4. Create pod to mount volumes using PVCs created in step 2. (AttachDisk storm)
    	5. Wait for pod status to be running.
    	6. Verify all volumes accessible and available in the pod.
    	7. Delete pod.
    	8. wait until volumes gets detached. (DetachDisk storm)
    	9. Delete all PVCs. This should delete all Disks. (DeleteVolume storm)
		10. Delete storage class.
*/

var _ = utils.SIGDescribe("Volume Operations Storm [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("volume-ops-storm")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	const defaultVolumeOpsScale = 30
	var (
		client            clientset.Interface
		namespace         string
		storageclass      *storagev1.StorageClass
		pvclaims          []*v1.PersistentVolumeClaim
		persistentvolumes []*v1.PersistentVolume
		err               error
		volumeOpsScale    int
	)
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		gomega.Expect(GetReadySchedulableNodeInfos()).NotTo(gomega.BeEmpty())
		if scale := os.Getenv("VOLUME_OPS_SCALE"); scale != "" {
			volumeOpsScale, err = strconv.Atoi(scale)
			framework.ExpectNoError(err)
		} else {
			volumeOpsScale = defaultVolumeOpsScale
		}
		pvclaims = make([]*v1.PersistentVolumeClaim, volumeOpsScale)
	})
	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting PVCs")
		for _, claim := range pvclaims {
			e2epv.DeletePersistentVolumeClaim(client, claim.Name, namespace)
		}
		ginkgo.By("Deleting StorageClass")
		err = client.StorageV1().StorageClasses().Delete(context.TODO(), storageclass.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
	})

	ginkgo.It("should create pod with many volumes and verify no attach call fails", func() {
		ginkgo.By(fmt.Sprintf("Running test with VOLUME_OPS_SCALE: %v", volumeOpsScale))
		ginkgo.By("Creating Storage Class")
		scParameters := make(map[string]string)
		scParameters["diskformat"] = "thin"
		storageclass, err = client.StorageV1().StorageClasses().Create(context.TODO(), getVSphereStorageClassSpec("thinsc", scParameters, nil, ""), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Creating PVCs using the Storage Class")
		count := 0
		for count < volumeOpsScale {
			pvclaims[count], err = e2epv.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "2Gi", storageclass))
			framework.ExpectNoError(err)
			count++
		}

		ginkgo.By("Waiting for all claims to be in bound phase")
		persistentvolumes, err = e2epv.WaitForPVClaimBoundPhase(client, pvclaims, f.Timeouts.ClaimProvision)
		framework.ExpectNoError(err)

		ginkgo.By("Creating pod to attach PVs to the node")
		pod, err := e2epod.CreatePod(client, namespace, nil, pvclaims, false, "")
		framework.ExpectNoError(err)

		ginkgo.By("Verify all volumes are accessible and available in the pod")
		verifyVSphereVolumesAccessible(client, pod, persistentvolumes)

		ginkgo.By("Deleting pod")
		framework.ExpectNoError(e2epod.DeletePodWithWait(client, pod))

		ginkgo.By("Waiting for volumes to be detached from the node")
		for _, pv := range persistentvolumes {
			waitForVSphereDiskToDetach(pv.Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
		}
	})
})
