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
	"os"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	openstack "k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/test/e2e/framework"
	utils "k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Volume Operations Storm [Feature:openstack]", func() {
	f := framework.NewDefaultFramework("volume-ops-storm")
	const DefaultVolumeOPSScale = 30
	var (
		client            clientset.Interface
		namespace         string
		storageclass      *storage.StorageClass
		pvclaims          []*v1.PersistentVolumeClaim
		persistentvolumes []*v1.PersistentVolume
		err               error
		volumeOpsScale    int
		osp               *openstack.OpenStack
	)

	osp, id, err := getOpenstack(client)
	Expect(err).NotTo(HaveOccurred())

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) == 0 {
			framework.Failf("Unable to find ready and schedulable Node")
		}
		if os.Getenv("VOLUME_OPS_SCALE") != "" {
			volumeOpsScale, err = strconv.Atoi(os.Getenv("VOLUME_OPS_SCALE"))
			Expect(err).NotTo(HaveOccurred())
		} else {
			volumeOpsScale = DefaultVolumeOPSScale
		}
		pvclaims = make([]*v1.PersistentVolumeClaim, volumeOpsScale)
	})
	AfterEach(func() {
		By("Deleting PVCs")
		for _, claim := range pvclaims {
			framework.DeletePersistentVolumeClaim(client, claim.Name, namespace)
		}
		By("Deleting StorageClass")
		err = client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create pod with many volumes and verify no attach call fails", func() {
		By(fmt.Sprintf("Running test with VOLUME_OPS_SCALE: %v", volumeOpsScale))
		By("Creating Storage Class")
		scParameters := make(map[string]string)
		scParameters["diskformat"] = "thin"
		storageclass, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec("thinsc", scParameters))
		Expect(err).NotTo(HaveOccurred())

		By("Creating PVCs using the Storage Class")
		count := 0
		for count < volumeOpsScale {
			pvclaims[count], err = framework.CreatePVC(client, namespace, getOpenstackClaimSpecWithStorageClassAnnotation(namespace, "2Gi", storageclass))
			Expect(err).NotTo(HaveOccurred())
			count++
		}

		By("Waiting for all claims to be in bound phase")
		persistentvolumes, err = framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pod to attach PVs to the node")
		pod, err := framework.CreatePod(client, namespace, nil, pvclaims, false, "")
		Expect(err).NotTo(HaveOccurred())

		By("Verify all volumes are accessible and available in the pod")
		verifyOpenstackVolumesAccessible(client, pod, persistentvolumes, id, osp)

		By("Deleting pod")
		framework.ExpectNoError(framework.DeletePodWithWait(f, client, pod))

		By("Waiting for volumes to be detached from the node")
		for _, pv := range persistentvolumes {
			WaitForVolumeStatus(osp, pv.Spec.Cinder.VolumeID, VolumeAvailableStatus)
		}
	})
})
