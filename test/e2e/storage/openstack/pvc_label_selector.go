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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("PersistentVolumes [Feature:LabelSelector]", func() {
	f := framework.NewDefaultFramework("pvclabelselector")
	var (
		c          clientset.Interface
		ns         string
		pvssd      *v1.PersistentVolume
		pvcssd     *v1.PersistentVolumeClaim
		pvcvvol    *v1.PersistentVolumeClaim
		volumeID   string
		ssdlabels  map[string]string
		vvollabels map[string]string
		err        error
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		ssdlabels = make(map[string]string)
		ssdlabels["volume-type"] = "ssd"
		vvollabels = make(map[string]string)
		vvollabels["volume-type"] = "vvol"

	})

	utils.SIGDescribe("Selector-Label Volume Binding:openstack", func() {
		AfterEach(func() {
			By("Running clean up actions")
			if framework.ProviderIs("openstack") {
				testCleanupOpenstackPVClabelselector(c, ns, volumeID, pvssd, pvcssd, pvcvvol)
			}
		})
		It("should bind volume with claim for given label", func() {
			volumeID, pvssd, pvcssd, pvcvvol, err = testSetupOpenstackPVClabelselector(c, ns, ssdlabels, vvollabels)
			Expect(err).NotTo(HaveOccurred())

			By("wait for the pvcssd to bind with pvssd")
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pvssd, pvcssd))

			By("Verify status of pvcvvol is pending")
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimPending, c, ns, pvcvvol.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete pvcssd")
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvcssd.Name, ns), "Failed to delete PVC ", pvcssd.Name)

			By("verify pvssd is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pvssd.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
			volumeID = ""

			By("delete pvcvvol")
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvcvvol.Name, ns), "Failed to delete PVC ", pvcvvol.Name)
		})
	})
})

func testSetupOpenstackPVClabelselector(c clientset.Interface, ns string, ssdlabels map[string]string, vvollabels map[string]string) (volumeID string, pvssd *v1.PersistentVolume, pvcssd *v1.PersistentVolumeClaim, pvcvvol *v1.PersistentVolumeClaim, err error) {
	volumeID = ""
	By("creating vmdk")
	osp, _, err := getOpenstack(c)
	Expect(err).NotTo(HaveOccurred())
	volumeID, err = createOpenstackVolume(osp)
	if err != nil {
		return
	}

	By("creating the pv with lable volume-type:ssd")
	pvssd = getOpenstackPersistentVolumeSpec(volumeID, v1.PersistentVolumeReclaimDelete, ssdlabels)
	pvssd, err = c.CoreV1().PersistentVolumes().Create(pvssd)
	if err != nil {
		return
	}

	By("creating pvc with label selector to match with volume-type:vvol")
	pvcvvol = getOpenstackPersistentVolumeClaimSpec(ns, vvollabels)
	pvcvvol, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvcvvol)
	if err != nil {
		return
	}

	By("creating pvc with label selector to match with volume-type:ssd")
	pvcssd = getOpenstackPersistentVolumeClaimSpec(ns, ssdlabels)
	pvcssd, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvcssd)
	return
}

func testCleanupOpenstackPVClabelselector(c clientset.Interface, ns string, volumeID string, pvssd *v1.PersistentVolume, pvcssd *v1.PersistentVolumeClaim, pvcvvol *v1.PersistentVolumeClaim) {
	By("running testCleanupOpenstackPVClabelselector")
	if len(volumeID) > 0 {
		osp, _, err := getOpenstack(c)
		Expect(err).NotTo(HaveOccurred())
		osp.DeleteVolume(volumeID)
	}
	if pvcssd != nil {
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvcssd.Name, ns), "Failed to delete PVC ", pvcssd.Name)
	}
	if pvcvvol != nil {
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvcvvol.Name, ns), "Failed to delete PVC ", pvcvvol.Name)
	}
	if pvssd != nil {
		framework.ExpectNoError(framework.DeletePersistentVolume(c, pvssd.Name), "Faled to delete PV ", pvssd.Name)
	}
}
