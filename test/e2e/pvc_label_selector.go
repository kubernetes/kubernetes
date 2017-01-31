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
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
	"time"
)

/*
   This is a function test for Selector-Label Volume Binding Feature
   Test verifies volume with the matching label is bounded with the PVC.

   Test Steps
   ----------
   1. Create VMDK.
   2. Create pv with lable volume-type:ssd, volume path set to vmdk created in previous step, and PersistentVolumeReclaimPolicy is set to Delete.
   3. Create PVC (pvc_vvol) with label selector to match with volume-type:vvol
   4. Create PVC (pvc_ssd) with label selector to match with volume-type:ssd
   5. Wait and verify pvc_ssd is bound with PV.
   6. Verify Status of pvc_vvol is still pending.
   7. Delete pvc_ssd.
   8. verify associated pv is also deleted.
   9. delete pvc_vvol

*/
var _ = framework.KubeDescribe("pvclabelselector", func() {
	f := framework.NewDefaultFramework("pvclabelselector")
	var c clientset.Interface
	var ns string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	framework.KubeDescribe("Selector-Label Volume Binding", func() {
		var (
			volumePath    string
			pv_ssd        *v1.PersistentVolume
			pvc_ssd       *v1.PersistentVolumeClaim
			pvc_vvol      *v1.PersistentVolumeClaim
			volumeoptions vsphere.VolumeOptions
		)

		It("should bind volume with claim for given label", func() {

			By("creating vmdk")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumeoptions.CapacityKB = 2097152
			volumeoptions.Name = "e2e-disk" + time.Now().Format("20060102150405")
			volumeoptions.DiskFormat = "thin"

			volumePath, err = vsp.CreateVolume(&volumeoptions)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pv with lable volume-type:ssd")
			ssdlabels := make(map[string]string)
			ssdlabels["volume-type"] = "ssd"
			pv_ssd = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimDelete, ssdlabels)
			pv_ssd, err := c.CoreV1().PersistentVolumes().Create(pv_ssd)
			Expect(err).NotTo(HaveOccurred())

			By("creating pvc with label selector to match with volume-type:vvol")
			vvollabels := make(map[string]string)
			vvollabels["volume-type"] = "vvol"
			pvc_vvol = getVSpherePersistentVolumeClaimSpec(ns, vvollabels)
			pvc_vvol, err := c.CoreV1().PersistentVolumeClaims(ns).Create(pvc_vvol)
			Expect(err).NotTo(HaveOccurred())

			By("creating pvc with label selector to match with volume-type:ssd")
			pvc_ssd = getVSpherePersistentVolumeClaimSpec(ns, ssdlabels)
			pvc_ssd, err := c.CoreV1().PersistentVolumeClaims(ns).Create(pvc_ssd)

			By("wait for the pvc_ssd to bind with pv_ssd")
			waitOnPVandPVC(c, ns, pv_ssd, pvc_ssd)

			By("Verify status of pvc_vvol is pending")
			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimPending, c, ns, pvc_vvol.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete pvc_ssd")
			deletePersistentVolumeClaim(c, pvc_ssd.Name, ns)

			By("verify pv_ssd is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv_ssd.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete pvc_vvol")
			deletePersistentVolumeClaim(c, pvc_vvol.Name, ns)
		})
	})
})
