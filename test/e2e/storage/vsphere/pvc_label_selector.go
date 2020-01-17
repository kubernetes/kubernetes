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
	"time"

	"github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
   This is a function test for Selector-Label Volume Binding Feature
   Test verifies volume with the matching label is bounded with the PVC.

   Test Steps
   ----------
   1. Create VMDK.
   2. Create pv with label volume-type:ssd, volume path set to vmdk created in previous step, and PersistentVolumeReclaimPolicy is set to Delete.
   3. Create PVC (pvc_vvol) with label selector to match with volume-type:vvol
   4. Create PVC (pvc_ssd) with label selector to match with volume-type:ssd
   5. Wait and verify pvc_ssd is bound with PV.
   6. Verify Status of pvc_vvol is still pending.
   7. Delete pvc_ssd.
   8. verify associated pv is also deleted.
   9. delete pvc_vvol

*/
var _ = utils.SIGDescribe("PersistentVolumes [Feature:LabelSelector]", func() {
	f := framework.NewDefaultFramework("pvclabelselector")
	var (
		c          clientset.Interface
		ns         string
		pv_ssd     *v1.PersistentVolume
		pvc_ssd    *v1.PersistentVolumeClaim
		pvc_vvol   *v1.PersistentVolumeClaim
		volumePath string
		ssdlabels  map[string]string
		vvollabels map[string]string
		err        error
		nodeInfo   *NodeInfo
	)
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		Bootstrap(f)
		nodeInfo = GetReadySchedulableRandomNodeInfo()
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		ssdlabels = make(map[string]string)
		ssdlabels["volume-type"] = "ssd"
		vvollabels = make(map[string]string)
		vvollabels["volume-type"] = "vvol"

	})

	utils.SIGDescribe("Selector-Label Volume Binding:vsphere", func() {
		ginkgo.AfterEach(func() {
			ginkgo.By("Running clean up actions")
			if framework.ProviderIs("vsphere") {
				testCleanupVSpherePVClabelselector(c, ns, nodeInfo, volumePath, pv_ssd, pvc_ssd, pvc_vvol)
			}
		})
		ginkgo.It("should bind volume with claim for given label", func() {
			volumePath, pv_ssd, pvc_ssd, pvc_vvol, err = testSetupVSpherePVClabelselector(c, nodeInfo, ns, ssdlabels, vvollabels)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the pvc_ssd to bind with pv_ssd")
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, ns, pv_ssd, pvc_ssd))

			ginkgo.By("Verify status of pvc_vvol is pending")
			err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimPending, c, ns, pvc_vvol.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)

			ginkgo.By("delete pvc_ssd")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc_ssd.Name, ns), "Failed to delete PVC ", pvc_ssd.Name)

			ginkgo.By("verify pv_ssd is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv_ssd.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)
			volumePath = ""

			ginkgo.By("delete pvc_vvol")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc_vvol.Name, ns), "Failed to delete PVC ", pvc_vvol.Name)
		})
	})
})

func testSetupVSpherePVClabelselector(c clientset.Interface, nodeInfo *NodeInfo, ns string, ssdlabels map[string]string, vvollabels map[string]string) (volumePath string, pv_ssd *v1.PersistentVolume, pvc_ssd *v1.PersistentVolumeClaim, pvc_vvol *v1.PersistentVolumeClaim, err error) {
	ginkgo.By("creating vmdk")
	volumePath = ""
	volumePath, err = nodeInfo.VSphere.CreateVolume(&VolumeOptions{}, nodeInfo.DataCenterRef)
	if err != nil {
		return
	}

	ginkgo.By("creating the pv with label volume-type:ssd")
	pv_ssd = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimDelete, ssdlabels)
	pv_ssd, err = c.CoreV1().PersistentVolumes().Create(pv_ssd)
	if err != nil {
		return
	}

	ginkgo.By("creating pvc with label selector to match with volume-type:vvol")
	pvc_vvol = getVSpherePersistentVolumeClaimSpec(ns, vvollabels)
	pvc_vvol, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc_vvol)
	if err != nil {
		return
	}

	ginkgo.By("creating pvc with label selector to match with volume-type:ssd")
	pvc_ssd = getVSpherePersistentVolumeClaimSpec(ns, ssdlabels)
	pvc_ssd, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc_ssd)
	return
}

func testCleanupVSpherePVClabelselector(c clientset.Interface, ns string, nodeInfo *NodeInfo, volumePath string, pv_ssd *v1.PersistentVolume, pvc_ssd *v1.PersistentVolumeClaim, pvc_vvol *v1.PersistentVolumeClaim) {
	ginkgo.By("running testCleanupVSpherePVClabelselector")
	if len(volumePath) > 0 {
		nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
	}
	if pvc_ssd != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc_ssd.Name, ns), "Failed to delete PVC ", pvc_ssd.Name)
	}
	if pvc_vvol != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc_vvol.Name, ns), "Failed to delete PVC ", pvc_vvol.Name)
	}
	if pv_ssd != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv_ssd.Name), "Failed to delete PV ", pv_ssd.Name)
	}
}
