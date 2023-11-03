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
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
This is a function test for Selector-Label Volume Binding Feature
Test verifies volume with the matching label is bounded with the PVC.

Test Steps
----------
1. Create VMDK.
2. Create pv with label volume-type:ssd, volume path set to vmdk created in previous step, and PersistentVolumeReclaimPolicy is set to Delete.
3. Create PVC (pvcVvol) with label selector to match with volume-type:vvol
4. Create PVC (pvcSsd) with label selector to match with volume-type:ssd
5. Wait and verify pvSsd is bound with PV.
6. Verify Status of pvcVvol is still pending.
7. Delete pvcSsd.
8. verify associated pv is also deleted.
9. delete pvcVvol
*/
var _ = utils.SIGDescribe("PersistentVolumes", feature.Vsphere, feature.LabelSelector, func() {
	f := framework.NewDefaultFramework("pvclabelselector")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var (
		c          clientset.Interface
		ns         string
		pvSsd      *v1.PersistentVolume
		pvcSsd     *v1.PersistentVolumeClaim
		pvcVvol    *v1.PersistentVolumeClaim
		volumePath string
		ssdlabels  map[string]string
		vvollabels map[string]string
		err        error
		nodeInfo   *NodeInfo
	)
	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		Bootstrap(f)
		nodeInfo = GetReadySchedulableRandomNodeInfo(ctx, c)
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, c, f.Timeouts.NodeSchedulable))
		ssdlabels = make(map[string]string)
		ssdlabels["volume-type"] = "ssd"
		vvollabels = make(map[string]string)
		vvollabels["volume-type"] = "vvol"

	})

	f.Describe("Selector-Label Volume Binding:vsphere", feature.Vsphere, func() {
		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Running clean up actions")
			if framework.ProviderIs("vsphere") {
				testCleanupVSpherePVClabelselector(ctx, c, ns, nodeInfo, volumePath, pvSsd, pvcSsd, pvcVvol)
			}
		})
		ginkgo.It("should bind volume with claim for given label", func(ctx context.Context) {
			volumePath, pvSsd, pvcSsd, pvcVvol, err = testSetupVSpherePVClabelselector(ctx, c, nodeInfo, ns, ssdlabels, vvollabels)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the pvcSsd to bind with pvSsd")
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(ctx, c, f.Timeouts, ns, pvSsd, pvcSsd))

			ginkgo.By("Verify status of pvcVvol is pending")
			err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimPending, c, ns, pvcVvol.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)

			ginkgo.By("delete pvcSsd")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, pvcSsd.Name, ns), "Failed to delete PVC ", pvcSsd.Name)

			ginkgo.By("verify pvSsd is deleted")
			err = e2epv.WaitForPersistentVolumeDeleted(ctx, c, pvSsd.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)
			volumePath = ""

			ginkgo.By("delete pvcVvol")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, pvcVvol.Name, ns), "Failed to delete PVC ", pvcVvol.Name)
		})
	})
})

func testSetupVSpherePVClabelselector(ctx context.Context, c clientset.Interface, nodeInfo *NodeInfo, ns string, ssdlabels map[string]string, vvollabels map[string]string) (volumePath string, pvSsd *v1.PersistentVolume, pvcSsd *v1.PersistentVolumeClaim, pvcVvol *v1.PersistentVolumeClaim, err error) {
	ginkgo.By("creating vmdk")
	volumePath, err = nodeInfo.VSphere.CreateVolume(&VolumeOptions{}, nodeInfo.DataCenterRef)
	if err != nil {
		return
	}

	ginkgo.By("creating the pv with label volume-type:ssd")
	pvSsd = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimDelete, ssdlabels)
	pvSsd, err = c.CoreV1().PersistentVolumes().Create(ctx, pvSsd, metav1.CreateOptions{})
	if err != nil {
		return
	}

	ginkgo.By("creating pvc with label selector to match with volume-type:vvol")
	pvcVvol = getVSpherePersistentVolumeClaimSpec(ns, vvollabels)
	pvcVvol, err = c.CoreV1().PersistentVolumeClaims(ns).Create(ctx, pvcVvol, metav1.CreateOptions{})
	if err != nil {
		return
	}

	ginkgo.By("creating pvc with label selector to match with volume-type:ssd")
	pvcSsd = getVSpherePersistentVolumeClaimSpec(ns, ssdlabels)
	pvcSsd, err = c.CoreV1().PersistentVolumeClaims(ns).Create(ctx, pvcSsd, metav1.CreateOptions{})
	return
}

func testCleanupVSpherePVClabelselector(ctx context.Context, c clientset.Interface, ns string, nodeInfo *NodeInfo, volumePath string, pvSsd *v1.PersistentVolume, pvcSsd *v1.PersistentVolumeClaim, pvcVvol *v1.PersistentVolumeClaim) {
	ginkgo.By("running testCleanupVSpherePVClabelselector")
	if len(volumePath) > 0 {
		nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
	}
	if pvcSsd != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, pvcSsd.Name, ns), "Failed to delete PVC ", pvcSsd.Name)
	}
	if pvcVvol != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, pvcVvol.Name, ns), "Failed to delete PVC ", pvcVvol.Name)
	}
	if pvSsd != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolume(ctx, c, pvSsd.Name), "Failed to delete PV ", pvSsd.Name)
	}
}
