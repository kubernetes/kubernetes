/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	"k8s.io/kubernetes/test/e2e/upgrades"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const devicePath = "/mnt/volume1"

// VolumeModeDowngradeTest tests that a VolumeMode Block PV is not mistakenly
// formatted and mounted like a nil/Filesystem PV after a downgrade to a version
// where the BlockVolume feature is disabled
type VolumeModeDowngradeTest struct {
	pvSource *v1.PersistentVolumeSource
	pv       *v1.PersistentVolume
	pvc      *v1.PersistentVolumeClaim
	pod      *v1.Pod
}

func (VolumeModeDowngradeTest) Name() string {
	return "[sig-storage] volume-mode-downgrade"
}

func (t *VolumeModeDowngradeTest) Skip(upgCtx upgrades.UpgradeContext) bool {
	if !framework.ProviderIs("openstack", "gce", "aws", "gke", "vsphere", "azure") {
		return true
	}

	// Only run when downgrading from >= 1.13 to < 1.13
	blockVersion := version.MustParseSemantic("1.13.0-alpha.0")
	if upgCtx.Versions[0].Version.LessThan(blockVersion) {
		return true
	}
	if !upgCtx.Versions[1].Version.LessThan(blockVersion) {
		return true
	}

	return false
}

// Setup creates a block pv and then verifies that a pod can consume it.  The pod writes data to the volume.
func (t *VolumeModeDowngradeTest) Setup(f *framework.Framework) {

	var err error

	cs := f.ClientSet
	ns := f.Namespace.Name

	By("Creating a PVC")
	block := v1.PersistentVolumeBlock
	pvcConfig := framework.PersistentVolumeClaimConfig{
		StorageClassName: nil,
		VolumeMode:       &block,
	}
	t.pvc = framework.MakePersistentVolumeClaim(pvcConfig, ns)
	t.pvc, err = framework.CreatePVC(cs, ns, t.pvc)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, ns, t.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	t.pvc, err = cs.CoreV1().PersistentVolumeClaims(t.pvc.Namespace).Get(t.pvc.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	t.pv, err = cs.CoreV1().PersistentVolumes().Get(t.pvc.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("Consuming the PVC before downgrade")
	t.pod, err = framework.CreateSecPod(cs, ns, []*v1.PersistentVolumeClaim{t.pvc}, false, "", false, false, framework.SELinuxLabel, nil, framework.PodStartTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("Checking if PV exists as expected volume mode")
	utils.CheckVolumeModeOfPath(t.pod, block, devicePath)

	By("Checking if read/write to PV works properly")
	utils.CheckReadWriteToPath(t.pod, block, devicePath)
}

// Test waits for the downgrade to complete, and then verifies that a pod can no
// longer consume the pv as it is not mapped nor mounted into the pod
func (t *VolumeModeDowngradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	By("Waiting for downgrade to finish")
	<-done

	By("Verifying that nothing exists at the device path in the pod")
	utils.VerifyExecInPodFail(t.pod, fmt.Sprintf("test -e %s", devicePath), 1)
}

// Teardown cleans up any remaining resources.
func (t *VolumeModeDowngradeTest) Teardown(f *framework.Framework) {
	By("Deleting the pod")
	framework.ExpectNoError(framework.DeletePodWithWait(f, f.ClientSet, t.pod))

	By("Deleting the PVC")
	framework.ExpectNoError(f.ClientSet.CoreV1().PersistentVolumeClaims(t.pvc.Namespace).Delete(t.pvc.Name, nil))

	By("Waiting for the PV to be deleted")
	framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(f.ClientSet, t.pv.Name, 5*time.Second, 20*time.Minute))
}
