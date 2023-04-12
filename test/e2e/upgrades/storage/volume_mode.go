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
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
)

const devicePath = "/mnt/volume1"

// VolumeModeDowngradeTest tests that a VolumeMode Block PV is not mistakenly
// formatted and mounted like a nil/Filesystem PV after a downgrade to a version
// where the BlockVolume feature is disabled
type VolumeModeDowngradeTest struct {
	pv  *v1.PersistentVolume
	pvc *v1.PersistentVolumeClaim
	pod *v1.Pod
}

// Name returns the tracking name of the test.
func (VolumeModeDowngradeTest) Name() string {
	return "[sig-storage] volume-mode-downgrade"
}

// Skip returns true when this test can be skipped.
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
func (t *VolumeModeDowngradeTest) Setup(ctx context.Context, f *framework.Framework) {

	var err error

	cs := f.ClientSet
	ns := f.Namespace.Name

	ginkgo.By("Creating a PVC")
	block := v1.PersistentVolumeBlock
	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		StorageClassName: nil,
		VolumeMode:       &block,
	}
	t.pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, ns)
	t.pvc, err = e2epv.CreatePVC(ctx, cs, ns, t.pvc)
	framework.ExpectNoError(err)

	err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, ns, t.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
	framework.ExpectNoError(err)

	t.pvc, err = cs.CoreV1().PersistentVolumeClaims(t.pvc.Namespace).Get(ctx, t.pvc.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	t.pv, err = cs.CoreV1().PersistentVolumes().Get(ctx, t.pvc.Spec.VolumeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Consuming the PVC before downgrade")
	podConfig := e2epod.Config{
		NS:           ns,
		PVCs:         []*v1.PersistentVolumeClaim{t.pvc},
		SeLinuxLabel: e2epv.SELinuxLabel,
	}
	t.pod, err = e2epod.CreateSecPod(ctx, cs, &podConfig, framework.PodStartTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("Checking if PV exists as expected volume mode")
	e2evolume.CheckVolumeModeOfPath(f, t.pod, block, devicePath)

	ginkgo.By("Checking if read/write to PV works properly")
	storageutils.CheckReadWriteToPath(f, t.pod, block, devicePath)
}

// Test waits for the downgrade to complete, and then verifies that a pod can no
// longer consume the pv as it is not mapped nor mounted into the pod
func (t *VolumeModeDowngradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	ginkgo.By("Waiting for downgrade to finish")
	<-done

	ginkgo.By("Verifying that nothing exists at the device path in the pod")
	e2evolume.VerifyExecInPodFail(f, t.pod, fmt.Sprintf("test -e %s", devicePath), 1)
}

// Teardown cleans up any remaining resources.
func (t *VolumeModeDowngradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	ginkgo.By("Deleting the pod")
	framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, f.ClientSet, t.pod))

	ginkgo.By("Deleting the PVC")
	framework.ExpectNoError(f.ClientSet.CoreV1().PersistentVolumeClaims(t.pvc.Namespace).Delete(ctx, t.pvc.Name, metav1.DeleteOptions{}))

	ginkgo.By("Waiting for the PV to be deleted")
	framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(ctx, f.ClientSet, t.pv.Name, 5*time.Second, 20*time.Minute))
}
