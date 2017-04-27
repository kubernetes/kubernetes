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

package upgrades

import (
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// PersistentVolumeUpgradeTest test that a pv is available before and after a cluster upgrade.
type PersistentVolumeUpgradeTest struct {
	pvSource *v1.PersistentVolumeSource
	pv       *v1.PersistentVolume
	pvc      *v1.PersistentVolumeClaim
}

func (PersistentVolumeUpgradeTest) Name() string { return "persistent-volume-upgrade" }

const (
	pvTestFile string = "/mnt/volume1/pv_upgrade_test"
	pvTestData string = "keep it pv"
	pvWriteCmd string = "echo \"" + pvTestData + "\" > " + pvTestFile
	pvReadCmd  string = "cat " + pvTestFile
)

func (t *PersistentVolumeUpgradeTest) createGCEVolume() *v1.PersistentVolumeSource {
	diskName, err := framework.CreatePDWithRetry()
	framework.ExpectNoError(err)
	return &v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   diskName,
			FSType:   "ext3",
			ReadOnly: false,
		},
	}
}
func (t *PersistentVolumeUpgradeTest) deleteGCEVolume(pvSource *v1.PersistentVolumeSource) error {
	return framework.DeletePDWithRetry(pvSource.GCEPersistentDisk.PDName)
}

// Setup creates a pv and then verifies that a pod can consume it.  The pod writes data to the volume.
func (t *PersistentVolumeUpgradeTest) Setup(f *framework.Framework) {

	var err error
	// TODO: generalize this to other providers
	framework.SkipUnlessProviderIs("gce", "gke")

	ns := f.Namespace.Name

	By("Initializing PV source")
	t.pvSource = t.createGCEVolume()
	pvConfig := framework.PersistentVolumeConfig{
		NamePrefix: "pv-upgrade",
		PVSource:   *t.pvSource,
		Prebind:    nil,
	}
	pvcConfig := framework.PersistentVolumeClaimConfig{
		Annotations: map[string]string{
			v1.BetaStorageClassAnnotation: "",
		},
	}

	By("Creating the PV and PVC")
	t.pv, t.pvc, err = framework.CreatePVPVC(f.ClientSet, pvConfig, pvcConfig, ns, true)
	Expect(err).NotTo(HaveOccurred())
	framework.ExpectNoError(framework.WaitOnPVandPVC(f.ClientSet, ns, t.pv, t.pvc))

	By("Consuming the PV before upgrade")
	t.testPod(f, pvWriteCmd+";"+pvReadCmd)
}

// Test waits for the upgrade to complete, and then verifies that a pod can still consume the pv
// and that the volume data persists.
func (t *PersistentVolumeUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	By("Consuming the PV after upgrade")
	t.testPod(f, pvReadCmd)
}

// Teardown cleans up any remaining resources.
func (t *PersistentVolumeUpgradeTest) Teardown(f *framework.Framework) {
	errs := framework.PVPVCCleanup(f.ClientSet, f.Namespace.Name, t.pv, t.pvc)
	if err := t.deleteGCEVolume(t.pvSource); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		framework.Failf("Failed to delete 1 or more PVs/PVCs and/or the GCE volume. Errors: %v", utilerrors.NewAggregate(errs))
	}
}

// testPod creates a pod that consumes a pv and prints it out. The output is then verified.
func (t *PersistentVolumeUpgradeTest) testPod(f *framework.Framework, cmd string) {
	pod := framework.MakePod(f.Namespace.Name, []*v1.PersistentVolumeClaim{t.pvc}, false, cmd)
	expectedOutput := []string{pvTestData}
	f.TestContainerOutput("pod consumes pv", pod, 0, expectedOutput)
}
