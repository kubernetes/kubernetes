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
	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
)

// PersistentVolumeUpgradeTest test that a pv is available before and after a cluster upgrade.
type PersistentVolumeUpgradeTest struct {
	pvc *v1.PersistentVolumeClaim
}

// Name returns the tracking name of the test.
func (PersistentVolumeUpgradeTest) Name() string { return "[sig-storage] persistent-volume-upgrade" }

const (
	pvTestFile string = "/mnt/volume1/pv_upgrade_test"
	pvTestData string = "keep it pv"
	pvWriteCmd string = "echo \"" + pvTestData + "\" > " + pvTestFile
	pvReadCmd  string = "cat " + pvTestFile
)

// Setup creates a pv and then verifies that a pod can consume it.  The pod writes data to the volume.
func (t *PersistentVolumeUpgradeTest) Setup(f *framework.Framework) {

	var err error
	e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws", "vsphere", "azure")

	ns := f.Namespace.Name

	ginkgo.By("Creating a PVC")
	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		StorageClassName: nil,
	}
	t.pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, ns)
	t.pvc, err = e2epv.CreatePVC(f.ClientSet, ns, t.pvc)
	framework.ExpectNoError(err)

	ginkgo.By("Consuming the PV before upgrade")
	t.testPod(f, pvWriteCmd+";"+pvReadCmd)
}

// Test waits for the upgrade to complete, and then verifies that a pod can still consume the pv
// and that the volume data persists.
func (t *PersistentVolumeUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Consuming the PV after upgrade")
	t.testPod(f, pvReadCmd)
}

// Teardown cleans up any remaining resources.
func (t *PersistentVolumeUpgradeTest) Teardown(f *framework.Framework) {
	errs := e2epv.PVPVCCleanup(f.ClientSet, f.Namespace.Name, nil, t.pvc)
	if len(errs) > 0 {
		framework.Failf("Failed to delete 1 or more PVs/PVCs. Errors: %v", utilerrors.NewAggregate(errs))
	}
}

// testPod creates a pod that consumes a pv and prints it out. The output is then verified.
func (t *PersistentVolumeUpgradeTest) testPod(f *framework.Framework, cmd string) {
	pod := e2epod.MakePod(f.Namespace.Name, nil, []*v1.PersistentVolumeClaim{t.pvc}, false, cmd)
	expectedOutput := []string{pvTestData}
	f.TestContainerOutput("pod consumes pv", pod, 0, expectedOutput)
}
