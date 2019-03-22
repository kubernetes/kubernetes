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

package testsuites

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	noProvisioner = "kubernetes.io/no-provisioner"
	pvNamePrefix  = "pv"
)

type volumeModeTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeModeTestSuite{}

// InitVolumeModeTestSuite returns volumeModeTestSuite that implements TestSuite interface
func InitVolumeModeTestSuite() TestSuite {
	return &volumeModeTestSuite{
		tsInfo: TestSuiteInfo{
			name: "volumeMode",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.FsVolModePreprovisionedPV,
				testpatterns.FsVolModeDynamicPV,
				testpatterns.BlockVolModePreprovisionedPV,
				testpatterns.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *volumeModeTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeModeTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		cs clientset.Interface
		ns *v1.Namespace
		// genericVolumeTestResource contains pv, pvc, sc, etc., owns cleaning that up
		genericVolumeTestResource
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	// No preconditions to test. Normally they would be in a BeforeEach here.

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volumemode")

	init := func() {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet

		// Now do the more expensive test initialization.
		l.config, l.testCleanup = driver.PrepareTest(f)

		fsType := pattern.FsType
		volBindMode := storagev1.VolumeBindingImmediate

		var (
			scName             string
			pvSource           *v1.PersistentVolumeSource
			volumeNodeAffinity *v1.VolumeNodeAffinity
		)

		l.genericVolumeTestResource = genericVolumeTestResource{
			driver:  driver,
			config:  l.config,
			pattern: pattern,
		}

		// Create volume for pre-provisioned volume tests
		l.volume = CreateVolume(driver, l.config, pattern.VolType)

		switch pattern.VolType {
		case testpatterns.PreprovisionedPV:
			if pattern.VolMode == v1.PersistentVolumeBlock {
				scName = fmt.Sprintf("%s-%s-sc-for-block", l.ns.Name, dInfo.Name)
			} else if pattern.VolMode == v1.PersistentVolumeFilesystem {
				scName = fmt.Sprintf("%s-%s-sc-for-file", l.ns.Name, dInfo.Name)
			}
			if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
				pvSource, volumeNodeAffinity = pDriver.GetPersistentVolumeSource(false, fsType, l.volume)
				if pvSource == nil {
					framework.Skipf("Driver %q does not define PersistentVolumeSource - skipping", dInfo.Name)
				}

				storageClass, pvConfig, pvcConfig := generateConfigsForPreprovisionedPVTest(scName, volBindMode, pattern.VolMode, *pvSource, volumeNodeAffinity)
				l.sc = storageClass
				l.pv = framework.MakePersistentVolume(pvConfig)
				l.pvc = framework.MakePersistentVolumeClaim(pvcConfig, l.ns.Name)
			}
		case testpatterns.DynamicPV:
			if dDriver, ok := driver.(DynamicPVTestDriver); ok {
				l.sc = dDriver.GetDynamicProvisionStorageClass(l.config, fsType)
				if l.sc == nil {
					framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
				}
				l.sc.VolumeBindingMode = &volBindMode

				claimSize := dDriver.GetClaimSize()
				l.pvc = getClaim(claimSize, l.ns.Name)
				l.pvc.Spec.StorageClassName = &l.sc.Name
				l.pvc.Spec.VolumeMode = &pattern.VolMode
			}
		default:
			framework.Failf("Volume mode test doesn't support: %s", pattern.VolType)
		}
	}

	cleanup := func() {
		l.cleanupResource()

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}
	}

	// We register different tests depending on the drive
	isBlockSupported := dInfo.Capabilities[CapBlock]
	switch pattern.VolType {
	case testpatterns.PreprovisionedPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			It("should fail to create pod by failing to mount volume", func() {
				init()
				defer cleanup()

				var err error

				By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				l.pv, err = l.cs.CoreV1().PersistentVolumes().Create(l.pv)
				Expect(err).NotTo(HaveOccurred())

				// Prebind pv
				l.pvc.Spec.VolumeName = l.pv.Name
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				Expect(err).NotTo(HaveOccurred())

				framework.ExpectNoError(framework.WaitOnPVandPVC(l.cs, l.ns.Name, l.pv, l.pvc))

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(l.cs, l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, l.config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
				}()
				Expect(err).To(HaveOccurred())
			})
		} else {
			It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
				init()
				defer cleanup()

				var err error

				By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				l.pv, err = l.cs.CoreV1().PersistentVolumes().Create(l.pv)
				Expect(err).NotTo(HaveOccurred())

				// Prebind pv
				l.pvc.Spec.VolumeName = l.pv.Name
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				Expect(err).NotTo(HaveOccurred())

				framework.ExpectNoError(framework.WaitOnPVandPVC(l.cs, l.ns.Name, l.pv, l.pvc))

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(l.cs, l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, l.config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
				}()
				Expect(err).NotTo(HaveOccurred())

				By("Checking if persistent volume exists as expected volume mode")
				utils.CheckVolumeModeOfPath(pod, pattern.VolMode, "/mnt/volume1")

				By("Checking if read/write to persistent volume works properly")
				utils.CheckReadWriteToPath(pod, pattern.VolMode, "/mnt/volume1")
			})
			// TODO(mkimuram): Add more tests
		}
	case testpatterns.DynamicPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			It("should fail in binding dynamic provisioned PV to PVC", func() {
				init()
				defer cleanup()

				var err error

				By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				Expect(err).NotTo(HaveOccurred())

				err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, l.cs, l.pvc.Namespace, l.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
				Expect(err).To(HaveOccurred())
			})
		} else {
			It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
				init()
				defer cleanup()

				var err error

				By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				Expect(err).NotTo(HaveOccurred())

				err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, l.cs, l.pvc.Namespace, l.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
				Expect(err).NotTo(HaveOccurred())

				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.pvc.Namespace).Get(l.pvc.Name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())

				l.pv, err = l.cs.CoreV1().PersistentVolumes().Get(l.pvc.Spec.VolumeName, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(l.cs, l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, l.config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
				}()
				Expect(err).NotTo(HaveOccurred())

				By("Checking if persistent volume exists as expected volume mode")
				utils.CheckVolumeModeOfPath(pod, pattern.VolMode, "/mnt/volume1")

				By("Checking if read/write to persistent volume works properly")
				utils.CheckReadWriteToPath(pod, pattern.VolMode, "/mnt/volume1")
			})
			// TODO(mkimuram): Add more tests
		}
	default:
		framework.Failf("Volume mode test doesn't support volType: %v", pattern.VolType)
	}

}

func generateConfigsForPreprovisionedPVTest(scName string, volBindMode storagev1.VolumeBindingMode,
	volMode v1.PersistentVolumeMode, pvSource v1.PersistentVolumeSource, volumeNodeAffinity *v1.VolumeNodeAffinity) (*storagev1.StorageClass,
	framework.PersistentVolumeConfig, framework.PersistentVolumeClaimConfig) {
	// StorageClass
	scConfig := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		Provisioner:       noProvisioner,
		VolumeBindingMode: &volBindMode,
	}
	// PV
	pvConfig := framework.PersistentVolumeConfig{
		PVSource:         pvSource,
		NodeAffinity:     volumeNodeAffinity,
		NamePrefix:       pvNamePrefix,
		StorageClassName: scName,
		VolumeMode:       &volMode,
	}
	// PVC
	pvcConfig := framework.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &scName,
		VolumeMode:       &volMode,
	}

	return scConfig, pvConfig, pvcConfig
}
