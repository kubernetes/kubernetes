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
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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
	var (
		dInfo       = driver.GetDriverInfo()
		config      *PerTestConfig
		testCleanup func()
		sc          *storagev1.StorageClass
		pvc         *v1.PersistentVolumeClaim
		pv          *v1.PersistentVolume
		volume      TestVolume
	)

	// No preconditions to test. Normally they would be in a BeforeEach here.

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volumemode")

	init := func() {
		// Now do the more expensive test initialization.
		config, testCleanup = driver.PrepareTest(f)

		ns := f.Namespace
		fsType := pattern.FsType
		volBindMode := storagev1.VolumeBindingImmediate

		var (
			scName             string
			pvSource           *v1.PersistentVolumeSource
			volumeNodeAffinity *v1.VolumeNodeAffinity
		)

		// Create volume for pre-provisioned volume tests
		volume = CreateVolume(driver, config, pattern.VolType)

		switch pattern.VolType {
		case testpatterns.PreprovisionedPV:
			if pattern.VolMode == v1.PersistentVolumeBlock {
				scName = fmt.Sprintf("%s-%s-sc-for-block", ns.Name, dInfo.Name)
			} else if pattern.VolMode == v1.PersistentVolumeFilesystem {
				scName = fmt.Sprintf("%s-%s-sc-for-file", ns.Name, dInfo.Name)
			}
			if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
				pvSource, volumeNodeAffinity = pDriver.GetPersistentVolumeSource(false, fsType, volume)
				if pvSource == nil {
					framework.Skipf("Driver %q does not define PersistentVolumeSource - skipping", dInfo.Name)
				}

				storageClass, pvConfig, pvcConfig := generateConfigsForPreprovisionedPVTest(scName, volBindMode, pattern.VolMode, *pvSource, volumeNodeAffinity)
				sc = storageClass
				pv = framework.MakePersistentVolume(pvConfig)
				pvc = framework.MakePersistentVolumeClaim(pvcConfig, ns.Name)
			}
		case testpatterns.DynamicPV:
			if dDriver, ok := driver.(DynamicPVTestDriver); ok {
				sc = dDriver.GetDynamicProvisionStorageClass(config, fsType)
				if sc == nil {
					framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
				}
				sc.VolumeBindingMode = &volBindMode

				claimSize := dDriver.GetClaimSize()
				pvc = getClaim(claimSize, ns.Name)
				pvc.Spec.StorageClassName = &sc.Name
				pvc.Spec.VolumeMode = &pattern.VolMode
			}
		default:
			framework.Failf("Volume mode test doesn't support: %s", pattern.VolType)
		}
	}

	cleanup := func() {
		if pv != nil || pvc != nil {
			By("Deleting pv and pvc")
			errs := framework.PVPVCCleanup(f.ClientSet, f.Namespace.Name, pv, pvc)
			if len(errs) > 0 {
				framework.Logf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
			}
			pv = nil
			pvc = nil
		}

		if sc != nil {
			By("Deleting sc")
			deleteStorageClass(f.ClientSet, sc.Name)
			sc = nil
		}

		if volume != nil {
			volume.DeleteVolume()
			volume = nil
		}

		if testCleanup != nil {
			testCleanup()
			testCleanup = nil
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

				cs := f.ClientSet
				ns := f.Namespace
				var err error

				By("Creating sc")
				sc, err = cs.StorageV1().StorageClasses().Create(sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				pv, err = cs.CoreV1().PersistentVolumes().Create(pv)
				Expect(err).NotTo(HaveOccurred())

				// Prebind pv
				pvc.Spec.VolumeName = pv.Name
				pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(pvc)
				Expect(err).NotTo(HaveOccurred())

				framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns.Name, pv, pvc))

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
				}()
				Expect(err).To(HaveOccurred())
			})
		} else {
			It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
				init()
				defer cleanup()

				cs := f.ClientSet
				ns := f.Namespace
				var err error

				By("Creating sc")
				sc, err = cs.StorageV1().StorageClasses().Create(sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				pv, err = cs.CoreV1().PersistentVolumes().Create(pv)
				Expect(err).NotTo(HaveOccurred())

				// Prebind pv
				pvc.Spec.VolumeName = pv.Name
				pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(pvc)
				Expect(err).NotTo(HaveOccurred())

				framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns.Name, pv, pvc))

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
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

				cs := f.ClientSet
				ns := f.Namespace
				var err error

				By("Creating sc")
				sc, err = cs.StorageV1().StorageClasses().Create(sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(pvc)
				Expect(err).NotTo(HaveOccurred())

				err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
				Expect(err).To(HaveOccurred())
			})
		} else {
			It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
				init()
				defer cleanup()

				cs := f.ClientSet
				ns := f.Namespace
				var err error

				By("Creating sc")
				sc, err = cs.StorageV1().StorageClasses().Create(sc)
				Expect(err).NotTo(HaveOccurred())

				By("Creating pv and pvc")
				pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(pvc)
				Expect(err).NotTo(HaveOccurred())

				err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
				Expect(err).NotTo(HaveOccurred())

				pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())

				pv, err = cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())

				By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{pvc},
					false, "", false, false, framework.SELinuxLabel,
					nil, config.ClientNodeName, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
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
