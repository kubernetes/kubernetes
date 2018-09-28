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
	"k8s.io/kubernetes/test/e2e/storage/drivers"
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
			name:       "volumeMode",
			featureTag: "[Feature:BlockVolume]",
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

func (t *volumeModeTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func createVolumeModeTestInput(pattern testpatterns.TestPattern, resource volumeModeTestResource) volumeModeTestInput {
	driver := resource.driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework

	return volumeModeTestInput{
		f:                f,
		sc:               resource.sc,
		pvc:              resource.pvc,
		pv:               resource.pv,
		testVolType:      pattern.VolType,
		nodeName:         dInfo.Config.ClientNodeName,
		volMode:          pattern.VolMode,
		isBlockSupported: dInfo.IsBlockSupported,
	}
}

func getVolumeModeTestFunc(pattern testpatterns.TestPattern, driver drivers.TestDriver) func(*volumeModeTestInput) {
	dInfo := driver.GetDriverInfo()
	isBlockSupported := dInfo.IsBlockSupported
	volMode := pattern.VolMode
	volType := pattern.VolType

	switch volType {
	case testpatterns.PreprovisionedPV:
		if volMode == v1.PersistentVolumeBlock && !isBlockSupported {
			return testVolumeModeFailForPreprovisionedPV
		}
		return testVolumeModeSuccessForPreprovisionedPV
	case testpatterns.DynamicPV:
		if volMode == v1.PersistentVolumeBlock && !isBlockSupported {
			return testVolumeModeFailForDynamicPV
		}
		return testVolumeModeSuccessForDynamicPV
	default:
		framework.Failf("Volume mode test doesn't support volType: %v", volType)
	}
	return nil
}

func (t *volumeModeTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(t, pattern), func() {
		var (
			resource     volumeModeTestResource
			input        volumeModeTestInput
			testFunc     func(*volumeModeTestInput)
			needsCleanup bool
		)

		testFunc = getVolumeModeTestFunc(pattern, driver)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(t, driver, pattern)
			needsCleanup = true

			// Setup test resource for driver and testpattern
			resource = volumeModeTestResource{}
			resource.setupResource(driver, pattern)

			// Create test input
			input = createVolumeModeTestInput(pattern, resource)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		testFunc(&input)
	})
}

type volumeModeTestResource struct {
	driver drivers.TestDriver

	sc  *storagev1.StorageClass
	pvc *v1.PersistentVolumeClaim
	pv  *v1.PersistentVolume

	driverTestResource interface{}
}

var _ TestResource = &volumeModeTestResource{}

func (s *volumeModeTestResource) setupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	s.driver = driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework
	ns := f.Namespace
	fsType := pattern.FsType
	volBindMode := storagev1.VolumeBindingImmediate
	volMode := pattern.VolMode
	volType := pattern.VolType

	var (
		scName   string
		pvSource *v1.PersistentVolumeSource
	)

	// Create volume for pre-provisioned volume tests
	s.driverTestResource = drivers.CreateVolume(driver, volType)

	switch volType {
	case testpatterns.PreprovisionedPV:
		if volMode == v1.PersistentVolumeBlock {
			scName = fmt.Sprintf("%s-%s-sc-for-block", ns.Name, dInfo.Name)
		} else if volMode == v1.PersistentVolumeFilesystem {
			scName = fmt.Sprintf("%s-%s-sc-for-file", ns.Name, dInfo.Name)
		}
		if pDriver, ok := driver.(drivers.PreprovisionedPVTestDriver); ok {
			pvSource = pDriver.GetPersistentVolumeSource(false, fsType, s.driverTestResource)
			if pvSource == nil {
				framework.Skipf("Driver %q does not define PersistentVolumeSource - skipping", dInfo.Name)
			}

			sc, pvConfig, pvcConfig := generateConfigsForPreprovisionedPVTest(scName, volBindMode, volMode, *pvSource)
			s.sc = sc
			s.pv = framework.MakePersistentVolume(pvConfig)
			s.pvc = framework.MakePersistentVolumeClaim(pvcConfig, ns.Name)
		}
	case testpatterns.DynamicPV:
		if dDriver, ok := driver.(drivers.DynamicPVTestDriver); ok {
			s.sc = dDriver.GetDynamicProvisionStorageClass(fsType)
			if s.sc == nil {
				framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
			}
			s.sc.VolumeBindingMode = &volBindMode

			claimSize := "2Gi"
			s.pvc = getClaim(claimSize, ns.Name)
			s.pvc.Spec.StorageClassName = &s.sc.Name
			s.pvc.Spec.VolumeMode = &volMode
		}
	default:
		framework.Failf("Volume mode test doesn't support: %s", volType)
	}
}

func (s *volumeModeTestResource) cleanupResource(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	dInfo := driver.GetDriverInfo()
	f := dInfo.Framework
	cs := f.ClientSet
	ns := f.Namespace
	volType := pattern.VolType

	By("Deleting pv and pvc")
	errs := framework.PVPVCCleanup(cs, ns.Name, s.pv, s.pvc)
	if len(errs) > 0 {
		framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
	}
	By("Deleting sc")
	if s.sc != nil {
		deleteStorageClass(cs, s.sc.Name)
	}

	// Cleanup volume for pre-provisioned volume tests
	drivers.DeleteVolume(driver, volType, s.driverTestResource)
}

type volumeModeTestInput struct {
	f                *framework.Framework
	sc               *storagev1.StorageClass
	pvc              *v1.PersistentVolumeClaim
	pv               *v1.PersistentVolume
	testVolType      testpatterns.TestVolType
	nodeName         string
	volMode          v1.PersistentVolumeMode
	isBlockSupported bool
}

func testVolumeModeFailForPreprovisionedPV(input *volumeModeTestInput) {
	It("should fail to create pod by failing to mount volume", func() {
		f := input.f
		cs := f.ClientSet
		ns := f.Namespace
		var err error

		By("Creating sc")
		input.sc, err = cs.StorageV1().StorageClasses().Create(input.sc)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pv and pvc")
		input.pv, err = cs.CoreV1().PersistentVolumes().Create(input.pv)
		Expect(err).NotTo(HaveOccurred())

		// Prebind pv
		input.pvc.Spec.VolumeName = input.pv.Name
		input.pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(input.pvc)
		Expect(err).NotTo(HaveOccurred())

		framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns.Name, input.pv, input.pvc))

		By("Creating pod")
		pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{input.pvc},
			false, "", false, false, framework.SELinuxLabel,
			nil, input.nodeName, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).To(HaveOccurred())
	})
}

func testVolumeModeSuccessForPreprovisionedPV(input *volumeModeTestInput) {
	It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
		f := input.f
		cs := f.ClientSet
		ns := f.Namespace
		var err error

		By("Creating sc")
		input.sc, err = cs.StorageV1().StorageClasses().Create(input.sc)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pv and pvc")
		input.pv, err = cs.CoreV1().PersistentVolumes().Create(input.pv)
		Expect(err).NotTo(HaveOccurred())

		// Prebind pv
		input.pvc.Spec.VolumeName = input.pv.Name
		input.pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(input.pvc)
		Expect(err).NotTo(HaveOccurred())

		framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns.Name, input.pv, input.pvc))

		By("Creating pod")
		pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{input.pvc},
			false, "", false, false, framework.SELinuxLabel,
			nil, input.nodeName, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred())

		By("Checking if persistent volume exists as expected volume mode")
		checkVolumeModeOfPath(pod, input.volMode, "/mnt/volume1")

		By("Checking if read/write to persistent volume works properly")
		checkReadWriteToPath(pod, input.volMode, "/mnt/volume1")
	})
	// TODO(mkimuram): Add more tests
}

func testVolumeModeFailForDynamicPV(input *volumeModeTestInput) {
	It("should fail in binding dynamic provisioned PV to PVC", func() {
		f := input.f
		cs := f.ClientSet
		ns := f.Namespace
		var err error

		By("Creating sc")
		input.sc, err = cs.StorageV1().StorageClasses().Create(input.sc)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pv and pvc")
		input.pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(input.pvc)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, input.pvc.Namespace, input.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).To(HaveOccurred())
	})
}

func testVolumeModeSuccessForDynamicPV(input *volumeModeTestInput) {
	It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
		f := input.f
		cs := f.ClientSet
		ns := f.Namespace
		var err error

		By("Creating sc")
		input.sc, err = cs.StorageV1().StorageClasses().Create(input.sc)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pv and pvc")
		input.pvc, err = cs.CoreV1().PersistentVolumeClaims(ns.Name).Create(input.pvc)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, input.pvc.Namespace, input.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())

		input.pvc, err = cs.CoreV1().PersistentVolumeClaims(input.pvc.Namespace).Get(input.pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		input.pv, err = cs.CoreV1().PersistentVolumes().Get(input.pvc.Spec.VolumeName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("Creating pod")
		pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{input.pvc},
			false, "", false, false, framework.SELinuxLabel,
			nil, input.nodeName, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred())

		By("Checking if persistent volume exists as expected volume mode")
		checkVolumeModeOfPath(pod, input.volMode, "/mnt/volume1")

		By("Checking if read/write to persistent volume works properly")
		checkReadWriteToPath(pod, input.volMode, "/mnt/volume1")
	})
	// TODO(mkimuram): Add more tests
}

func generateConfigsForPreprovisionedPVTest(scName string, volBindMode storagev1.VolumeBindingMode,
	volMode v1.PersistentVolumeMode, pvSource v1.PersistentVolumeSource) (*storagev1.StorageClass,
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

func checkVolumeModeOfPath(pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// Check if block exists
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("test -b %s", path))

		// Double check that it's not directory
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("test -d %s", path), 1)
	} else {
		// Check if directory exists
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("test -d %s", path))

		// Double check that it's not block
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("test -b %s", path), 1)
	}
}

func checkReadWriteToPath(pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// random -> file1
		utils.VerifyExecInPodSucceed(pod, "dd if=/dev/urandom of=/tmp/file1 bs=64 count=1")
		// file1 -> dev (write to dev)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("dd if=/tmp/file1 of=%s bs=64 count=1", path))
		// dev -> file2 (read from dev)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("dd if=%s of=/tmp/file2 bs=64 count=1", path))
		// file1 == file2 (check contents)
		utils.VerifyExecInPodSucceed(pod, "diff /tmp/file1 /tmp/file2")
		// Clean up temp files
		utils.VerifyExecInPodSucceed(pod, "rm -f /tmp/file1 /tmp/file2")

		// Check that writing file to block volume fails
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path), 1)
	} else {
		// text -> file1 (write to file)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path))
		// grep file1 (read from file and check contents)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("grep 'Hello world.' %s/file1.txt", path))

		// Check that writing to directory as block volume fails
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("dd if=/dev/urandom of=%s bs=64 count=1", path), 1)
	}
}
