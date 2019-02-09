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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

// TestSuite represents an interface for a set of tests which works with TestDriver
type TestSuite interface {
	// getTestSuiteInfo returns the TestSuiteInfo for this TestSuite
	getTestSuiteInfo() TestSuiteInfo
	// skipUnsupportedTest skips the test if this TestSuite is not suitable to be tested with the combination of TestPattern and TestDriver
	skipUnsupportedTest(testpatterns.TestPattern, TestDriver)
	// execTest executes test of the testpattern for the driver
	execTest(TestDriver, testpatterns.TestPattern)
}

// TestSuiteInfo represents a set of parameters for TestSuite
type TestSuiteInfo struct {
	name         string                     // name of the TestSuite
	featureTag   string                     // featureTag for the TestSuite
	testPatterns []testpatterns.TestPattern // Slice of TestPattern for the TestSuite
}

// TestResource represents an interface for resources that is used by TestSuite
type TestResource interface {
	// setupResource sets up test resources to be used for the tests with the
	// combination of TestDriver and TestPattern
	setupResource(TestDriver, testpatterns.TestPattern)
	// cleanupResource clean up the test resources created in SetupResource
	cleanupResource(TestDriver, testpatterns.TestPattern)
}

func getTestNameStr(suite TestSuite, pattern testpatterns.TestPattern) string {
	tsInfo := suite.getTestSuiteInfo()
	return fmt.Sprintf("[Testpattern: %s]%s %s%s", pattern.Name, pattern.FeatureTag, tsInfo.name, tsInfo.featureTag)
}

// RunTestSuite runs all testpatterns of all testSuites for a driver
func RunTestSuite(f *framework.Framework, driver TestDriver, tsInits []func() TestSuite, tunePatternFunc func([]testpatterns.TestPattern) []testpatterns.TestPattern) {
	for _, testSuiteInit := range tsInits {
		suite := testSuiteInit()
		patterns := tunePatternFunc(suite.getTestSuiteInfo().testPatterns)

		for _, pattern := range patterns {
			suite.execTest(driver, pattern)
		}
	}
}

// skipUnsupportedTest will skip tests if the combination of driver, testsuite, and testpattern
// is not suitable to be tested.
// Whether it needs to be skipped is checked by following steps:
// 1. Check if Whether SnapshotType is supported by driver from its interface
// 2. Check if Whether volType is supported by driver from its interface
// 3. Check if fsType is supported
// 4. Check with driver specific logic
// 5. Check with testSuite specific logic
func skipUnsupportedTest(suite TestSuite, driver TestDriver, pattern testpatterns.TestPattern) {
	dInfo := driver.GetDriverInfo()
	var isSupported bool

	// 1. Check if Whether SnapshotType is supported by driver from its interface
	// if isSupported, we still execute the driver and suite tests
	if len(pattern.SnapshotType) > 0 {
		switch pattern.SnapshotType {
		case testpatterns.DynamicCreatedSnapshot:
			_, isSupported = driver.(SnapshottableTestDriver)
		default:
			isSupported = false
		}
		if !isSupported {
			framework.Skipf("Driver %s doesn't support snapshot type %v -- skipping", dInfo.Name, pattern.SnapshotType)
		}
	} else {
		// 2. Check if Whether volType is supported by driver from its interface
		switch pattern.VolType {
		case testpatterns.InlineVolume:
			_, isSupported = driver.(InlineVolumeTestDriver)
		case testpatterns.PreprovisionedPV:
			_, isSupported = driver.(PreprovisionedPVTestDriver)
		case testpatterns.DynamicPV:
			_, isSupported = driver.(DynamicPVTestDriver)
		default:
			isSupported = false
		}

		if !isSupported {
			framework.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}

		// 3. Check if fsType is supported
		if !dInfo.SupportedFsType.Has(pattern.FsType) {
			framework.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.FsType)
		}
		if pattern.FsType == "xfs" && framework.NodeOSDistroIs("gci") {
			framework.Skipf("Distro doesn't support xfs -- skipping")
		}
	}

	// 4. Check with driver specific logic
	driver.SkipUnsupportedTest(pattern)

	// 5. Check with testSuite specific logic
	suite.skipUnsupportedTest(pattern, driver)
}

// genericVolumeTestResource is a generic implementation of TestResource that wil be able to
// be used in most of TestSuites.
// See volume_io.go or volumes.go in test/e2e/storage/testsuites/ for how to use this resource.
// Also, see subpath.go in the same directory for how to extend and use it.
type genericVolumeTestResource struct {
	driver    TestDriver
	volType   string
	volSource *v1.VolumeSource
	pvc       *v1.PersistentVolumeClaim
	pv        *v1.PersistentVolume
	sc        *storagev1.StorageClass

	driverTestResource interface{}
}

var _ TestResource = &genericVolumeTestResource{}

// setupResource sets up genericVolumeTestResource
func (r *genericVolumeTestResource) setupResource(driver TestDriver, pattern testpatterns.TestPattern) {
	r.driver = driver
	dInfo := driver.GetDriverInfo()
	f := dInfo.Config.Framework
	cs := f.ClientSet
	fsType := pattern.FsType
	volType := pattern.VolType

	// Create volume for pre-provisioned volume tests
	r.driverTestResource = CreateVolume(driver, volType)

	switch volType {
	case testpatterns.InlineVolume:
		framework.Logf("Creating resource for inline volume")
		if iDriver, ok := driver.(InlineVolumeTestDriver); ok {
			r.volSource = iDriver.GetVolumeSource(false, fsType, r.driverTestResource)
			r.volType = dInfo.Name
		}
	case testpatterns.PreprovisionedPV:
		framework.Logf("Creating resource for pre-provisioned PV")
		if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
			pvSource, volumeNodeAffinity := pDriver.GetPersistentVolumeSource(false, fsType, r.driverTestResource)
			if pvSource != nil {
				r.volSource, r.pv, r.pvc = createVolumeSourceWithPVCPV(f, dInfo.Name, pvSource, volumeNodeAffinity, false)
			}
			r.volType = fmt.Sprintf("%s-preprovisionedPV", dInfo.Name)
		}
	case testpatterns.DynamicPV:
		framework.Logf("Creating resource for dynamic PV")
		if dDriver, ok := driver.(DynamicPVTestDriver); ok {
			claimSize := dDriver.GetClaimSize()
			r.sc = dDriver.GetDynamicProvisionStorageClass(fsType)

			By("creating a StorageClass " + r.sc.Name)
			var err error
			r.sc, err = cs.StorageV1().StorageClasses().Create(r.sc)
			Expect(err).NotTo(HaveOccurred())

			if r.sc != nil {
				r.volSource, r.pv, r.pvc = createVolumeSourceWithPVCPVFromDynamicProvisionSC(
					f, dInfo.Name, claimSize, r.sc, false, nil)
			}
			r.volType = fmt.Sprintf("%s-dynamicPV", dInfo.Name)
		}
	default:
		framework.Failf("genericVolumeTestResource doesn't support: %s", volType)
	}

	if r.volSource == nil {
		framework.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, volType)
	}
}

// cleanupResource cleans up genericVolumeTestResource
func (r *genericVolumeTestResource) cleanupResource(driver TestDriver, pattern testpatterns.TestPattern) {
	dInfo := driver.GetDriverInfo()
	f := dInfo.Config.Framework
	volType := pattern.VolType

	if r.pvc != nil || r.pv != nil {
		switch volType {
		case testpatterns.PreprovisionedPV:
			By("Deleting pv and pvc")
			if errs := framework.PVPVCCleanup(f.ClientSet, f.Namespace.Name, r.pv, r.pvc); len(errs) != 0 {
				framework.Failf("Failed to delete PVC or PV: %v", utilerrors.NewAggregate(errs))
			}
		case testpatterns.DynamicPV:
			By("Deleting pvc")
			// We only delete the PVC so that PV (and disk) can be cleaned up by dynamic provisioner
			if r.pv.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimDelete {
				framework.Failf("Test framework does not currently support Dynamically Provisioned Persistent Volume %v specified with reclaim policy that isnt %v",
					r.pv.Name, v1.PersistentVolumeReclaimDelete)
			}
			err := framework.DeletePersistentVolumeClaim(f.ClientSet, r.pvc.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to delete PVC %v", r.pvc.Name)
			err = framework.WaitForPersistentVolumeDeleted(f.ClientSet, r.pv.Name, 5*time.Second, 5*time.Minute)
			framework.ExpectNoError(err, "Persistent Volume %v not deleted by dynamic provisioner", r.pv.Name)
		default:
			framework.Failf("Found PVC (%v) or PV (%v) but not running Preprovisioned or Dynamic test pattern", r.pvc, r.pv)
		}
	}

	if r.sc != nil {
		By("Deleting sc")
		deleteStorageClass(f.ClientSet, r.sc.Name)
	}

	// Cleanup volume for pre-provisioned volume tests
	DeleteVolume(driver, volType, r.driverTestResource)
}

func createVolumeSourceWithPVCPV(
	f *framework.Framework,
	name string,
	pvSource *v1.PersistentVolumeSource,
	volumeNodeAffinity *v1.VolumeNodeAffinity,
	readOnly bool,
) (*v1.VolumeSource, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	pvConfig := framework.PersistentVolumeConfig{
		NamePrefix:       fmt.Sprintf("%s-", name),
		StorageClassName: f.Namespace.Name,
		PVSource:         *pvSource,
		NodeAffinity:     volumeNodeAffinity,
	}
	pvcConfig := framework.PersistentVolumeClaimConfig{
		StorageClassName: &f.Namespace.Name,
	}

	framework.Logf("Creating PVC and PV")
	pv, pvc, err := framework.CreatePVCPV(f.ClientSet, pvConfig, pvcConfig, f.Namespace.Name, false)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV creation failed")

	err = framework.WaitOnPVandPVC(f.ClientSet, f.Namespace.Name, pv, pvc)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV failed to bind")

	volSource := &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: pvc.Name,
			ReadOnly:  readOnly,
		},
	}
	return volSource, pv, pvc
}

func createVolumeSourceWithPVCPVFromDynamicProvisionSC(
	f *framework.Framework,
	name string,
	claimSize string,
	sc *storagev1.StorageClass,
	readOnly bool,
	volMode *v1.PersistentVolumeMode,
) (*v1.VolumeSource, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	cs := f.ClientSet
	ns := f.Namespace.Name

	By("creating a claim")
	pvc := getClaim(claimSize, ns)
	pvc.Spec.StorageClassName = &sc.Name
	if volMode != nil {
		pvc.Spec.VolumeMode = volMode
	}

	var err error
	pvc, err = cs.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
	Expect(err).NotTo(HaveOccurred())

	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	pv, err := cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	volSource := &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: pvc.Name,
			ReadOnly:  readOnly,
		},
	}
	return volSource, pv, pvc
}

func getClaim(claimSize string, ns string) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(claimSize),
				},
			},
		},
	}

	return &claim
}

// deleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func deleteStorageClass(cs clientset.Interface, className string) {
	err := cs.StorageV1().StorageClasses().Delete(className, nil)
	if err != nil && !apierrs.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}
}

// convertTestConfig returns a framework test config with the
// parameters specified for the testsuite or (if available) the
// dynamically created config for the volume server.
//
// This is done because TestConfig is the public API for
// the testsuites package whereas framework.VolumeTestConfig is merely
// an implementation detail. It contains fields that have no effect,
// which makes it unsuitable for use in the testsuits public API.
func convertTestConfig(in *TestConfig) framework.VolumeTestConfig {
	if in.ServerConfig != nil {
		return *in.ServerConfig
	}

	return framework.VolumeTestConfig{
		Namespace:      in.Framework.Namespace.Name,
		Prefix:         in.Prefix,
		ClientNodeName: in.ClientNodeName,
		NodeSelector:   in.ClientNodeSelector,
	}
}

func getSnapshot(claimName string, ns, snapshotClassName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshot",
			"apiVersion": snapshotAPIVersion,
			"metadata": map[string]interface{}{
				"generateName": "snapshot-",
				"namespace":    ns,
			},
			"spec": map[string]interface{}{
				"snapshotClassName": snapshotClassName,
				"source": map[string]interface{}{
					"name": claimName,
					"kind": "PersistentVolumeClaim",
				},
			},
		},
	}

	return snapshot
}
