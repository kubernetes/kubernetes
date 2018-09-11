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

package types

import (
	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

//// TEST SUITES

// TestSuite represents an interface for a set of tests whchi works with TestDriver
type TestSuite interface {
	// GetTestSuiteInfo returns the TestSuiteInfo for this TestSuite
	GetTestSuiteInfo() TestSuiteInfo
	// SkipUnsupportedTest skips the test if this TestSuite is not suitable to be tested with the combination of testpatterns.TestPattern and TestDriver
	SkipUnsupportedTest(testpatterns.TestPattern, TestDriver)
	// execTest executes test of the testpattern for the driver
	ExecTest(TestDriver, testpatterns.TestPattern)
}

type TestSuiteInfo struct {
	Name         string                     // name of the TestSuite
	FeatureTag   string                     // featureTag for the TestSuite
	TestPatterns []testpatterns.TestPattern // Slice of testpatterns.TestPattern for the TestSuite
}

// TestResource represents an interface for resources that is used by TestSuite
type TestResource interface {
	// setupResource sets up test resources to be used for the tests with the
	// combination of TestDriver and testpatterns.TestPattern
	SetupResource(TestDriver, testpatterns.TestPattern)
	// cleanupResource clean up the test resources created in SetupResource
	CleanupResource(TestDriver, testpatterns.TestPattern)
}

//// DRIVERS

// TestDriver represents an interface for a driver to be tested in TestSuite
type TestDriver interface {
	// GetDriverInfo returns DriverInfo for the TestDriver
	GetDriverInfo() *DriverInfo
	// CreateDriver creates all driver resources that is required for TestDriver method
	// except CreateVolume
	CreateDriver()
	// CreateDriver cleanup all the resources that is created in CreateDriver
	CleanupDriver()
	// SkipUnsupportedTest skips test in Testpattern is not suitable to test with the TestDriver
	SkipUnsupportedTest(testpatterns.TestPattern)
}

// PreprovisionedVolumeTestDriver represents an interface for a TestDriver that has pre-provisioned volume
type PreprovisionedVolumeTestDriver interface {
	TestDriver
	// CreateVolume creates a pre-provisioned volume.
	CreateVolume(testpatterns.TestVolType) DriverTestResources
	// DeleteVolume deletes a volume that is created in CreateVolume
	DeleteVolume(testpatterns.TestVolType, DriverTestResources)
}

// InlineVolumeTestDriver represents an interface for a TestDriver that supports InlineVolume
type InlineVolumeTestDriver interface {
	PreprovisionedVolumeTestDriver
	// GetVolumeSource returns a volumeSource for inline volume.
	// It will set readOnly and fsType to the volumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetVolumeSource(readOnly bool, fsType string, dtr DriverTestResources) *v1.VolumeSource
}

// PreprovisionedPVTestDriver represents an interface for a TestDriver that supports PreprovisionedPV
type PreprovisionedPVTestDriver interface {
	PreprovisionedVolumeTestDriver
	// GetPersistentVolumeSource returns a PersistentVolumeSource for pre-provisioned Persistent Volume.
	// It will set readOnly and fsType to the PersistentVolumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetPersistentVolumeSource(readOnly bool, fsType string, dtr DriverTestResources) *v1.PersistentVolumeSource
}

// DynamicPVTestDriver represents an interface for a TestDriver that supports DynamicPV
type DynamicPVTestDriver interface {
	TestDriver
	// GetDynamicProvisionStorageClass returns a StorageClass dynamic provision Persistent Volume.
	// It will set fsType to the StorageClass, if TestDriver supports it.
	// It will return nil, if the TestDriver doesn't support it.
	GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass
}

type DriverTestResources interface {
}

// DriverInfo represents a combination of parameters to be used in implementation of TestDriver
type DriverInfo struct {
	Name       string // Name of the driver
	FeatureTag string // FeatureTag for the driver

	MaxFileSize        int64       // Max file size to be tested for this driver
	SupportedFsType    sets.String // Map of string for supported fs type
	IsPersistent       bool        // Flag to represent whether it provides persistency
	IsFsGroupSupported bool        // Flag to represent whether it supports fsGroup
	IsBlockSupported   bool        // Flag to represent whether it supports Block Volume

	// Parameters below will be set inside test loop by using SetCommonDriverParameters.
	// Drivers that implement TestDriver is required to set all the above parameters
	// and return DriverInfo on GetDriverInfo() call.
	Framework *framework.Framework       // Framework for the test
	Config    framework.VolumeTestConfig // VolumeTestConfig for thet test
}
