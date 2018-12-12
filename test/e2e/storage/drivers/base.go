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

package drivers

import (
	"fmt"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

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
	CreateVolume(testpatterns.TestVolType) interface{}
	// DeleteVolume deletes a volume that is created in CreateVolume
	DeleteVolume(testpatterns.TestVolType, interface{})
}

// InlineVolumeTestDriver represents an interface for a TestDriver that supports InlineVolume
type InlineVolumeTestDriver interface {
	PreprovisionedVolumeTestDriver
	// GetVolumeSource returns a volumeSource for inline volume.
	// It will set readOnly and fsType to the volumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetVolumeSource(readOnly bool, fsType string, testResource interface{}) *v1.VolumeSource
}

// PreprovisionedPVTestDriver represents an interface for a TestDriver that supports PreprovisionedPV
type PreprovisionedPVTestDriver interface {
	PreprovisionedVolumeTestDriver
	// GetPersistentVolumeSource returns a PersistentVolumeSource for pre-provisioned Persistent Volume.
	// It will set readOnly and fsType to the PersistentVolumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetPersistentVolumeSource(readOnly bool, fsType string, testResource interface{}) *v1.PersistentVolumeSource
}

// DynamicPVTestDriver represents an interface for a TestDriver that supports DynamicPV
type DynamicPVTestDriver interface {
	TestDriver
	// GetDynamicProvisionStorageClass returns a StorageClass dynamic provision Persistent Volume.
	// It will set fsType to the StorageClass, if TestDriver supports it.
	// It will return nil, if the TestDriver doesn't support it.
	GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass
}

// Capability represents a feature that a volume plugin supports
type Capability string

const (
	CapPersistence Capability = "persistence" // data is persisted across pod restarts
	CapBlock       Capability = "block"       // raw block mode
	CapFsGroup     Capability = "fsGroup"     // volume ownership via fsGroup
	CapExec        Capability = "exec"        // exec a file in the volume
)

// DriverInfo represents a combination of parameters to be used in implementation of TestDriver
type DriverInfo struct {
	Name       string // Name of the driver
	FeatureTag string // FeatureTag for the driver

	MaxFileSize          int64               // Max file size to be tested for this driver
	SupportedFsType      sets.String         // Map of string for supported fs type
	SupportedMountOption sets.String         // Map of string for supported mount option
	RequiredMountOption  sets.String         // Map of string for required mount option (Optional)
	Capabilities         map[Capability]bool // Map that represents plugin capabilities

	// Parameters below will be set inside test loop by using SetCommonDriverParameters.
	// Drivers that implement TestDriver is required to set all the above parameters
	// and return DriverInfo on GetDriverInfo() call.
	Framework *framework.Framework       // Framework for the test
	Config    framework.VolumeTestConfig // VolumeTestConfig for thet test
}

// GetDriverNameWithFeatureTags returns driver name with feature tags
// For example)
//  - [Driver: nfs]
//  - [Driver: rbd][Feature:Volumes]
func GetDriverNameWithFeatureTags(driver TestDriver) string {
	dInfo := driver.GetDriverInfo()

	return fmt.Sprintf("[Driver: %s]%s", dInfo.Name, dInfo.FeatureTag)
}

// CreateVolume creates volume for test unless dynamicPV test
func CreateVolume(driver TestDriver, volType testpatterns.TestVolType) interface{} {
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		if pDriver, ok := driver.(PreprovisionedVolumeTestDriver); ok {
			return pDriver.CreateVolume(volType)
		}
	case testpatterns.DynamicPV:
		// No need to create volume
	default:
		framework.Failf("Invalid volType specified: %v", volType)
	}
	return nil
}

// DeleteVolume deletes volume for test unless dynamicPV test
func DeleteVolume(driver TestDriver, volType testpatterns.TestVolType, testResource interface{}) {
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		if pDriver, ok := driver.(PreprovisionedVolumeTestDriver); ok {
			pDriver.DeleteVolume(volType, testResource)
		}
	case testpatterns.DynamicPV:
		// No need to delete volume
	default:
		framework.Failf("Invalid volType specified: %v", volType)
	}
}

// SetCommonDriverParameters sets a common driver parameters to TestDriver
// This function is intended to be called in BeforeEach() inside test loop.
func SetCommonDriverParameters(
	driver TestDriver,
	f *framework.Framework,
	config framework.VolumeTestConfig,
) {
	dInfo := driver.GetDriverInfo()

	dInfo.Framework = f
	dInfo.Config = config
}

func getStorageClass(
	provisioner string,
	parameters map[string]string,
	bindingMode *storagev1.VolumeBindingMode,
	ns string,
	suffix string,
) *storagev1.StorageClass {
	if bindingMode == nil {
		defaultBindingMode := storagev1.VolumeBindingImmediate
		bindingMode = &defaultBindingMode
	}
	return &storagev1.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			// Name must be unique, so let's base it on namespace name
			Name: ns + "-" + suffix,
		},
		Provisioner:       provisioner,
		Parameters:        parameters,
		VolumeBindingMode: bindingMode,
	}
}

// GetUniqueDriverName returns unique driver name that can be used parallelly in tests
func GetUniqueDriverName(driver TestDriver) string {
	return fmt.Sprintf("%s-%s", driver.GetDriverInfo().Name, driver.GetDriverInfo().Framework.UniqueName)
}
