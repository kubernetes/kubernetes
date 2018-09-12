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

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/types"
)

// GetDriverNameWithFeatureTags returns driver name with feature tags
// For example)
//  - [Driver: nfs]
//  - [Driver: rbd][Feature:Volumes]
func GetDriverNameWithFeatureTags(driver types.TestDriver) string {
	dInfo := driver.GetDriverInfo()

	return fmt.Sprintf("[Driver: %s]%s", dInfo.Name, dInfo.FeatureTag)
}

func CreateVolume(driver types.TestDriver, volType testpatterns.TestVolType) types.DriverTestResources {
	// Create Volume for test unless dynamicPV test
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		if pDriver, ok := driver.(types.PreprovisionedVolumeTestDriver); ok {
			return pDriver.CreateVolume(volType)
		}
	case testpatterns.DynamicPV:
		// No need to create volume
	default:
		framework.Failf("Invalid volType specified: %v", volType)
	}
	return nil
}

func DeleteVolume(driver types.TestDriver, volType testpatterns.TestVolType, dtr types.DriverTestResources) {
	// Delete Volume for test unless dynamicPV test
	switch volType {
	case testpatterns.InlineVolume:
		fallthrough
	case testpatterns.PreprovisionedPV:
		if pDriver, ok := driver.(types.PreprovisionedVolumeTestDriver); ok {
			pDriver.DeleteVolume(volType, dtr)
		}
	case testpatterns.DynamicPV:
		// No need to delete volume
	default:
		framework.Failf("Invalid volType specified: %v", volType)
	}
}

// SetCommonDriverParameters sets a common driver parameters to types.TestDriver
// This function is intended to be called in BeforeEach() inside test loop.
func SetCommonDriverParameters(
	driver types.TestDriver,
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
