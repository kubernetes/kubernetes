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

package framework

import (
	"context"
	"fmt"

	storagev1 "k8s.io/api/storage/v1"
	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
)

// GetDriverNameWithFeatureTags returns parameters that can be passed to framework.Context.
// For example:
//   - [Driver: nfs]
//   - [Driver: rbd], feature.Volumes
func GetDriverNameWithFeatureTags(driver TestDriver) []interface{} {
	dInfo := driver.GetDriverInfo()

	return append([]interface{}{fmt.Sprintf("[Driver: %s]", dInfo.Name)}, dInfo.TestTags...)
}

// CreateVolume creates volume for test unless dynamicPV or CSI ephemeral inline volume test
func CreateVolume(ctx context.Context, driver TestDriver, config *PerTestConfig, volType TestVolType) TestVolume {
	switch volType {
	case InlineVolume, PreprovisionedPV:
		if pDriver, ok := driver.(PreprovisionedVolumeTestDriver); ok {
			return pDriver.CreateVolume(ctx, config, volType)
		}
	case CSIInlineVolume, GenericEphemeralVolume, DynamicPV:
		// No need to create volume
	default:
		framework.Failf("Invalid volType specified: %v", volType)
	}
	return nil
}

// CopyStorageClass constructs a new StorageClass instance
// with a unique name that is based on namespace + suffix
// using the same storageclass setting from the parameter
func CopyStorageClass(sc *storagev1.StorageClass, ns string, suffix string) *storagev1.StorageClass {
	copy := sc.DeepCopy()
	copy.ObjectMeta.Name = names.SimpleNameGenerator.GenerateName(ns + "-" + suffix)
	copy.ResourceVersion = ""

	// Remove the default annotation from the storage class if they exists.
	// Multiple storage classes with this annotation will result in failure.
	delete(copy.Annotations, util.BetaIsDefaultStorageClassAnnotation)
	delete(copy.Annotations, util.IsDefaultStorageClassAnnotation)
	return copy
}

// GetStorageClass constructs a new StorageClass instance
// with a unique name that is based on namespace + suffix.
func GetStorageClass(
	provisioner string,
	parameters map[string]string,
	bindingMode *storagev1.VolumeBindingMode,
	ns string,
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
			// Name must be unique, so let's base it on namespace name and use GenerateName
			Name: names.SimpleNameGenerator.GenerateName(ns),
		},
		Provisioner:       provisioner,
		Parameters:        parameters,
		VolumeBindingMode: bindingMode,
	}
}

// CopyVolumeAttributesClass constructs a new VolumeAttributesClass instance
// with a unique name that is based on namespace + suffix
// using the VolumeAttributesClass passed in as a parameter
func CopyVolumeAttributesClass(vac *storagev1alpha1.VolumeAttributesClass, ns string, suffix string) *storagev1alpha1.VolumeAttributesClass {
	copy := vac.DeepCopy()
	copy.ObjectMeta.Name = names.SimpleNameGenerator.GenerateName(ns + "-" + suffix)
	copy.ResourceVersion = ""
	return copy
}
