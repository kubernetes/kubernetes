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

package util

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledAlphaFields removes disabled fields from the StorageClass object.
func DropDisabledAlphaFields(class *storage.StorageClass) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		class.VolumeBindingMode = nil
		class.AllowedTopologies = nil
	}
}
