/*
Copyright The Kubernetes Authors.

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

// Package install registers the metadata internal and versioned types with a
// runtime.Scheme. Consumers should call NewScheme() to get a scheme that can
// decode any supported metadata version into the internal types.
package install

import (
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/api/metadata/v1alpha1"
)

// Install registers the internal and v1alpha1 metadata types with the given scheme.
func Install(scheme *runtime.Scheme) {
	utilruntime.Must(metadata.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
}

// NewScheme returns a new runtime.Scheme with all metadata versions registered.
// The returned scheme can decode any supported metadata version into the
// internal metadata.DeviceMetadata type.
func NewScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	Install(scheme)
	return scheme
}
