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

package policyconfig

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ValidatingAdmissionPolicyConfiguration provides configuration for the validating admission policy controller.
type ValidatingAdmissionPolicyConfiguration struct {
	metav1.TypeMeta

	// StaticManifestsDir is the path to a directory containing static
	// ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding
	// resources to be loaded at startup. Files with extensions .yaml,
	// .yml, and .json are read. Only admissionregistration.k8s.io/v1
	// resources are supported.
	// Using this field requires the ManifestBasedAdmissionControlConfig
	// feature gate to be enabled.
	// +optional
	StaticManifestsDir string
}

// GetStaticManifestsDir returns the static manifests directory path.
func (c *ValidatingAdmissionPolicyConfiguration) GetStaticManifestsDir() string {
	return c.StaticManifestsDir
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MutatingAdmissionPolicyConfiguration provides configuration for the mutating admission policy controller.
type MutatingAdmissionPolicyConfiguration struct {
	metav1.TypeMeta

	// StaticManifestsDir is the path to a directory containing static
	// MutatingAdmissionPolicy and MutatingAdmissionPolicyBinding
	// resources to be loaded at startup. Files with extensions .yaml,
	// .yml, and .json are read. Only admissionregistration.k8s.io/v1
	// resources are supported.
	// Using this field requires the ManifestBasedAdmissionControlConfig
	// feature gate to be enabled.
	// +optional
	StaticManifestsDir string
}

// GetStaticManifestsDir returns the static manifests directory path.
func (c *MutatingAdmissionPolicyConfiguration) GetStaticManifestsDir() string {
	return c.StaticManifestsDir
}
