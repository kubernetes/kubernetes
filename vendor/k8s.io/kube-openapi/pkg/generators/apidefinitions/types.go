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

package apidefinitions

// Metadata is the kubernetes-style metadata block used by codegen manifests.
type Metadata struct {
	Name string `json:"name"`
}

// APIVersion declares an external versioned API package, such as
// staging/src/k8s.io/api/<group>/<version>/. Each entry corresponds to one
// served GroupVersion.
type APIVersion struct {
	APIVersion string         `json:"apiVersion"`
	Kind       string         `json:"kind"`
	Metadata   Metadata       `json:"metadata"`
	Spec       APIVersionSpec `json:"spec"`
}

// APIVersionSpec provides a specification for an APIVersion.
type APIVersionSpec struct {
	// ModelPackage is the OpenAPI model package name for this group/version
	// (e.g. "io.k8s.api.apps.v1").
	ModelPackage string `json:"modelPackage,omitempty"`
}
