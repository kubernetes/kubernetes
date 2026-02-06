/*
Copyright 2026 The Kubernetes Authors.

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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package demonstrating the issue where multiple short-circuit
// validators (immutable, required, maxItems) all run and return errors,
// instead of failing fast on immutability during updates.
//
// Issue: https://github.com/kubernetes/kubernetes/issues/136262
//
// Expected behavior: On update, if a field is modified and marked +k8s:immutable,
// the immutability check should fail first and NOT run required/maxItems checks.
//
// Current behavior: All short-circuit validators run and all errors are returned.
package combinedshortcircuits

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// PodGroup represents a simple struct for testing.
type PodGroup struct {
	// +k8s:required
	Name string `json:"name"`
}

// Struct demonstrates the issue with combined short-circuit validators.
type Struct struct {
	TypeMeta int

	// This field has multiple short-circuit validators.
	// On update, if the field is modified:
	// - immutable should fail first and short-circuit
	// - required and maxItems checks should NOT run
	//
	// +k8s:immutable
	// +k8s:required
	// +k8s:maxItems=8
	PodGroups []PodGroup `json:"podGroups"`

	// Another example with just immutable + required
	// +k8s:immutable
	// +k8s:required
	ImportantField *string `json:"importantField"`
}
