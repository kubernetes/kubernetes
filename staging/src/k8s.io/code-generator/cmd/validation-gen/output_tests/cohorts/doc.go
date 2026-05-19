/*
Copyright 2025 The Kubernetes Authors.

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
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package cohorts

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Note: the following are out of order on purpose.
// They should be emitted as:
//   - T {ShortCircuit,Regular}
//   - T c2 {ShortCircuit,Regular}
//   - T c1 {ShortCircuit,Regular}
//
// +k8s:validateFalse(cohort: "c2")="type T c2 Regular"
// +k8s:validateFalse(cohort: "c1")="type T c1 Regular"
// +k8s:validateFalse(cohort: "c1", flags: "ShortCircuit")="type T c1 ShortCircuit"
// +k8s:validateFalse(cohort: "c2", flags: "ShortCircuit")="type T c2 ShortCircuit"
// +k8s:validateFalse="type T Regular"
// +k8s:validateFalse(flags: "ShortCircuit")="type T ShortCircuit"
type T struct {
	TypeMeta int

	// Note: the following are out of order on purpose.
	// They should be emitted as:
	//   - T.S {ShortCircuit,Regular}
	//   - T.S c2 {ShortCircuit,Regular}
	//   - T.S c1 {ShortCircuit,Regular}
	// +k8s:validateFalse(cohort: "c2")="field T.S c2 Regular"
	// +k8s:validateFalse(cohort: "c1")="field T.S c1 Regular"
	// +k8s:validateFalse(cohort: "c1", flags: "ShortCircuit")="field T.S c1 ShortCircuit"
	// +k8s:validateFalse(cohort: "c2", flags: "ShortCircuit")="field T.S c2 ShortCircuit"
	// +k8s:validateFalse="field T.S Regular"
	// +k8s:validateFalse(flags: "ShortCircuit")="field T.S ShortCircuit"
	S string `json:"s"`
}
