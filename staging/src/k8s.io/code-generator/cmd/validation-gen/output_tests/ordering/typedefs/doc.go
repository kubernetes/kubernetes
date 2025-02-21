/*
Copyright 2024 The Kubernetes Authors.

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

// +k8s:validation-gen=*
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package typedefs

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

// Treat these as 4 bits, and ensure all combinations
//   bit 0: no flags
//   bit 1: ShortCircuit

// Note: No validations.
type E00 string

// +k8s:validateFalse={"flags":[], "msg":"E01, no flags"}
type E01 string

// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"E02, ShortCircuit"}
type E02 string

// +k8s:validateFalse={"flags":[], "msg":"E03, no flags"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"E03, ShortCircuit"}
type E03 string

// Note: these are intentionally in the wrong final order.
// +k8s:validateFalse={"flags":[], "msg":"EMultiple, no flags 1"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"EMultiple, ShortCircuit 1"}
// +k8s:validateFalse="E0, string payload"
// +k8s:validateFalse={"flags":[], "msg":"EMultiple, no flags 2"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"EMultiple, ShortCircuit 2"}
type EMultiple string
