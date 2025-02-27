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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-test-fixture=validateFalse

// This is a test package.
package structs

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Tother struct {
	// +k8s:validateFalse={"flags":[], "msg":"Tother, no flags"}
	OS string `json:"os"`
}

// Treat these as 4 bits, and ensure all combinations
//   bit 0: no flags
//   bit 1: ShortCircuit

// Note: No validations.
type T00 struct {
	TypeMeta int
	S        string  `json:"s"`
	PS       *string `json:"ps"`
	T        Tother  `json:"t"`
	PT       *Tother `json:"pt"`
}

// +k8s:validateFalse={"flags":[], "msg":"T01, no flags"}
type T01 struct {
	TypeMeta int
	// +k8s:validateFalse={"flags":[], "msg":"T01.S, no flags"}
	S string `json:"s"`
	// +k8s:validateFalse={"flags":[], "msg":"T01.PS, no flags"}
	PS *string `json:"ps"`
	// +k8s:validateFalse={"flags":[], "msg":"T01.T, no flags"}
	T Tother `json:"t"`
	// +k8s:validateFalse={"flags":[], "msg":"T01.PT, no flags"}
	PT *Tother `json:"pt"`
}

// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T02, ShortCircuit"}
type T02 struct {
	TypeMeta int
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T02.S, ShortCircuit"}
	S string `json:"s"`
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T02.PS, ShortCircuit"}
	PS *string `json:"ps"`
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T02.T, ShortCircuit"}
	T Tother `json:"t"`
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T02.PT, ShortCircuit"}
	PT *Tother `json:"pt"`
}

// +k8s:validateFalse={"flags":[], "msg":"T03, no flags"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T03, ShortCircuit"}
type T03 struct {
	TypeMeta int
	// +k8s:validateFalse={"flags":[], "msg":"T03.S, no flags"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T03.S, ShortCircuit"}
	S string `json:"s"`
	// +k8s:validateFalse={"flags":[], "msg":"T03.PS, no flags"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T03.PS, ShortCircuit"}
	PS *string `json:"ps"`
	// +k8s:validateFalse={"flags":[], "msg":"T03.T, no flags"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T03.T, ShortCircuit"}
	T Tother `json:"t"`
	// +k8s:validateFalse={"flags":[], "msg":"T03.PT, no flags"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"T03.PT, ShortCircuit"}
	PT *Tother `json:"pt"`
}

// Note: these are intentionally in the wrong final order.
// +k8s:validateFalse={"flags":[], "msg":"TMultiple, no flags 1"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple, ShortCircuit 1"}
// +k8s:validateFalse="T0, string payload"
// +k8s:validateFalse={"flags":[], "msg":"TMultiple, no flags 2"}
// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple, ShortCircuit 2"}
type TMultiple struct {
	TypeMeta int
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.S, no flags 1"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.S, ShortCircuit 1"}
	// +k8s:validateFalse="T0, string payload"
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.S, no flags 2"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.S, ShortCircuit 2"}
	S string `json:"s"`
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.PS, no flags 1"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.PS, ShortCircuit 1"}
	// +k8s:validateFalse="T0, string payload"
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.PS, no flags 2"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.PS, ShortCircuit 2"}
	PS *string `json:"ps"`
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.T, no flags 1"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.T, ShortCircuit 1"}
	// +k8s:validateFalse="T0, string payload"
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.T, no flags 2"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.T, ShortCircuit 2"}
	T Tother `json:"t"`
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.PT, no flags 1"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.PT, ShortCircuit 1"}
	// +k8s:validateFalse="T0, string payload"
	// +k8s:validateFalse={"flags":[], "msg":"TMultiple.PT, no flags 2"}
	// +k8s:validateFalse={"flags":["ShortCircuit"], "msg":"TMultiple.PT, ShortCircuit 2"}
	PT *Tother `json:"pt"`
}
