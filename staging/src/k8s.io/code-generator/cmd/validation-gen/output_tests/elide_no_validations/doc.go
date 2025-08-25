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
package elidenovalidations

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type T1 struct {
	TypeMeta int

	HasTypeVal HasTypeVal `json:"hasTypeVal"`

	HasFieldVal HasFieldVal `json:"hasFieldVal"`

	HasNoVal HasNoVal `json:"hasNoVal"`

	// +k8s:validateFalse="field T1.HasNoValFieldVal"
	HasNoValFieldVal HasNoVal `json:"hasNoValFieldVal"`
}

// +k8s:validateFalse="type HasTypeVal"
type HasTypeVal struct {
	// Note: no field validation.
	S string `json:"s"`
}

// Note: no type validation.
type HasFieldVal struct {
	// +k8s:validateFalse="field HasFieldVal.S"
	S string `json:"s"`
}

// Note: no type validation.
type HasNoVal struct {
	// Note: no field validation.
	S string `json:"s"`
}

// +k8s:validateFalse="type HasTypeValNotLinked"
type HasTypeValNotLinked struct {
	// Note: no field validation.
	S string `json:"s"`
}

// Note: no type validation.
type HasFieldValNotLinked struct {
	// +k8s:validateFalse="field HasFieldValNotLinked.S"
	S string `json:"s"`
}

// Note: no type validation.
type HasNoValNotLinked struct {
	// Note: no field validation.
	S string `json:"s"`
}
