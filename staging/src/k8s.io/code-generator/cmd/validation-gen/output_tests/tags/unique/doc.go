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

// This is a test package.
package unique

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// Basic unique=set on primitive slice
	// +k8s:listType=atomic
	// +k8s:unique=set
	PrimitiveListUniqueSet []string `json:"primitiveListUniqueSet"`

	// unique=map with multiple keys
	// +k8s:listType=atomic
	// +k8s:unique=map
	// +k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	SliceMapFieldWithMultipleKeys []ItemWithMultipleKeys `json:"sliceMapFieldWithMultipleKeys"`

	// atomic + unique=set combination
	// +k8s:listType=atomic
	// +k8s:unique=set
	AtomicListUniqueSet []Item `json:"atomicListUniqueSet"`

	// atomic + unique=map combination
	// +k8s:listType=atomic
	// +k8s:unique=map
	// +k8s:listMapKey=key
	AtomicListUniqueMap []Item `json:"atomicListUniqueMap"`
}

type Item struct {
	Key  string `json:"key"`
	Data string `json:"data"`
}

type ItemWithMultipleKeys struct {
	Key1 string `json:"key1"`
	Key2 string `json:"key2"`
	Data string `json:"data"`
}
