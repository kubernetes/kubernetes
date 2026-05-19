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
package singlekey

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int `json:"typeMeta"`

	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "target")=+k8s:validateFalse="item Items[key=target] 1"
	// +k8s:item(key: "target")=+k8s:validateFalse="item Items[key=target] 2"
	Items []Item `json:"items"`

	// +k8s:listType=map
	// +k8s:listMapKey=intField
	// +k8s:item(intField: 42)=+k8s:validateFalse="item IntKeyItems[intField=42] 1"
	// +k8s:item(intField: 42)=+k8s:validateFalse="item IntKeyItems[intField=42] 2"
	IntKeyItems []IntKeyItem `json:"intKeyItems"`

	// +k8s:listType=map
	// +k8s:listMapKey=boolField
	// +k8s:item(boolField: true)=+k8s:validateFalse="item BoolKeyItems[boolField=true] 1"
	// +k8s:item(boolField: true)=+k8s:validateFalse="item BoolKeyItems[boolField=true] 2"
	BoolKeyItems []BoolKeyItem `json:"boolKeyItems"`

	// +k8s:listType=map
	// +k8s:listMapKey=id
	// +k8s:item(id: "typedef-target")=+k8s:validateFalse="item TypedefItems[id=typedef-target] 1"
	// +k8s:item(id: "typedef-target")=+k8s:validateFalse="item TypedefItems[id=typedef-target] 2"
	TypedefItems TypedefItemList `json:"typedefItems"`

	// Test atomic + unique=map + item combination
	// +k8s:listType=atomic
	// +k8s:unique=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "target")=+k8s:validateFalse="item AtomicUniqueMapItems[key=target]"
	AtomicUniqueMapItems []Item `json:"atomicUniqueMapItems"`

	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "target-ptr")=+k8s:validateFalse="item PtrKeyItems[key=target-ptr]"
	PtrKeyItems []PtrKeyItem `json:"ptrKeyItems"`
}

type StructWithNestedTypedef struct {
	TypeMeta int `json:"typeMeta"`

	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:item(key: "nested-target")=+k8s:validateFalse="item NestedItems[key=nested-target]"
	NestedItems []NestedTypedefItem `json:"nestedItems"`
}

type Item struct {
	Key  string `json:"key"`
	Data string `json:"data"`
}

type IntKeyItem struct {
	IntField int    `json:"intField"`
	Data     string `json:"data"`
}

type BoolKeyItem struct {
	BoolField bool   `json:"boolField"`
	Data      string `json:"data"`
}

type TypedefItem struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

type TypedefItemList []TypedefItem

type StringAlias string
type NestedTypedefItem struct {
	Key  StringAlias `json:"key"`
	Name string      `json:"name"`
}

type PtrKeyItem struct {
	Key  *string `json:"key"`
	Data string  `json:"data"`
}
