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
package typedef

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int `json:"typeMeta"`

	TypedefItems       ItemList      `json:"typedefItems"`
	NestedTypedefItems ItemListAlias `json:"nestedTypedefItems"`

	// +k8s:item(id: "field-target")=+k8s:validateFalse="item DualItems[id=field-target] from field"
	DualItems DualItemList `json:"dualItems"`
}

type Item struct {
	Key  string `json:"key"`
	Data string `json:"data"`
}

// +k8s:listType=map
// +k8s:listMapKey=key
// +k8s:item(key: "immutable")=+k8s:immutable
// +k8s:item(key: "validated")=+k8s:validateFalse="item ItemList[key=validated]"
type ItemList []Item

// +k8s:listType=map
// +k8s:listMapKey=key
// +k8s:item(key: "aliased")=+k8s:validateFalse="item ItemListAlias[key=aliased]"
type ItemListAlias ItemList

type DualItem struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// +k8s:listType=map
// +k8s:listMapKey=id
// +k8s:item(id: "typedef-target")=+k8s:validateFalse="item DualItems[id=typedef-target] from typedef"
type DualItemList []DualItem
