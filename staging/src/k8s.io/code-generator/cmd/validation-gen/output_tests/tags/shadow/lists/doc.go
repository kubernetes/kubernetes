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

package lists

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type ListTypeMixed struct {
	TypeMeta int

	// +k8s:shadow=+k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:eachVal=+k8s:subfield(inner)=+k8s:subfield(value)=+k8s:minimum=10
	List []ComplexMapItem `json:"list"`

	// +k8s:shadow=+k8s:listType=set
	// +k8s:eachVal=+k8s:subfield(inner)=+k8s:subfield(stringVal)=+k8s:maxLength=5
	Set []ComplexSetItem `json:"set"`

	// +k8s:shadow=+k8s:eachVal=+k8s:maxLength=2
	// +k8s:shadow=+k8s:eachKey=+k8s:maxLength=2
	MapField map[string]string `json:"mapField"`
}

type ComplexMapItem struct {
	Key   string    `json:"key"`
	Inner InnerItem `json:"inner"`
}

type ComplexSetItem struct {
	Inner InnerItem `json:"inner"`
}

type InnerItem struct {
	Value     int    `json:"value"`
	StringVal string `json:"stringVal"`
}

type MapItem struct {
	Key   string `json:"key"`
	Value int    `json:"value"`
}

type MultiKeyItem struct {
	Key1  string `json:"key1"`
	Key2  int    `json:"key2"`
	Value int    `json:"value"`
}

type ListItemStruct struct {
	TypeMeta int

	// Case 1: Shadowed list (map), Non-shadowed Item validation
	// +k8s:listMapKey=key
	// +k8s:shadow=+k8s:listType=map
	// +k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:minimum=10
	ShadowListNonShadowItem []MapItem `json:"shadowListNonShadowItem"`

	// Case 2: Shadowed list, Shadowed Item validation
	// +k8s:listMapKey=key
	// +k8s:shadow=+k8s:listType=map
	// +k8s:shadow=+k8s:item(key: "foo")=+k8s:subfield(value)=+k8s:minimum=10
	ShadowListShadowItem []MapItem `json:"shadowListShadowItem"`

	// Case 3: Non-shadowed list, Mixed Items (one shadowed, one normal)
	// +k8s:listType=map
	// +k8s:listMapKey=key
	// +k8s:shadow=+k8s:item(key: "shadow")=+k8s:subfield(value)=+k8s:minimum=10
	// +k8s:item(key: "normal")=+k8s:subfield(value)=+k8s:minimum=10
	MixedItems []MapItem `json:"mixedItems"`

	// Case 4: Multiple Keys
	// +k8s:listType=map
	// +k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	// +k8s:shadow=+k8s:item(key1: "a", key2: 10)=+k8s:subfield(value)=+k8s:minimum=10
	MultiKeyItems []MultiKeyItem `json:"multiKeyItems"`

	// case 5: listMapKey non shadowed
	// +k8s:shadow=+k8s:listType=map
	// +k8s:listMapKey=key
	MapList []MapItem `json:"mapList"`
}
