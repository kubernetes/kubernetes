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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
// +k8s:validation-gen-nolint
package lists

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type UpdateListStruct struct {
	TypeMeta int

	// Slice NoSet/NoUnset (len == 0 semantics, no match function needed).

	// +k8s:listType=atomic
	// +k8s:update=NoSet
	StringSliceNoSet []string `json:"stringSliceNoSet"`

	// +k8s:listType=atomic
	// +k8s:update=NoUnset
	StringSliceNoUnset []string `json:"stringSliceNoUnset"`

	// listType=set -> DirectEqual match for directly-comparable elements.

	// +k8s:listType=set
	// +k8s:update=NoAddItem
	StringSetNoAdd []string `json:"stringSetNoAdd"`

	// +k8s:listType=set
	// +k8s:update=NoRemoveItem
	StringSetNoRemove []string `json:"stringSetNoRemove"`

	// +k8s:listType=set
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	StringSetFrozenShape []string `json:"stringSetFrozenShape"`

	// listType=map -> inline listMapKey comparison closure.

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:update=NoAddItem
	MapListNoAdd []UpdateItem `json:"mapListNoAdd"`

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	MapListFrozenShape []UpdateItem `json:"mapListFrozenShape"`

	// listType=map with a composite key (every key field is ANDed).

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:listMapKey=priority
	// +k8s:update=NoRemoveItem
	CompositeKeyList []CompositeKeyItem `json:"compositeKeyList"`

	// listType=atomic + unique={set,map} -> same as the non-atomic variant.

	// +k8s:listType=atomic
	// +k8s:unique=set
	// +k8s:update=NoAddItem
	AtomicUniqueSetNoAdd []string `json:"atomicUniqueSetNoAdd"`

	// +k8s:listType=atomic
	// +k8s:unique=map
	// +k8s:listMapKey=name
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	AtomicUniqueMapFrozenShape []UpdateItem `json:"atomicUniqueMapFrozenShape"`

	// listType=set over a non-directly-comparable element -> SemanticDeepEqual.

	// +k8s:listType=set
	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	NonComparableSetFrozenShape []NonComparableItem `json:"nonComparableSetFrozenShape"`

	// Typedef list: list metadata lives on the type, the field inherits it.

	// +k8s:update=NoAddItem
	// +k8s:update=NoRemoveItem
	TypedefFrozenList FrozenUserList `json:"typedefFrozenList"`

	// eachVal composition: per-item NoModify, list shape can still change.

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:eachVal=+k8s:update=NoModify
	EachValNoModifyList []UpdateItem `json:"eachValNoModifyList"`
}

type UpdateItem struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type CompositeKeyItem struct {
	Name     string `json:"name"`
	Priority int    `json:"priority"`
	Value    string `json:"value"`
}

// NonComparableItem holds a slice, so Go's == is unavailable.
type NonComparableItem struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
}

// +k8s:listType=map
// +k8s:listMapKey=name
type FrozenUserList []UpdateItem
