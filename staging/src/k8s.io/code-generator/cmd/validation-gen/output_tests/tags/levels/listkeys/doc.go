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

package listkeys

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type MapItem struct {
	Key   string `json:"key"`
	Value int    `json:"value"`
}

type MultiKeyItem struct {
	Key1  string `json:"key1"`
	Key2  int    `json:"key2"`
	Value int    `json:"value"`
}

type ListKeyStruct struct {
	TypeMeta int

	// Case: Alpha listType, Standard Key
	// +k8s:alpha=+k8s:listType=map
	// +k8s:listMapKey=key
	AlphaListTypeStandardKey []MapItem `json:"alphaListTypeStandardKey"`

	// Case: Standard listType, Alpha Key
	// +k8s:listType=map
	// +k8s:alpha=+k8s:listMapKey=key
	StandardListTypeAlphaKey []MapItem `json:"standardListTypeAlphaKey"`

	// Case: Alpha listType, Alpha Key
	// +k8s:alpha=+k8s:listType=map
	// +k8s:alpha=+k8s:listMapKey=key
	AlphaListTypeAlphaKey []MapItem `json:"alphaListTypeAlphaKey"`

	// Case: Standard listType, Alpha Key1, Standard Key2
	// +k8s:listType=map
	// +k8s:alpha=+k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	StandardListTypeMixedKeys1 []MultiKeyItem `json:"standardListTypeMixedKeys1"`

	// Case: Standard listType, Standard Key1, Alpha Key2
	// +k8s:listType=map
	// +k8s:listMapKey=key1
	// +k8s:alpha=+k8s:listMapKey=key2
	StandardListTypeMixedKeys2 []MultiKeyItem `json:"standardListTypeMixedKeys2"`

	// Case: Alpha listType, Alpha Key1, Standard Key2
	// +k8s:alpha=+k8s:listType=map
	// +k8s:alpha=+k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	AlphaListTypeMixedKeys1 []MultiKeyItem `json:"alphaListTypeMixedKeys1"`

	// Case: Alpha listType, Standard Key1, Alpha Key2
	// +k8s:alpha=+k8s:listType=map
	// +k8s:listMapKey=key1
	// +k8s:alpha=+k8s:listMapKey=key2
	AlphaListTypeMixedKeys2 []MultiKeyItem `json:"alphaListTypeMixedKeys2"`

	// Case: Standard listType, Beta Key1, Standard Key2
	// +k8s:listType=map
	// +k8s:beta=+k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	StandardListTypeMixedKeysBeta1 []MultiKeyItem `json:"standardListTypeMixedKeysBeta1"`

	// Case: Standard listType, Standard Key1, Beta Key2
	// +k8s:listType=map
	// +k8s:listMapKey=key1
	// +k8s:beta=+k8s:listMapKey=key2
	StandardListTypeMixedKeysBeta2 []MultiKeyItem `json:"standardListTypeMixedKeysBeta2"`

	// Case: Beta listType, Beta Key1, Standard Key2
	// +k8s:beta=+k8s:listType=map
	// +k8s:beta=+k8s:listMapKey=key1
	// +k8s:listMapKey=key2
	BetaListTypeMixedKeys1 []MultiKeyItem `json:"betaListTypeMixedKeys1"`

	// Case: Beta listType, Standard Key1, Beta Key2
	// +k8s:beta=+k8s:listType=map
	// +k8s:listMapKey=key1
	// +k8s:beta=+k8s:listMapKey=key2
	BetaListTypeMixedKeys2 []MultiKeyItem `json:"betaListTypeMixedKeys2"`

	// Case: Beta listType, Standard Key
	// +k8s:beta=+k8s:listType=map
	// +k8s:listMapKey=key
	BetaListTypeStandardKey []MapItem `json:"betaListTypeStandardKey"`

	// Case: Standard listType, Beta Key
	// +k8s:listType=map
	// +k8s:beta=+k8s:listMapKey=key
	StandardListTypeBetaKey []MapItem `json:"standardListTypeBetaKey"`

	// Case: Beta listType, Beta Key
	// +k8s:beta=+k8s:listType=map
	// +k8s:beta=+k8s:listMapKey=key
	BetaListTypeBetaKey []MapItem `json:"betaListTypeBetaKey"`
}
