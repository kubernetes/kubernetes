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

// This is a test package.
// +k8s:validation-gen-nolint
package lists

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:ifEnabled(FeatureX)=+k8s:listType=map
	// +k8s:ifEnabled(FeatureX)=+k8s:listMapKey=name
	ListMap []ListItem `json:"listMap"`

	// +k8s:ifDisabled(FeatureX)=+k8s:listType=map
	// +k8s:ifDisabled(FeatureX)=+k8s:listMapKey=name
	ListMapDisabled []ListItem `json:"listMapDisabled"`

	// +k8s:ifEnabled(FeatureX)=+k8s:eachVal=+k8s:validateFalse="field Struct.ListEachVal/val"
	ListEachVal []ListItem `json:"listEachVal"`

	// +k8s:ifDisabled(FeatureX)=+k8s:eachVal=+k8s:validateFalse="field Struct.ListEachValDisabled/val"
	ListEachValDisabled []ListItem `json:"listEachValDisabled"`
}

type ListItem struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}
