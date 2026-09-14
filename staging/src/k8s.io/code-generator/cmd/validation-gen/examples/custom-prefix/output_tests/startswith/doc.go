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

// +xyz:validation-gen=TypesWithField=TypeMeta
// +xyz:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package startswith

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +xyz:optional
	// +xyz:startsWith="abc"
	StringField string `json:"stringField"`

	// +xyz:optional
	// +xyz:startsWith="abc"
	StringPtrField *string `json:"stringPtrField"`

	// +xyz:optional
	// +xyz:eachVal=+xyz:startsWith="abc"
	SliceField []string `json:"sliceField"`

	// +xyz:optional
	// +xyz:ifEnabled(Feature)=+xyz:startsWith="abc"
	GatedField string `json:"gatedField"`

	// +xyz:optional
	TypedefField StringType `json:"typedefField"`
}

// +xyz:startsWith="abc"
type StringType string
