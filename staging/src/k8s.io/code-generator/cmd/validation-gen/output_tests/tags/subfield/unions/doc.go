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

// Package unions contains test types for testing subfield union validation tags.
// +k8s:validation-gen-nolint
package unions

import (
	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int `json:"typeMeta"`

	// +k8s:subfield(d)=+k8s:unionDiscriminator
	// +k8s:subfield(m1)=+k8s:unionMember
	// +k8s:subfield(m2)=+k8s:unionMember
	Subfield SubStruct `json:"subfield"`
}

type SubStruct struct {
	D  string `json:"d"`
	M1 *int   `json:"m1,omitempty"`
	M2 *int   `json:"m2,omitempty"`
}
