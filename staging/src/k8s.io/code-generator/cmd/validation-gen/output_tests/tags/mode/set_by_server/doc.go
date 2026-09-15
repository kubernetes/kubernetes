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

// +k8s:validation-gen=TypesWithField=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
// +k8s:validation-gen-nolint
package setbyserver

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type TypeMeta int

type StructWithModeSetByServer struct {
	TypeMeta int

	// +k8s:modeDiscriminator
	Mode string `json:"mode"`

	// +k8s:optional
	// +k8s:ifMode("Auto")=+k8s:setByServer
	AutoSetField *string `json:"autoSetField,omitempty"`

	// +k8s:ifMode("Manual")=+k8s:optional
	// +k8s:ifMode("Manual")=+k8s:setByServer
	ManualField *string `json:"manualField,omitempty"`
}
