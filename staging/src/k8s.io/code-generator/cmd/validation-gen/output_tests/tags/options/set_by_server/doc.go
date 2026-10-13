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

type Struct struct {
	TypeMeta int

	// +k8s:optional
	// +k8s:ifEnabled(FeatureX)=+k8s:setByServer
	XEnabledField string `json:"xEnabledField"`

	// +k8s:optional
	// +k8s:ifDisabled(FeatureX)=+k8s:setByServer
	XDisabledField string `json:"xDisabledField"`

	// +k8s:ifEnabled(FeatureY)=+k8s:optional
	// +k8s:ifEnabled(FeatureY)=+k8s:setByServer
	YEnabledField *string `json:"yEnabledField"`

	// +k8s:optional
	// +k8s:ifDisabled(FeatureY)=+k8s:setByServer
	YDisabledField *string `json:"yDisabledField"`

	// +k8s:optional
	// +k8s:ifEnabled(FeatureX)=+k8s:subfield(xEnabledSubfield)=+k8s:setByServer
	// +k8s:ifDisabled(FeatureY)=+k8s:subfield(yDisabledSubfield)=+k8s:setByServer
	SubStruct *Submarker `json:"subStruct"`
}

type Submarker struct {
	// +k8s:optional
	XEnabledSubfield string `json:"xEnabledSubfield"`
	// +k8s:optional
	YDisabledSubfield string `json:"yDisabledSubfield"`
}
