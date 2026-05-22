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
package unions

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	UnionField         Union         `json:"unionField"`
	UnionFieldDisabled UnionDisabled `json:"unionFieldDisabled"`

	ZeroOrOneOfField         ZeroOrOneOf         `json:"zeroOrOneOfField"`
	ZeroOrOneOfFieldDisabled ZeroOrOneOfDisabled `json:"zeroOrOneOfFieldDisabled"`

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:ifEnabled(FeatureX)=+k8s:item(name: "succeeded")=+k8s:zeroOrOneOfMember
	// +k8s:ifEnabled(FeatureX)=+k8s:item(name: "failed")=+k8s:zeroOrOneOfMember
	ZeroOrOneOfItem []Task `json:"zeroOrOneOfItem"`

	// +k8s:listType=map
	// +k8s:listMapKey=name
	// +k8s:ifDisabled(FeatureX)=+k8s:item(name: "succeeded")=+k8s:zeroOrOneOfMember
	// +k8s:ifDisabled(FeatureX)=+k8s:item(name: "failed")=+k8s:zeroOrOneOfMember
	ZeroOrOneOfItemDisabled []Task `json:"zeroOrOneOfItemDisabled"`
}

type Task struct {
	Name  string `json:"name"`
	State string `json:"state"`
}

type Union struct {
	// +k8s:ifEnabled(FeatureX)=+k8s:unionDiscriminator
	Discriminator string `json:"discriminator"`

	// +k8s:ifEnabled(FeatureX)=+k8s:unionMember
	XEnabledField string `json:"xEnabledField"`

	// +k8s:ifEnabled(FeatureX)=+k8s:unionMember
	XDisabledField string `json:"xDisabledField"`
}

type UnionDisabled struct {
	// +k8s:ifDisabled(FeatureX)=+k8s:unionDiscriminator
	Discriminator string `json:"discriminator"`

	// +k8s:ifDisabled(FeatureX)=+k8s:unionMember
	XEnabledField string `json:"xEnabledField"`

	// +k8s:ifDisabled(FeatureX)=+k8s:unionMember
	XDisabledField string `json:"xDisabledField"`
}

type ZeroOrOneOf struct {
	// +k8s:ifEnabled(FeatureX)=+k8s:zeroOrOneOfMember
	XEnabledField string `json:"xEnabledField"`

	// +k8s:ifEnabled(FeatureX)=+k8s:zeroOrOneOfMember
	XDisabledField string `json:"xDisabledField"`
}

type ZeroOrOneOfDisabled struct {
	// +k8s:ifDisabled(FeatureX)=+k8s:zeroOrOneOfMember
	XEnabledField string `json:"xEnabledField"`

	// +k8s:ifDisabled(FeatureX)=+k8s:zeroOrOneOfMember
	XDisabledField string `json:"xDisabledField"`
}
