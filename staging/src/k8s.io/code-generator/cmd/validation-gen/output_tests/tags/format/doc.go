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
package format

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:format=k8s-ip
	IPField string `json:"ipField"`

	// +k8s:format=k8s-ip
	IPPtrField *string `json:"ipPtrField"`

	// Note: no validation here
	IPTypedefField IPStringType `json:"ipTypedefField"`

	// +k8s:format=k8s-short-name
	DNSLabelField string `json:"dnsLabelField"`

	// +k8s:format=k8s-short-name
	DNSLabelPtrField *string `json:"dnsLabelPtrField"`

	// +k8s:format=k8s-short-name
	DNSLabelTypedefField DNSLabelStringType `json:"dnsLabelTypedefField"`
}

// +k8s:format=k8s-ip
type IPStringType string

// +k8s:format=k8s-short-name
type DNSLabelStringType string
