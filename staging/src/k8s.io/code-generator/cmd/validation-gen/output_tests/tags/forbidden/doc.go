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
package forbidden

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// +k8s:forbidden
	StringField string `json:"stringField"`

	// +k8s:forbidden
	StringPtrField *string `json:"stringPtrField"`

	// +k8s:forbidden
	StringTypedefField StringType `json:"stringTypedefField"`

	// +k8s:forbidden
	StringTypedefPtrField *StringType `json:"stringTypedefPtrField"`

	// +k8s:forbidden
	IntField int `json:"intField"`

	// +k8s:forbidden
	IntPtrField *int `json:"intPtrField"`

	// +k8s:forbidden
	IntTypedefField IntType `json:"intTypedefField"`

	// +k8s:forbidden
	IntTypedefPtrField *IntType `json:"intTypedefPtrField"`

	// +k8s:forbidden
	BoolField bool `json:"boolField"`

	// +k8s:forbidden
	FloatField float64 `json:"floatField"`

	// +k8s:forbidden
	ByteField byte `json:"byteField"`

	// +k8s:forbidden
	OtherStructPtrField *OtherStruct `json:"otherStructPtrField"`

	// +k8s:forbidden
	SliceField []string `json:"sliceField"`

	// +k8s:forbidden
	SliceTypedefField SliceType `json:"sliceTypedefField"`

	// +k8s:forbidden
	ByteArrayField []byte `json:"byteArrayField"`

	// +k8s:forbidden
	MapField map[string]string `json:"mapField"`

	// +k8s:forbidden
	MapTypedefField MapType `json:"mapTypedefField"`
}

type StringType string

type IntType int

type OtherStruct struct{}

type SliceType []string

type MapType map[string]string
