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
package formats

import "k8s.io/code-generator/cmd/validation-gen/testscheme"

var localSchemeBuilder = testscheme.New()

type Struct struct {
	TypeMeta int

	// A format this project declares.
	// +k8s:optional
	// +k8s:format=example-uri
	RegexField string `json:"regexField"`

	// The same format again: the compiled pattern must be declared only once.
	// +k8s:optional
	// +k8s:format=example-uri
	OtherRegexField string `json:"otherRegexField"`

	// A second format from the same file.
	// +k8s:optional
	// +k8s:format=example-hex-digest
	DigestField string `json:"digestField"`

	// Composes with other tags like a built-in format.
	// +k8s:optional
	// +k8s:eachVal=+k8s:format=example-uri
	SliceField []string `json:"sliceField"`

	// The pattern is package-level, so it is hoisted out of the iteration.
	// +k8s:optional
	// +k8s:eachKey=+k8s:format=example-hex-digest
	MapField map[string]string `json:"mapField"`

	// +k8s:optional
	// +k8s:ifEnabled(Feature)=+k8s:format=example-uri
	GatedField string `json:"gatedField"`

	// +k8s:optional
	TypedefField URI `json:"typedefField"`

	// A built-in format still works alongside the project's.
	// +k8s:optional
	// +k8s:format=k8s-short-name
	BuiltInField string `json:"builtInField"`

	// The only use of example-base64, behind a deferred tag: its pattern must
	// still be declared at package level.
	// +k8s:optional
	// +k8s:eachVal=+k8s:subfield(digest)=+k8s:format=example-base64
	NestedList []Nested `json:"nestedList"`
}

type Nested struct {
	// +k8s:optional
	Digest string `json:"digest"`
}

// +k8s:format=example-uri
type URI string

// ModeStruct is the only use of example-semver, under ifMode: its pattern must
// still be declared at package level.
type ModeStruct struct {
	TypeMeta int

	// +k8s:required
	// +k8s:modeDiscriminator
	Kind string `json:"kind"`

	// +k8s:optional
	// +k8s:ifMode("release")=+k8s:format=example-semver
	Version *string `json:"version,omitempty"`
}
