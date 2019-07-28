/*
Copyright 2018 The Kubernetes Authors.

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

// Package ifc holds miscellaneous interfaces used by kustomize.
package ifc

import (
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/types"
)

// Validator provides functions to validate annotations and labels
type Validator interface {
	MakeAnnotationValidator() func(map[string]string) error
	MakeLabelValidator() func(map[string]string) error
	ValidateNamespace(string) []string
}

// Loader interface exposes methods to read bytes.
type Loader interface {
	// Root returns the root location for this Loader.
	Root() string
	// New returns Loader located at newRoot.
	New(newRoot string) (Loader, error)
	// Load returns the bytes read from the location or an error.
	Load(location string) ([]byte, error)
	// Cleanup cleans the loader
	Cleanup() error
}

// Kunstructured allows manipulation of k8s objects
// that do not have Golang structs.
type Kunstructured interface {
	Map() map[string]interface{}
	SetMap(map[string]interface{})
	Copy() Kunstructured
	GetFieldValue(string) (string, error)
	MarshalJSON() ([]byte, error)
	UnmarshalJSON([]byte) error
	GetGvk() gvk.Gvk
	GetKind() string
	GetName() string
	SetName(string)
	GetLabels() map[string]string
	SetLabels(map[string]string)
	GetAnnotations() map[string]string
	SetAnnotations(map[string]string)
}

// KunstructuredFactory makes instances of Kunstructured.
type KunstructuredFactory interface {
	SliceFromBytes([]byte) ([]Kunstructured, error)
	FromMap(m map[string]interface{}) Kunstructured
	MakeConfigMap(args *types.ConfigMapArgs, options *types.GeneratorOptions) (Kunstructured, error)
	MakeSecret(args *types.SecretArgs, options *types.GeneratorOptions) (Kunstructured, error)
	Set(ldr Loader)
}

// See core.v1.SecretTypeOpaque
const SecretTypeOpaque = "Opaque"
