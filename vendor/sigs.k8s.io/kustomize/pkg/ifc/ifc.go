// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

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
	ErrIfInvalidKey(string) error
	IsEnvVarName(k string) error
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
	// Validator validates data for use in various k8s fields.
	Validator() Validator
	// Loads pairs.
	LoadKvPairs(args types.GeneratorArgs) ([]types.Pair, error)
}

// Kunstructured allows manipulation of k8s objects
// that do not have Golang structs.
type Kunstructured interface {
	Map() map[string]interface{}
	SetMap(map[string]interface{})
	Copy() Kunstructured
	GetFieldValue(string) (interface{}, error)
	GetString(string) (string, error)
	GetStringSlice(string) ([]string, error)
	GetBool(path string) (bool, error)
	GetFloat64(path string) (float64, error)
	GetInt64(path string) (int64, error)
	GetSlice(path string) ([]interface{}, error)
	GetStringMap(path string) (map[string]string, error)
	GetMap(path string) (map[string]interface{}, error)
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
	Hasher() KunstructuredHasher
	MakeConfigMap(
		ldr Loader,
		options *types.GeneratorOptions,
		args *types.ConfigMapArgs) (Kunstructured, error)
	MakeSecret(
		ldr Loader,
		options *types.GeneratorOptions,
		args *types.SecretArgs) (Kunstructured, error)
}

// KunstructuredHasher returns a hash of the argument
// or an error.
type KunstructuredHasher interface {
	Hash(Kunstructured) (string, error)
}

// See core.v1.SecretTypeOpaque
const SecretTypeOpaque = "Opaque"
