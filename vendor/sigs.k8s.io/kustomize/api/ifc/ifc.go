// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package ifc holds miscellaneous interfaces used by kustomize.
package ifc

import (
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Validator provides functions to validate annotations and labels
type Validator interface {
	MakeAnnotationValidator() func(map[string]string) error
	MakeAnnotationNameValidator() func([]string) error
	MakeLabelValidator() func(map[string]string) error
	MakeLabelNameValidator() func([]string) error
	ValidateNamespace(string) []string
	ErrIfInvalidKey(string) error
	IsEnvVarName(k string) error
}

// KvLoader reads and validates KV pairs.
type KvLoader interface {
	Validator() Validator
	Load(args types.KvPairSources) (all []types.Pair, err error)
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

// KustHasher returns a hash of the argument
// or an error.
type KustHasher interface {
	Hash(*yaml.RNode) (string, error)
}

// See core.v1.SecretTypeOpaque
const SecretTypeOpaque = "Opaque"
