// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package ifc holds miscellaneous interfaces used by kustomize.
package ifc

import (
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/types"
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

// Kunstructured represents a Kubernetes Resource Model object.
type Kunstructured interface {
	// Several uses.
	Copy() Kunstructured

	// GetAnnotations returns the k8s annotations.
	GetAnnotations() map[string]string

	// GetData returns a top-level "data" field, as in a ConfigMap.
	GetDataMap() map[string]string

	// GetData returns a top-level "binaryData" field, as in a ConfigMap.
	GetBinaryDataMap() map[string]string

	// Used by ResAccumulator and ReplacementTransformer.
	GetFieldValue(string) (interface{}, error)

	// Used by Resource.OrgId
	GetGvk() resid.Gvk

	// Used by resource.Factory.SliceFromBytes
	GetKind() string

	// GetLabels returns the k8s labels.
	GetLabels() map[string]string

	// Used by Resource.CurId and resource factory.
	GetName() string

	// Used by special case code in
	// ResMap.SubsetThatCouldBeReferencedByResource
	GetSlice(path string) ([]interface{}, error)

	// GetString returns the value of a string field.
	// Used by Resource.GetNamespace
	GetString(string) (string, error)

	// Several uses.
	Map() (map[string]interface{}, error)

	// Used by Resource.AsYAML and Resource.String
	MarshalJSON() ([]byte, error)

	// Used by resWrangler.Select
	MatchesAnnotationSelector(selector string) (bool, error)

	// Used by resWrangler.Select
	MatchesLabelSelector(selector string) (bool, error)

	// SetAnnotations replaces the k8s annotations.
	SetAnnotations(map[string]string)

	// SetDataMap sets a top-level "data" field, as in a ConfigMap.
	SetDataMap(map[string]string)

	// SetDataMap sets a top-level "binaryData" field, as in a ConfigMap.
	SetBinaryDataMap(map[string]string)
	// Used by PatchStrategicMergeTransformer.
	SetGvk(resid.Gvk)

	// SetLabels replaces the k8s labels.
	SetLabels(map[string]string)

	// SetName changes the name.
	SetName(string)

	// SetNamespace changes the namespace.
	SetNamespace(string)

	// Needed, for now, by kyaml/filtersutil.ApplyToJSON.
	UnmarshalJSON([]byte) error
}

// KunstructuredFactory makes instances of Kunstructured.
type KunstructuredFactory interface {
	SliceFromBytes([]byte) ([]Kunstructured, error)
	FromMap(m map[string]interface{}) Kunstructured
	Hasher() KunstructuredHasher
	MakeConfigMap(kvLdr KvLoader, args *types.ConfigMapArgs) (Kunstructured, error)
	MakeSecret(kvLdr KvLoader, args *types.SecretArgs) (Kunstructured, error)
}

// KunstructuredHasher returns a hash of the argument
// or an error.
type KunstructuredHasher interface {
	Hash(Kunstructured) (string, error)
}

// See core.v1.SecretTypeOpaque
const SecretTypeOpaque = "Opaque"
