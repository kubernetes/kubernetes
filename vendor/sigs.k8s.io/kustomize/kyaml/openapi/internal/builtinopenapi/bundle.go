// Copyright 2026 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package builtinopenapi defines the on-disk format of the compiled built-in
// Kubernetes OpenAPI bundle.
package builtinopenapi

import (
	"encoding/hex"
	"fmt"
	"sort"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	// FormatVersion is the version of the bundle's JSON representation.
	FormatVersion = 1

	// SelectionPolicy identifies how schemas from Kubernetes releases are
	// selected. A single-release bundle is the degenerate case of this policy.
	SelectionPolicy = "latest-wins-fill-missing"

	gvkExtension = "x-kubernetes-group-version-kind"
)

// Scope describes whether a Kubernetes resource is namespace or cluster
// scoped. An empty Scope means that the source OpenAPI document did not expose
// a resource path from which scope could be determined.
type Scope string

const (
	ScopeUnknown    Scope = ""
	ScopeNamespaced Scope = "Namespaced"
	ScopeCluster    Scope = "Cluster"
)

// Coverage identifies the Kubernetes release range represented by a bundle.
type Coverage struct {
	Floor   string `json:"floor"`
	Ceiling string `json:"ceiling"`
}

// Source identifies one OpenAPI input used to compile a bundle.
type Source struct {
	KubernetesVersion string `json:"kubernetesVersion"`
	SHA256            string `json:"sha256"`
}

// Resource maps a GVK to its root definition and, when known, its scope.
// Definition may be empty for a GVK that was present in an API path but not in
// the OpenAPI definitions.
type Resource struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Definition string `json:"definition,omitempty"`
	Scope      Scope  `json:"scope,omitempty"`
}

// Bundle is the compiled representation consumed by kyaml at runtime.
type Bundle struct {
	FormatVersion   int              `json:"formatVersion"`
	Coverage        Coverage         `json:"coverage"`
	SelectionPolicy string           `json:"selectionPolicy"`
	Sources         []Source         `json:"sources"`
	Definitions     spec.Definitions `json:"definitions"`
	Resources       []Resource       `json:"resources"`
}

// Validate checks invariants required by the runtime loader.
func (b *Bundle) Validate() error {
	if b.FormatVersion != FormatVersion {
		return fmt.Errorf("unsupported built-in OpenAPI bundle format %d", b.FormatVersion)
	}
	if b.Coverage.Floor == "" || b.Coverage.Ceiling == "" {
		return fmt.Errorf("built-in OpenAPI bundle coverage is incomplete")
	}
	if b.SelectionPolicy != SelectionPolicy {
		return fmt.Errorf("unsupported built-in OpenAPI selection policy %q", b.SelectionPolicy)
	}
	if len(b.Sources) == 0 {
		return fmt.Errorf("built-in OpenAPI bundle has no sources")
	}
	for _, source := range b.Sources {
		if source.KubernetesVersion == "" {
			return fmt.Errorf("built-in OpenAPI bundle has a source without a Kubernetes version")
		}
		if len(source.SHA256) != 64 {
			return fmt.Errorf("built-in OpenAPI source %q has an invalid SHA-256", source.KubernetesVersion)
		}
		if _, err := hex.DecodeString(source.SHA256); err != nil {
			return fmt.Errorf("built-in OpenAPI source %q has an invalid SHA-256", source.KubernetesVersion)
		}
	}
	if len(b.Definitions) == 0 {
		return fmt.Errorf("built-in OpenAPI bundle has no definitions")
	}
	if len(b.Resources) == 0 {
		return fmt.Errorf("built-in OpenAPI bundle has no resources")
	}

	resourcesByGVK := make(map[string]Resource, len(b.Resources))
	for i, resource := range b.Resources {
		if resource.APIVersion == "" || resource.Kind == "" {
			return fmt.Errorf("built-in OpenAPI resource %d has an incomplete GVK", i)
		}
		switch resource.Scope {
		case ScopeUnknown, ScopeNamespaced, ScopeCluster:
		default:
			return fmt.Errorf("built-in OpenAPI resource %s/%s has invalid scope %q",
				resource.APIVersion, resource.Kind, resource.Scope)
		}
		if resource.Definition != "" {
			if _, found := b.Definitions[resource.Definition]; !found {
				return fmt.Errorf("built-in OpenAPI resource %s/%s references missing definition %q",
					resource.APIVersion, resource.Kind, resource.Definition)
			}
		}
		key := resourceKey(resource.APIVersion, resource.Kind)
		if _, found := resourcesByGVK[key]; found {
			return fmt.Errorf("built-in OpenAPI resource %s/%s is duplicated",
				resource.APIVersion, resource.Kind)
		}
		resourcesByGVK[key] = resource
		if i > 0 && lessResource(resource, b.Resources[i-1]) {
			return fmt.Errorf("built-in OpenAPI resources are not sorted")
		}
	}
	return validateDefinitionResources(b.Definitions, resourcesByGVK)
}

func validateDefinitionResources(definitions spec.Definitions, resourcesByGVK map[string]Resource) error {
	definitionsByGVK := make(map[string]string)
	for definitionName, definition := range definitions {
		extension, found := definition.Extensions[gvkExtension]
		if !found {
			continue
		}
		entries, ok := extension.([]interface{})
		if !ok {
			return fmt.Errorf("built-in OpenAPI definition %q has malformed %s extension: expected an array",
				definitionName, gvkExtension)
		}
		for i, entry := range entries {
			apiVersion, kind, err := parseGVK(entry)
			if err != nil {
				return fmt.Errorf("built-in OpenAPI definition %q has malformed %s extension entry %d: %w",
					definitionName, gvkExtension, i, err)
			}
			key := resourceKey(apiVersion, kind)
			if previousDefinition, found := definitionsByGVK[key]; found && previousDefinition != definitionName {
				return fmt.Errorf("built-in OpenAPI GVK %s/%s is advertised by definitions %q and %q",
					apiVersion, kind, previousDefinition, definitionName)
			}
			definitionsByGVK[key] = definitionName

			resource, found := resourcesByGVK[key]
			if !found {
				return fmt.Errorf("built-in OpenAPI definition %q advertises GVK %s/%s without a resource mapping",
					definitionName, apiVersion, kind)
			}
			if resource.Definition != definitionName {
				return fmt.Errorf("built-in OpenAPI resource %s/%s references definition %q, but definition %q advertises it",
					apiVersion, kind, resource.Definition, definitionName)
			}
		}
	}

	for _, resource := range resourcesByGVK {
		if resource.Definition == "" {
			continue
		}
		definitionName, found := definitionsByGVK[resourceKey(resource.APIVersion, resource.Kind)]
		if !found {
			return fmt.Errorf("built-in OpenAPI resource %s/%s references definition %q, but that definition does not advertise the GVK",
				resource.APIVersion, resource.Kind, resource.Definition)
		}
		if definitionName != resource.Definition {
			return fmt.Errorf("built-in OpenAPI resource %s/%s references definition %q, but definition %q advertises it",
				resource.APIVersion, resource.Kind, resource.Definition, definitionName)
		}
	}
	return nil
}

func parseGVK(value interface{}) (string, string, error) {
	entry, ok := value.(map[string]interface{})
	if !ok {
		return "", "", fmt.Errorf("expected an object")
	}
	version, versionOK := entry["version"].(string)
	kind, kindOK := entry["kind"].(string)
	if !versionOK || version == "" || !kindOK || kind == "" {
		return "", "", fmt.Errorf("version and kind must be non-empty strings")
	}
	groupValue, hasGroup := entry["group"]
	group, groupOK := groupValue.(string)
	if hasGroup && !groupOK {
		return "", "", fmt.Errorf("group must be a string")
	}
	if group != "" {
		return group + "/" + version, kind, nil
	}
	return version, kind, nil
}

func resourceKey(apiVersion, kind string) string {
	return apiVersion + "\x00" + kind
}

// SortResources orders resources deterministically for serialization.
func SortResources(resources []Resource) {
	sort.Slice(resources, func(i, j int) bool {
		return lessResource(resources[i], resources[j])
	})
}

func lessResource(left, right Resource) bool {
	if left.APIVersion != right.APIVersion {
		return left.APIVersion < right.APIVersion
	}
	return left.Kind < right.Kind
}
