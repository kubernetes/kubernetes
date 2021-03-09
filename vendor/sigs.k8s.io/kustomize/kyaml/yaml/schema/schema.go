// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package schema contains libraries for working with the yaml and openapi packages.
package schema

import (
	"strings"

	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// IsAssociative returns true if all elements in the list contain an
// AssociativeSequenceKey as a field.
func IsAssociative(schema *openapi.ResourceSchema, nodes []*yaml.RNode, infer bool) bool {
	if schema != nil {
		return schemaHasMergeStrategy(schema)
	}
	if !infer {
		return false
	}
	for i := range nodes {
		node := nodes[i]
		if yaml.IsMissingOrNull(node) {
			continue
		}
		if node.IsAssociative() {
			return true
		}
	}
	return false
}

func schemaHasMergeStrategy(schema *openapi.ResourceSchema) bool {
	tmp, _ := schema.PatchStrategyAndKey()
	strategies := strings.Split(tmp, ",")
	for _, s := range strategies {
		if s == "merge" {
			return true
		}
	}
	return false
}
