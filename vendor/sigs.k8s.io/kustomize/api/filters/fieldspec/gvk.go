// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0
package fieldspec

import (
	"strings"

	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Return true for 'v' followed by a 1 or 2, and don't look at rest.
// I.e. 'v1', 'v1beta1', 'v2', would return true.
func looksLikeACoreApiVersion(s string) bool {
	if len(s) < 2 {
		return false
	}
	if s[0:1] != "v" {
		return false
	}
	return s[1:2] == "1" || s[1:2] == "2"
}

// parseGV parses apiVersion field into group and version.
func parseGV(apiVersion string) (group, version string) {
	// parse the group and version from the apiVersion field
	parts := strings.SplitN(apiVersion, "/", 2)
	group = parts[0]
	if len(parts) > 1 {
		version = parts[1]
	}
	// Special case the original "apiVersion" of what
	// we now call the "core" (empty) group.
	if version == "" && looksLikeACoreApiVersion(group) {
		version = group
		group = ""
	}
	return
}

// GetGVK parses the metadata into a GVK
func GetGVK(meta yaml.ResourceMeta) resid.Gvk {
	group, version := parseGV(meta.APIVersion)
	return resid.Gvk{
		Group:   group,
		Version: version,
		Kind:    meta.Kind,
	}
}
