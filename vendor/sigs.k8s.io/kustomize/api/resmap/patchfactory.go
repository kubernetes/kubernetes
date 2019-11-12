/// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package patch holds miscellaneous interfaces used by kustomize.
package resmap

import (
	"sigs.k8s.io/kustomize/api/resource"
)

// PatchFactory makes transformers that require k8sdeps.
type PatchFactory interface {
	MergePatches(patches []*resource.Resource,
		rf *resource.Factory) (ResMap, error)
}
