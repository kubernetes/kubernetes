/// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package patch holds miscellaneous interfaces used by kustomize.
package transformer

import (
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers"
)

// Factory makes transformers that require k8sdeps.
type Factory interface {
	MakePatchTransformer(
		slice []*resource.Resource,
		rf *resource.Factory) (transformers.Transformer, error)
}
