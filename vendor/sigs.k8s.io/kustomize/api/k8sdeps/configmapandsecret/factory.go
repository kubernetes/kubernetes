// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package configmapandsecret

import (
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/types"
)

// Factory makes ConfigMaps and Secrets.
type Factory struct {
	kvLdr   ifc.KvLoader
	options *types.GeneratorOptions
}

// NewFactory returns a new factory that makes ConfigMaps and Secrets.
func NewFactory(
	kvLdr ifc.KvLoader, o *types.GeneratorOptions) *Factory {
	return &Factory{kvLdr: kvLdr, options: o}
}

const keyExistsErrorMsg = "cannot add key %s, another key by that name already exists: %v"
