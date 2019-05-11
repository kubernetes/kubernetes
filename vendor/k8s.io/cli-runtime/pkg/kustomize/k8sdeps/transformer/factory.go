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

// Package transformer provides transformer factory
package transformer

import (
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/transformer/hash"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/transformer/patch"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers"
)

// FactoryImpl makes patch transformer and name hash transformer
type FactoryImpl struct{}

// NewFactoryImpl makes a new factoryImpl instance
func NewFactoryImpl() *FactoryImpl {
	return &FactoryImpl{}
}

// MakePatchTransformer makes a new patch transformer
func (p *FactoryImpl) MakePatchTransformer(slice []*resource.Resource, rf *resource.Factory) (transformers.Transformer, error) {
	return patch.NewPatchTransformer(slice, rf)
}

// MakeHashTransformer makes a new name hash transformer
func (p *FactoryImpl) MakeHashTransformer() transformers.Transformer {
	return hash.NewNameHashTransformer()
}
