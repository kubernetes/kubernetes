// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package provider

import (
	"sigs.k8s.io/kustomize/api/hasher"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/validate"
	"sigs.k8s.io/kustomize/api/resource"
)

// DepProvider is a dependency provider, injecting different
// implementations depending on the context.
type DepProvider struct {
	resourceFactory *resource.Factory
	// implemented by api/internal/validate.FieldValidator
	// See TODO inside the validator for status.
	// At time of writing, this is a do-nothing
	// validator as it's not critical to kustomize function.
	fieldValidator ifc.Validator
}

func NewDepProvider() *DepProvider {
	rf := resource.NewFactory(&hasher.Hasher{})
	return &DepProvider{
		resourceFactory: rf,
		fieldValidator:  validate.NewFieldValidator(),
	}
}

func NewDefaultDepProvider() *DepProvider {
	return NewDepProvider()
}

func (dp *DepProvider) GetResourceFactory() *resource.Factory {
	return dp.resourceFactory
}

func (dp *DepProvider) GetFieldValidator() ifc.Validator {
	return dp.fieldValidator
}
