/*
Copyright 2020 The Kubernetes Authors.

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

package rest

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/v3/fieldpath"
)

// CreationPreparator provides an interface to prepare new objects for being persisted
type CreationPreparator interface {
	// PrepareForCreate is invoked on create before validation to normalize
	// the object.  For example: remove fields that are not to be persisted,
	// sort order-insensitive list fields, etc.  This should not remove fields
	// whose presence would be considered a validation error.
	//
	// Often implemented as a type check and an initailization or clearing of
	// status. Clear the status because status changes are internal. External
	// callers of an api (users) should not be setting an initial status on
	// newly created objects.
	PrepareForCreate(ctx context.Context, obj runtime.Object)
	// ResetFieldsForCreate returns a set of fields for the provided version that get reset before persisting the object.
	// If no fieldset is defined for a version, nil is returned.
	ResetFieldsForCreate(version string) *fieldpath.Set
}

// ResetFields maps versions to the sets of fields that will be reset by the Preparator
type ResetFields map[string]*fieldpath.Set

// NewCreationPreparator using the PrepareForCreate function and exposing the defined resetFields.
// If a fieldBuilder is provided, it will be called once for every version at initialization to allow adding fields based on runtime information.
// The fieldBuilder can call Insert on the fields to add paths.
func NewCreationPreparator(fn prepareForCreate, resetFields ResetFields, fieldBuilder ...func(version string, fields *fieldpath.Set)) CreationPreparator {
	for version, fields := range resetFields {
		for _, builder := range fieldBuilder {
			builder(version, fields)
		}
	}

	return &creationPreparator{
		prepare:     fn,
		resetFields: resetFields,
	}
}

type creationPreparator struct {
	prepare     prepareForCreate
	resetFields ResetFields
}
type prepareForCreate func(ctx context.Context, obj runtime.Object)

var _ CreationPreparator = &creationPreparator{}

func (p *creationPreparator) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	p.prepare(ctx, obj)
}

// ResetFieldsForCreate returns a set of fields for the provided version that get reset before persisting the object.
// If no fieldset is defined for a version, nil is returned.
func (p *creationPreparator) ResetFieldsForCreate(version string) *fieldpath.Set {
	set, ok := p.resetFields[version]
	if !ok {
		return nil
	}
	return set
}
