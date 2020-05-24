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
)

// UpdatePreparator provides an interface to prepare updates for being persisted
type UpdatePreparator interface {
	// PrepareForUpdate is invoked on update before validation to normalize
	// the object.  For example: remove fields that are not to be persisted,
	// sort order-insensitive list fields, etc.  This should not remove fields
	// whose presence would be considered a validation error.
	PrepareForUpdate(ctx context.Context, obj, old runtime.Object)
	// ResetFieldsForUpdate returns the set of fields per version that get reset before persisting the object.
	ResetFieldsForUpdate() ResetFields
}

// NewUpdatePreparator using the PrepareForUpdate function and exposing the defined resetFields.
// Providing a fieldBuilder allows modifying the ResetFields with information available at runtime.
// The fieldBuilder can call Insert on the fields to add paths.
func NewUpdatePreparator(fn prepareForUpdate, resetFields ResetFields, fieldBuilder ...func(ResetFields)) UpdatePreparator {
	for _, builder := range fieldBuilder {
		builder(resetFields)
	}

	return &updatePreparator{
		prepare:     fn,
		resetFields: resetFields,
	}
}

type updatePreparator struct {
	prepare     prepareForUpdate
	resetFields ResetFields
}
type prepareForUpdate func(ctx context.Context, obj, old runtime.Object)

var _ UpdatePreparator = &updatePreparator{}

func (p *updatePreparator) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	p.prepare(ctx, obj, old)
}

// ResetFieldsForUpdate returns the set of fields per version that get reset before persisting the object.
func (p *updatePreparator) ResetFieldsForUpdate() ResetFields {
	return p.resetFields
}
