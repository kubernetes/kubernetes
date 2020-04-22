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
}

// UpdatePreparator provides an interface to prepare updates for being persisted
type UpdatePreparator interface {
	// PrepareForUpdate is invoked on update before validation to normalize
	// the object.  For example: remove fields that are not to be persisted,
	// sort order-insensitive list fields, etc.  This should not remove fields
	// whose presence would be considered a validation error.
	PrepareForUpdate(ctx context.Context, obj, old runtime.Object)
}

// GenericPreparator wraps available create and update preparators
type GenericPreparator struct {
	CreationPreparator
	UpdatePreparator
}

// NewGenericPreparator wrapping the Creation and UpdatePreparator
func NewGenericPreparator(c CreationPreparator, u UpdatePreparator) *GenericPreparator {
	return &GenericPreparator{c, u}
}

// Prepare the object through the appropriate preparator.
// In cases where both the old and new object are provided, the operation will be considered
// an update and the respective preparator will be used.
// In cases where the old object is nil, the CreationPreparator will be used.
func (p *GenericPreparator) Prepare(ctx context.Context, obj, old runtime.Object) {
	if old != nil && p.UpdatePreparator != nil {
		p.PrepareForUpdate(ctx, obj, old)
	} else if obj != nil && p.CreationPreparator != nil {
		p.PrepareForCreate(ctx, obj)
	}
}
