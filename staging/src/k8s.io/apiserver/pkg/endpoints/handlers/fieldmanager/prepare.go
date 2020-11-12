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
package fieldmanager

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// Preparator defines the interface required to prepare objects for storage.
// It is used to determine if the object will be modified before storage.
// Implementations should determine if the preparation is for an update or create operation
type Preparator interface {
	Prepare(ctx context.Context, newObj, liveObj runtime.Object)
}

// UpdatePreparator provides PrepareForUpdate
type UpdatePreparator interface {
	PrepareForUpdate(ctx context.Context, obj, old runtime.Object)
}

// CreationPreparator provides PrepareForCreate
type CreationPreparator interface {
	PrepareForCreate(ctx context.Context, obj runtime.Object)
}

type preparator struct {
	createStrategy, updateStrategy interface{}
}

// NewPreparator using strategy
// Preparation will use the strategies PrepareForCreate or PrepareForUpdate methods
// depending on the requestInfo and the strategy
func NewPreparator(createStrategy, updateStrategy interface{}) Preparator {
	return &preparator{
		createStrategy: createStrategy,
		updateStrategy: updateStrategy,
	}
}

const createVerb = "create"

func (p preparator) Prepare(ctx context.Context, newObj, liveObj runtime.Object) {
	isCreate := false
	// TODO(kwiesmueller): check if this even works. Looks like the requestInfo might not be in ctx yet
	if requestInfo, hasRequestInfo := request.RequestInfoFrom(ctx); hasRequestInfo {
		isCreate = requestInfo.Verb == createVerb
	}
	if creationPreparator, isCreationPreparator := p.createStrategy.(CreationPreparator); isCreationPreparator && isCreate {
		creationPreparator.PrepareForCreate(ctx, newObj)
	}
	// TODO(kwiesmueller): do we have to check for the verb as well?
	if updatePreparator, isUpdatePreparator := p.updateStrategy.(UpdatePreparator); isUpdatePreparator {
		updatePreparator.PrepareForUpdate(ctx, newObj, liveObj)
	}
}
