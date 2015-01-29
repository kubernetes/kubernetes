/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// RESTCreateStrategy defines the minimum validation, accepted input, and
// name generation behavior to create an object that follows Kubernetes
// API conventions.
type RESTCreateStrategy interface {
	runtime.ObjectTyper
	// The name generate is used when the standard GenerateName field is set.
	// The NameGenerator will be invoked prior to validation.
	api.NameGenerator

	// NamespaceScoped returns true if the object must be within a namespace.
	NamespaceScoped() bool
	// ResetBeforeCreate is invoked on create before validation to remove any fields
	// that may not be persisted.
	ResetBeforeCreate(obj runtime.Object)
	// Validate is invoked after default fields in the object have been filled in before
	// the object is persisted.
	Validate(obj runtime.Object) errors.ValidationErrorList
}

// BeforeCreate ensures that common operations for all resources are performed on creation. It only returns
// errors that can be converted to api.Status. It invokes ResetBeforeCreate, then GenerateName, then Validate.
// It returns nil if the object should be created.
func BeforeCreate(strategy RESTCreateStrategy, ctx api.Context, obj runtime.Object) error {
	_, kind, err := strategy.ObjectVersionAndKind(obj)
	if err != nil {
		return errors.NewInternalError(err)
	}
	objectMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return errors.NewInternalError(err)
	}

	if strategy.NamespaceScoped() {
		if !api.ValidNamespace(ctx, objectMeta) {
			return errors.NewBadRequest("the namespace of the provided object does not match the namespace sent on the request")
		}
	} else {
		objectMeta.Namespace = api.NamespaceNone
	}
	strategy.ResetBeforeCreate(obj)
	api.FillObjectMetaSystemFields(ctx, objectMeta)
	api.GenerateName(strategy, objectMeta)

	if errs := strategy.Validate(obj); len(errs) > 0 {
		return errors.NewInvalid(kind, objectMeta.Name, errs)
	}
	return nil
}
