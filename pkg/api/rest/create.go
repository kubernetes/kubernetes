/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
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
	// PrepareForCreate is invoked on create before validation to normalize
	// the object.  For example: remove fields that are not to be persisted,
	// sort order-insensitive list fields, etc.  This should not remove fields
	// whose presence would be considered a validation error.
	PrepareForCreate(obj runtime.Object)
	// Validate is invoked after default fields in the object have been filled in before
	// the object is persisted.  This method should not mutate the object.
	Validate(ctx api.Context, obj runtime.Object) field.ErrorList
	// Canonicalize is invoked after validation has succeeded but before the
	// object has been persisted.  This method may mutate the object.
	Canonicalize(obj runtime.Object)
}

// BeforeCreate ensures that common operations for all resources are performed on creation. It only returns
// errors that can be converted to api.Status. It invokes PrepareForCreate, then GenerateName, then Validate.
// It returns nil if the object should be created.
func BeforeCreate(strategy RESTCreateStrategy, ctx api.Context, obj runtime.Object) error {
	objectMeta, kind, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return kerr
	}

	if strategy.NamespaceScoped() {
		if !api.ValidNamespace(ctx, objectMeta) {
			return errors.NewBadRequest("the namespace of the provided object does not match the namespace sent on the request")
		}
	} else {
		objectMeta.Namespace = api.NamespaceNone
	}
	objectMeta.DeletionTimestamp = nil
	objectMeta.DeletionGracePeriodSeconds = nil
	strategy.PrepareForCreate(obj)
	api.FillObjectMetaSystemFields(ctx, objectMeta)
	api.GenerateName(strategy, objectMeta)

	if errs := strategy.Validate(ctx, obj); len(errs) > 0 {
		return errors.NewInvalid(kind.GroupKind(), objectMeta.Name, errs)
	}

	// Custom validation (including name validation) passed
	// Now run common validation on object meta
	// Do this *after* custom validation so that specific error messages are shown whenever possible
	if errs := validation.ValidateObjectMeta(objectMeta, strategy.NamespaceScoped(), validation.ValidatePathSegmentName, field.NewPath("metadata")); len(errs) > 0 {
		return errors.NewInvalid(kind.GroupKind(), objectMeta.Name, errs)
	}

	strategy.Canonicalize(obj)

	return nil
}

// CheckGeneratedNameError checks whether an error that occurred creating a resource is due
// to generation being unable to pick a valid name.
func CheckGeneratedNameError(strategy RESTCreateStrategy, err error, obj runtime.Object) error {
	if !errors.IsAlreadyExists(err) {
		return err
	}

	objectMeta, kind, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return kerr
	}

	if len(objectMeta.GenerateName) == 0 {
		return err
	}

	return errors.NewServerTimeoutForKind(kind.GroupKind(), "POST", 0)
}

// objectMetaAndKind retrieves kind and ObjectMeta from a runtime object, or returns an error.
func objectMetaAndKind(typer runtime.ObjectTyper, obj runtime.Object) (*api.ObjectMeta, unversioned.GroupVersionKind, error) {
	objectMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return nil, unversioned.GroupVersionKind{}, errors.NewInternalError(err)
	}
	kind, err := typer.ObjectKind(obj)
	if err != nil {
		return nil, unversioned.GroupVersionKind{}, errors.NewInternalError(err)
	}
	return objectMeta, kind, nil
}

// NamespaceScopedStrategy has a method to tell if the object must be in a namespace.
type NamespaceScopedStrategy interface {
	// NamespaceScoped returns if the object must be in a namespace.
	NamespaceScoped() bool
}
