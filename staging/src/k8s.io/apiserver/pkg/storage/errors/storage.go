/*
Copyright 2014 The Kubernetes Authors.

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

package storage

import (
	"errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
)

// InterpretListError converts a generic error on a retrieval
// operation into the appropriate API error.
func InterpretListError(err error, qualifiedResource schema.GroupResource) error {
	var storageError *storage.StorageError
	if !errors.As(err, &storageError) {
		return err
	}

	switch {
	case storage.IsNotFound(storageError):
		return apierrors.NewNotFound(qualifiedResource, "")
	case storage.IsCorruptedData(storageError):
		return apierrors.NewStorageReadError(qualifiedResource, "list", storageError.KeyErrors)
	case storage.IsUnreachable(storageError), storage.IsRequestTimeout(storageError):
		return apierrors.NewServerTimeout(qualifiedResource, "list", 2) // TODO: make configurable or handled at a higher level
	case storage.IsInternalError(storageError):
		return apierrors.NewInternalError(storageError)
	default:
		return storageError
	}
}

// InterpretGetError converts a generic error on a retrieval
// operation into the appropriate API error.
func InterpretGetError(err error, qualifiedResource schema.GroupResource, name string) error {
	var storageError *storage.StorageError
	if !errors.As(err, &storageError) {
		return err
	}

	switch {
	case storage.IsNotFound(storageError):
		return apierrors.NewNotFound(qualifiedResource, name)
	case storage.IsCorruptedData(storageError):
		return apierrors.NewStorageReadError(qualifiedResource, name, storageError.KeyErrors)
	case storage.IsUnreachable(storageError):
		return apierrors.NewServerTimeout(qualifiedResource, "get", 2) // TODO: make configurable or handled at a higher level
	case storage.IsInternalError(storageError):
		return apierrors.NewInternalError(storageError)
	default:
		return storageError
	}
}

// InterpretCreateError converts a generic error on a create
// operation into the appropriate API error.
func InterpretCreateError(err error, qualifiedResource schema.GroupResource, name string) error {
	switch {
	case storage.IsExist(err):
		return apierrors.NewAlreadyExists(qualifiedResource, name)
	case storage.IsUnreachable(err):
		return apierrors.NewServerTimeout(qualifiedResource, "create", 2) // TODO: make configurable or handled at a higher level
	case storage.IsInternalError(err):
		return apierrors.NewInternalError(err)
	default:
		return err
	}
}

// InterpretUpdateError converts a generic error on an update
// operation into the appropriate API error.
func InterpretUpdateError(err error, qualifiedResource schema.GroupResource, name string) error {
	switch {
	case storage.IsConflict(err), storage.IsExist(err), storage.IsInvalidObj(err):
		return apierrors.NewConflict(qualifiedResource, name, err)
	case storage.IsUnreachable(err):
		return apierrors.NewServerTimeout(qualifiedResource, "update", 2) // TODO: make configurable or handled at a higher level
	case storage.IsNotFound(err):
		return apierrors.NewNotFound(qualifiedResource, name)
	case storage.IsInternalError(err):
		return apierrors.NewInternalError(err)
	default:
		return err
	}
}

// InterpretDeleteError converts a generic error on a delete
// operation into the appropriate API error.
func InterpretDeleteError(err error, qualifiedResource schema.GroupResource, name string) error {
	switch {
	case storage.IsNotFound(err):
		return apierrors.NewNotFound(qualifiedResource, name)
	case storage.IsUnreachable(err):
		return apierrors.NewServerTimeout(qualifiedResource, "delete", 2) // TODO: make configurable or handled at a higher level
	case storage.IsConflict(err), storage.IsExist(err), storage.IsInvalidObj(err):
		return apierrors.NewConflict(qualifiedResource, name, err)
	case storage.IsInternalError(err):
		return apierrors.NewInternalError(err)
	default:
		return err
	}
}

// InterpretWatchError converts a generic error on a watch
// operation into the appropriate API error.
func InterpretWatchError(err error, resource schema.GroupResource, name string) error {
	switch {
	case storage.IsInvalidError(err):
		invalidError, _ := err.(storage.InvalidError)
		return apierrors.NewInvalid(schema.GroupKind{Group: resource.Group, Kind: resource.Resource}, name, invalidError.Errs)
	case storage.IsInternalError(err):
		return apierrors.NewInternalError(err)
	default:
		return err
	}
}
