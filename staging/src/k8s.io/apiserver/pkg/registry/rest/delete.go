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

package rest

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// RESTDeleteStrategy defines deletion behavior on an object that follows Kubernetes
// API conventions.
type RESTDeleteStrategy interface {
	runtime.ObjectTyper
}

type GarbageCollectionPolicy string

const (
	DeleteDependents GarbageCollectionPolicy = "DeleteDependents"
	OrphanDependents GarbageCollectionPolicy = "OrphanDependents"
)

// GarbageCollectionDeleteStrategy must be implemented by the registry that wants to
// orphan dependents by default.
type GarbageCollectionDeleteStrategy interface {
	// DefaultGarbageCollectionPolicy returns the default garbage collection behavior.
	DefaultGarbageCollectionPolicy() GarbageCollectionPolicy
}

// RESTGracefulDeleteStrategy must be implemented by the registry that supports
// graceful deletion.
type RESTGracefulDeleteStrategy interface {
	// CheckGracefulDelete should return true if the object can be gracefully deleted and set
	// any default values on the DeleteOptions.
	CheckGracefulDelete(ctx genericapirequest.Context, obj runtime.Object, options *metav1.DeleteOptions) bool
}

// BeforeDelete tests whether the object can be gracefully deleted. If graceful is set the object
// should be gracefully deleted, if gracefulPending is set the object has already been gracefully deleted
// (and the provided grace period is longer than the time to deletion), and an error is returned if the
// condition cannot be checked or the gracePeriodSeconds is invalid. The options argument may be updated with
// default values if graceful is true. Second place where we set deletionTimestamp is pkg/registry/generic/registry/store.go
// this function is responsible for setting deletionTimestamp during gracefulDeletion, other one for cascading deletions.
func BeforeDelete(strategy RESTDeleteStrategy, ctx genericapirequest.Context, obj runtime.Object, options *metav1.DeleteOptions) (graceful, gracefulPending bool, err error) {
	objectMeta, gvk, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return false, false, kerr
	}
	if errs := validation.ValidateDeleteOptions(options); len(errs) > 0 {
		return false, false, errors.NewInvalid(schema.GroupKind{}, "", errs)
	}
	// Checking the Preconditions here to fail early. They'll be enforced later on when we actually do the deletion, too.
	if options.Preconditions != nil && options.Preconditions.UID != nil && *options.Preconditions.UID != objectMeta.GetUID() {
		return false, false, errors.NewConflict(schema.GroupResource{Group: gvk.Group, Resource: gvk.Kind}, objectMeta.GetName(), fmt.Errorf("the UID in the precondition (%s) does not match the UID in record (%s). The object might have been deleted and then recreated", *options.Preconditions.UID, objectMeta.GetUID()))
	}
	gracefulStrategy, ok := strategy.(RESTGracefulDeleteStrategy)
	if !ok {
		// If we're not deleting gracefully there's no point in updating Generation, as we won't update
		// the obcject before deleting it.
		return false, false, nil
	}
	// if the object is already being deleted, no need to update generation.
	if objectMeta.GetDeletionTimestamp() != nil {
		// if we are already being deleted, we may only shorten the deletion grace period
		// this means the object was gracefully deleted previously but deletionGracePeriodSeconds was not set,
		// so we force deletion immediately
		// IMPORTANT:
		// The deletion operation happens in two phases.
		// 1. Update to set DeletionGracePeriodSeconds and DeletionTimestamp
		// 2. Delete the object from storage.
		// If the update succeeds, but the delete fails (network error, internal storage error, etc.),
		// a resource was previously left in a state that was non-recoverable.  We
		// check if the existing stored resource has a grace period as 0 and if so
		// attempt to delete immediately in order to recover from this scenario.
		if objectMeta.GetDeletionGracePeriodSeconds() == nil || *objectMeta.GetDeletionGracePeriodSeconds() == 0 {
			return false, false, nil
		}
		// only a shorter grace period may be provided by a user
		if options.GracePeriodSeconds != nil {
			period := int64(*options.GracePeriodSeconds)
			if period >= *objectMeta.GetDeletionGracePeriodSeconds() {
				return false, true, nil
			}
			newDeletionTimestamp := metav1.NewTime(
				objectMeta.GetDeletionTimestamp().Add(-time.Second * time.Duration(*objectMeta.GetDeletionGracePeriodSeconds())).
					Add(time.Second * time.Duration(*options.GracePeriodSeconds)))
			objectMeta.SetDeletionTimestamp(&newDeletionTimestamp)
			objectMeta.SetDeletionGracePeriodSeconds(&period)
			return true, false, nil
		}
		// graceful deletion is pending, do nothing
		options.GracePeriodSeconds = objectMeta.GetDeletionGracePeriodSeconds()
		return false, true, nil
	}

	if !gracefulStrategy.CheckGracefulDelete(ctx, obj, options) {
		return false, false, nil
	}
	now := metav1.NewTime(metav1.Now().Add(time.Second * time.Duration(*options.GracePeriodSeconds)))
	objectMeta.SetDeletionTimestamp(&now)
	objectMeta.SetDeletionGracePeriodSeconds(options.GracePeriodSeconds)
	// If it's the first graceful deletion we are going to set the DeletionTimestamp to non-nil.
	// Controllers of the object that's being deleted shouldn't take any nontrivial actions, hence its behavior changes.
	// Thus we need to bump object's Generation (if set). This handles generation bump during graceful deletion.
	// The bump for objects that don't support graceful deletion is handled in pkg/registry/generic/registry/store.go.
	if objectMeta.GetGeneration() > 0 {
		objectMeta.SetGeneration(objectMeta.GetGeneration() + 1)
	}
	return true, false, nil
}
