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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// RESTDeleteStrategy defines deletion behavior on an object that follows Kubernetes
// API conventions.
type RESTDeleteStrategy interface {
	runtime.ObjectTyper

	// CheckGracefulDelete should return true if the object can be gracefully deleted and set
	// any default values on the DeleteOptions.
	CheckGracefulDelete(obj runtime.Object, options *api.DeleteOptions) bool
}

// BeforeDelete tests whether the object can be gracefully deleted. If graceful is set the object
// should be gracefully deleted, if gracefulPending is set the object has already been gracefully deleted
// (and the provided grace period is longer than the time to deletion), and an error is returned if the
// condition cannot be checked or the gracePeriodSeconds is invalid. The options argument may be updated with
// default values if graceful is true.
func BeforeDelete(strategy RESTDeleteStrategy, ctx api.Context, obj runtime.Object, options *api.DeleteOptions) (graceful, gracefulPending bool, err error) {
	if strategy == nil {
		return false, false, nil
	}
	objectMeta, _, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return false, false, kerr
	}

	// if the object is already being deleted
	if objectMeta.DeletionTimestamp != nil {
		// if we are already being deleted, we may only shorten the deletion grace period
		// this means the object was gracefully deleted previously but deletionGracePeriodSeconds was not set,
		// so we force deletion immediately
		if objectMeta.DeletionGracePeriodSeconds == nil {
			return false, false, nil
		}
		// only a shorter grace period may be provided by a user
		if options.GracePeriodSeconds != nil {
			period := int64(*options.GracePeriodSeconds)
			if period > *objectMeta.DeletionGracePeriodSeconds {
				return false, true, nil
			}
			now := unversioned.NewTime(unversioned.Now().Add(time.Second * time.Duration(*options.GracePeriodSeconds)))
			objectMeta.DeletionTimestamp = &now
			objectMeta.DeletionGracePeriodSeconds = &period
			options.GracePeriodSeconds = &period
			return true, false, nil
		}
		// graceful deletion is pending, do nothing
		options.GracePeriodSeconds = objectMeta.DeletionGracePeriodSeconds
		return false, true, nil
	}

	if !strategy.CheckGracefulDelete(obj, options) {
		return false, false, nil
	}
	now := unversioned.NewTime(unversioned.Now().Add(time.Second * time.Duration(*options.GracePeriodSeconds)))
	objectMeta.DeletionTimestamp = &now
	objectMeta.DeletionGracePeriodSeconds = options.GracePeriodSeconds
	return true, false, nil
}
