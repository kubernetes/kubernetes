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
	_, _, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return false, false, kerr
	}
	if !strategy.CheckGracefulDelete(obj, options) {
		return false, false, nil
	}
	return true, false, nil
}
