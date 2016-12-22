/*
Copyright 2015 The Kubernetes Authors.

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

package admission

import (
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/runtime/schema"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

func extractResourceName(a Attributes) (name string, resource schema.GroupResource, err error) {
	name = "Unknown"
	resource = a.GetResource().GroupResource()
	obj := a.GetObject()
	if obj != nil {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return "", schema.GroupResource{}, err
		}

		// this is necessary because name object name generation has not occurred yet
		if len(accessor.GetName()) > 0 {
			name = accessor.GetName()
		} else if len(accessor.GetGenerateName()) > 0 {
			name = accessor.GetGenerateName()
		}
	}
	return name, resource, nil
}

// NewForbidden is a utility function to return a well-formatted admission control error response
func NewForbidden(a Attributes, internalError error) error {
	// do not double wrap an error of same type
	if apierrors.IsForbidden(internalError) {
		return internalError
	}
	name, resource, err := extractResourceName(a)
	if err != nil {
		return apierrors.NewInternalError(utilerrors.NewAggregate([]error{internalError, err}))
	}
	return apierrors.NewForbidden(resource, name, internalError)
}

// NewNotFound is a utility function to return a well-formatted admission control error response
func NewNotFound(a Attributes) error {
	name, resource, err := extractResourceName(a)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	return apierrors.NewNotFound(resource, name)
}
