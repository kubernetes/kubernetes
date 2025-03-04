/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// ValidateDeclaratively validates obj against declarative validation tags
// defined in its Go type. It uses the API version extracted from ctx and the
// provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context or if version
// conversion fails.
func ValidateDeclaratively(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object) field.ErrorList {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		versionedObj, err := scheme.ConvertToVersion(obj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		subresources, err := parseSubresourcePath(requestInfo.Subresource)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", err))}
		}
		return scheme.Validate(ctx, options, versionedObj, subresources...)
	} else {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("could not find requestInfo in context"))}
	}
}

// ValidateUpdateDeclaratively validates obj and oldObj against declarative
// validation tags defined in its Go type. It uses the API version extracted from
// ctx and the provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context or if version
// conversion fails.
func ValidateUpdateDeclaratively(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object) field.ErrorList {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		versionedObj, err := scheme.ConvertToVersion(obj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		versionedOldObj, err := scheme.ConvertToVersion(oldObj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		subresources, err := parseSubresourcePath(requestInfo.Subresource)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", err))}
		}
		return scheme.ValidateUpdate(ctx, options, versionedObj, versionedOldObj, subresources...)
	} else {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("could not find requestInfo in context"))}
	}
}

func parseSubresourcePath(subresourcePath string) ([]string, error) {
	if len(subresourcePath) == 0 {
		return nil, nil
	}
	if subresourcePath[0] != '/' {
		return nil, fmt.Errorf("invalid subresource path: %s", subresourcePath)
	}
	parts := strings.Split(subresourcePath[1:], "/")
	return parts, nil
}
