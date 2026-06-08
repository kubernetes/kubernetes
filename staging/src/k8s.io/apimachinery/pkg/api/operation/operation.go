/*
Copyright 2024 The Kubernetes Authors.

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

package operation

import (
	"slices"
	"strings"
)

// Operation provides contextual information about a validation request and the API
// operation being validated.
// This type is intended for use with generate validation code and may be enhanced
// in the future to include other information needed to validate requests.
type Operation struct {
	// Type is the category of operation being validated.  This does not
	// differentiate between HTTP verbs like PUT and PATCH, but rather merges
	// those into a single "Update" category.
	Type Type

	// Options declare the options enabled for validation.
	//
	// Options should be set according to a resource validation strategy before validation
	// is performed, and must be treated as read-only during validation.
	//
	// Options are identified by string names. Option string names may match the name of a feature
	// gate, in which case the presence of the name in the set indicates that the feature is
	// considered enabled for the resource being validated.  Note that a resource may have a
	// feature enabled even when the feature gate is disabled. This can happen when feature is
	// already in-use by a resource, often because the feature gate was enabled when the
	// resource first began using the feature.
	//
	// Unset options are disabled/false.
	Options []string

	// Request provides information about the request being validated.
	Request Request
}

// HasOption returns true if the given string is in the Options slice.
func (o Operation) HasOption(option string) bool {
	return slices.Contains(o.Options, option)
}

// Request provides information about the request being validated.
type Request struct {
	// Subresources identifies the subresource path components of the request. For
	// example, Subresources for a request to `/api/v1/pods/my-pod/status` would be
	// `["status"]`. For `/api/v1/widget/my-widget/x/y/z`, it would be `["x", "y",
	// "z"]`. For a root resource (`/api/v1/pods/my-pod`), Subresources will be an
	// empty slice.
	//
	// Validation logic should only consult this field if the validation rules for a
	// particular field differ depending on whether the main resource or a specific
	// subresource is being accessed. For example:
	//
	// Updates to a Pod resource (`/`) normally cannot change container resource
	// requests/limits after the Pod is created (they are immutable). However, when
	// accessing the Pod's "resize" subresource (`/resize`), these specific fields
	// are allowed to be modified. In this scenario, the validation logic for
	// `spec.container[*].resources` must check `Subresources` to permit changes only
	// when the request targets the "resize" subresource.
	//
	// Note: This field should not be used to control which fields a subresource
	// operation is allowed to write. This is the responsibility of "field wiping".
	// Field wiping logic is expected to be handled in resource strategies by
	// modifying the incoming object before it is validated.
	Subresources []string
}

// SubresourcePath returns the path is a slash-separated list of subresource
// names. For example, `/status`, `/resize`, or `/x/y/z`.
func (r Request) SubresourcePath() string {
	if len(r.Subresources) == 0 {
		return "/"
	}
	return "/" + strings.Join(r.Subresources, "/")
}

// Code is the request operation to be validated.
type Type uint32

const (
	// Create indicates the request being validated is for a resource create operation.
	Create Type = iota

	// Update indicates the request being validated is for a resource update operation.
	Update
)
