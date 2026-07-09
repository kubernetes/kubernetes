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
	"fmt"
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

	// Options are the validation options in effect for this operation, mapping option
	// name to whether it is enabled. Option names typically match feature gates. Set by
	// the resource strategy and read-only during validation.
	//
	// Every option referenced by a validation tag must be declared here. Referencing an
	// undeclared option is a programming error (see HasOption).
	Options map[string]bool

	// Request provides information about the request being validated.
	Request Request
}

// HasOption returns whether the named option is enabled. Every option referenced by a
// validation tag must be declared by the strategy; referencing an undeclared option
// returns an error rather than silently treating it as disabled, so declarative
// validation can surface the mistake as an internal error.
func (o Operation) HasOption(option string) (bool, error) {
	enabled, declared := o.Options[option]
	if !declared {
		return false, fmt.Errorf("undeclared validation option %q", option)
	}
	return enabled, nil
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
