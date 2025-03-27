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

import "k8s.io/apimachinery/pkg/util/sets"

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
	Options sets.Set[string]

	// Request provides information about the request being validated.
	Request Request
}

// Request provides information about the request being validated.
type Request struct {
	// Subresources is the path to the subresource being validated. Validations
	// should use this only when the validation of field differs for a particular
	// subresource. For example, the resize subresource of pod is allowed to change
	// the value of spec.container[*].resources fields, which are immutable updated
	// via the root pod resource. In this example, because validation of the fields
	// are conditional to the subresource, the validator should use Subresources.
	// Note field wiping, which limits which fields a subresource is allowed to write
	// to, is handled in strategies, and does should to be implemented using
	// Subresources since field wiping drops field changes from the request, but does
	// not require changes to validation logic.
	Subresources []string
}

// Code is the request operation to be validated.
type Type uint32

const (
	// Create indicates the request being validated is for a resource create operation.
	Create Type = iota

	// Update indicates the request being validated is for a resource update operation.
	Update
)
