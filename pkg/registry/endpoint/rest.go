/*
Copyright 2014 Google Inc. All rights reserved.

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

package endpoint

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
)

// endpointsStrategy implements behavior for Endpoints
type endpointsStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Endpoint
// objects via the REST API.
var Strategy = endpointsStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for endpoints.
func (endpointsStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (endpointsStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*api.Endpoints)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (endpointsStrategy) PrepareForUpdate(obj, old runtime.Object) {
	_ = obj.(*api.Endpoints)
	_ = old.(*api.Endpoints)
}

// Validate validates a new endpoints.
func (endpointsStrategy) Validate(obj runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateEndpoints(obj.(*api.Endpoints))
}

// AllowCreateOnUpdate is true for endpoints.
func (endpointsStrategy) AllowCreateOnUpdate() bool {
	return true
}

// ValidateUpdate is the default update validation for an end user.
func (endpointsStrategy) ValidateUpdate(obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateEndpointsUpdate(old.(*api.Endpoints), obj.(*api.Endpoints))
}

// MatchEndpoints returns a generic matcher for a given label and field selector.
func MatchEndpoints(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		endpoints, ok := obj.(*api.Endpoints)
		if !ok {
			return false, fmt.Errorf("not a endpoints")
		}
		fields := EndpointsToSelectableFields(endpoints)
		return label.Matches(labels.Set(endpoints.Labels)) && field.Matches(fields), nil
	})
}

// EndpointsToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func EndpointsToSelectableFields(endpoints *api.Endpoints) labels.Set {
	return labels.Set{
		"name": endpoints.Name,
	}
}
