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

package endpoint

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	endptspkg "k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
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
	endpoints := obj.(*api.Endpoints)
	endpoints.Subsets = endptspkg.RepackSubsets(endpoints.Subsets)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (endpointsStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newEndpoints := obj.(*api.Endpoints)
	_ = old.(*api.Endpoints)
	newEndpoints.Subsets = endptspkg.RepackSubsets(newEndpoints.Subsets)
}

// Validate validates a new endpoints.
func (endpointsStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateEndpoints(obj.(*api.Endpoints))
}

// AllowCreateOnUpdate is true for endpoints.
func (endpointsStrategy) AllowCreateOnUpdate() bool {
	return true
}

// ValidateUpdate is the default update validation for an end user.
func (endpointsStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	errorList := validation.ValidateEndpoints(obj.(*api.Endpoints))
	return append(errorList, validation.ValidateEndpointsUpdate(old.(*api.Endpoints), obj.(*api.Endpoints))...)
}

func (endpointsStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// MatchEndpoints returns a generic matcher for a given label and field selector.
func MatchEndpoints(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{label, field, EndpointsAttributes}
}

// EndpointsAttributes returns the attributes of an endpoint such that a
// generic.SelectionPredicate can match appropriately.
func EndpointsAttributes(obj runtime.Object) (objLabels labels.Set, objFields fields.Set, err error) {
	endpoints, ok := obj.(*api.Endpoints)
	if !ok {
		return nil, nil, fmt.Errorf("invalid object type %#v", obj)
	}
	return endpoints.Labels, fields.Set{
		"metadata.name": endpoints.Name,
	}, nil
}
