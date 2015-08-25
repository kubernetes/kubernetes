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

package network

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// networkStrategy implements behavior for Networks
type networkStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Network
// objects via the REST API.
var Strategy = networkStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for networks.
func (networkStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears subnets if
func (networkStrategy) PrepareForCreate(obj runtime.Object) {
	// on create, status is Initializing
	network := obj.(*api.Network)
	network.Status = api.NetworkStatus{
		Phase: api.NetworkInitializing,
	}

	if network.Spec.ProviderNetworkID != "" {
		network.Spec.Subnets = nil
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (networkStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newNetwork := obj.(*api.Network)
	oldNetwork := old.(*api.Network)
	newNetwork.Status = oldNetwork.Status
}

// Validate validates a new network.
func (networkStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	network := obj.(*api.Network)
	return validation.ValidateNetwork(network)
}

// AllowCreateOnUpdate is false for networks.
func (networkStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (networkStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	errorList := validation.ValidateNetwork(obj.(*api.Network))
	return append(errorList, validation.ValidateNetworkUpdate(obj.(*api.Network), old.(*api.Network))...)
}

func (networkStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type networkStatusStrategy struct {
	networkStrategy
}

var StatusStrategy = networkStatusStrategy{Strategy}

func (networkStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newNetwork := obj.(*api.Network)
	oldNetwork := old.(*api.Network)
	newNetwork.Spec = oldNetwork.Spec
}

func (networkStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateNetworkStatusUpdate(obj.(*api.Network), old.(*api.Network))
}

// MatchNetwork returns a generic matcher for a given label and field selector.
func MatchNetwork(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		networkObj, ok := obj.(*api.Network)
		if !ok {
			return false, fmt.Errorf("not a network")
		}
		fields := NetworkToSelectableFields(networkObj)
		return label.Matches(labels.Set(networkObj.Labels)) && field.Matches(fields), nil
	})
}

// NetworkToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func NetworkToSelectableFields(network *api.Network) labels.Set {
	return labels.Set{
		"name":         network.Name,
		"status.phase": string(network.Status.Phase),
	}
}
