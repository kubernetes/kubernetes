/*
Copyright 2014 The Kubernetes Authors.

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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/validation"
)

// networkStrategy implements verification logic for network.
type networkStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Network objects.
var Strategy = networkStrategy{api.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all Networks need to be within a namespace.
func (networkStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a nework before creation.
func (networkStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	return
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (networkStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newNetwork := obj.(*extensions.Network)
	oldNetwork := old.(*extensions.Network)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the manager has *seen*.
	//
	// TODO: Any changes to a part of the object that represents desired state (labels,
	// annotations etc) should also increment the generation.
	if !apiequality.Semantic.DeepEqual(oldNetwork.Spec, newNetwork.Spec) {
		newNetwork.Generation = oldNetwork.Generation + 1
	}
}

// Validate validates a new network.
func (networkStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	network := obj.(*extensions.Network)
	return validation.ValidateNetwork(network)
}

// Canonicalize normalizes the object after validation.
func (networkStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for network; this means a POST is
// needed to create one
func (networkStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (networkStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateNetwork(obj.(*extensions.Network))
	updateErrorList := validation.ValidateNetworkUpdate(obj.(*extensions.Network), old.(*extensions.Network))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for network objects.
func (networkStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// NetworkStrategyToSelectableFields returns a field set that represents the object.
func NetworkStrategyToSelectableFields(network *extensions.Network) fields.Set {
	return generic.ObjectMetaFieldsSet(&network.ObjectMeta, true)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	n, ok := obj.(*extensions.Network)
	if !ok {
		return nil, nil, false, fmt.Errorf("given object is not a Network.")
	}
	return labels.Set(n.ObjectMeta.Labels), NetworkStrategyToSelectableFields(n), n.Initializers != nil, nil
}

// MatchNetwork is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchNetwork(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}
