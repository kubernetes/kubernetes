/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/discovery"
	"k8s.io/kubernetes/pkg/apis/discovery/validation"
	"k8s.io/kubernetes/pkg/features"
)

// endpointSliceStrategy implements verification logic for Replication.
type endpointSliceStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication EndpointSlice objects.
var Strategy = endpointSliceStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all EndpointSlices need to be within a namespace.
func (endpointSliceStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of an EndpointSlice before creation.
func (endpointSliceStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	endpointSlice := obj.(*discovery.EndpointSlice)
	endpointSlice.Generation = 1

	dropDisabledConditionsOnCreate(endpointSlice)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (endpointSliceStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newEPS := obj.(*discovery.EndpointSlice)
	oldEPS := old.(*discovery.EndpointSlice)

	// Increment generation if anything other than meta changed
	// This needs to be changed if a status attribute is added to EndpointSlice
	ogNewMeta := newEPS.ObjectMeta
	ogOldMeta := oldEPS.ObjectMeta
	newEPS.ObjectMeta = v1.ObjectMeta{}
	oldEPS.ObjectMeta = v1.ObjectMeta{}

	if !apiequality.Semantic.DeepEqual(newEPS, oldEPS) {
		ogNewMeta.Generation = ogOldMeta.Generation + 1
	}

	newEPS.ObjectMeta = ogNewMeta
	oldEPS.ObjectMeta = ogOldMeta

	dropDisabledConditionsOnUpdate(oldEPS, newEPS)
}

// Validate validates a new EndpointSlice.
func (endpointSliceStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	endpointSlice := obj.(*discovery.EndpointSlice)
	err := validation.ValidateEndpointSliceCreate(endpointSlice)
	return err
}

// Canonicalize normalizes the object after validation.
func (endpointSliceStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for EndpointSlice; this means POST is needed to create one.
func (endpointSliceStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (endpointSliceStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newEPS := new.(*discovery.EndpointSlice)
	oldEPS := old.(*discovery.EndpointSlice)
	return validation.ValidateEndpointSliceUpdate(newEPS, oldEPS)
}

// AllowUnconditionalUpdate is the default update policy for EndpointSlice objects.
func (endpointSliceStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// dropDisabledConditionsOnCreate will drop the terminating condition if the
// EndpointSliceTerminatingCondition is disabled. Otherwise the field is left untouched.
func dropDisabledConditionsOnCreate(endpointSlice *discovery.EndpointSlice) {
	if utilfeature.DefaultFeatureGate.Enabled(features.EndpointSliceTerminatingCondition) {
		return
	}

	// Always drop the serving/terminating conditions on create when feature gate is disabled.
	for i := range endpointSlice.Endpoints {
		endpointSlice.Endpoints[i].Conditions.Serving = nil
		endpointSlice.Endpoints[i].Conditions.Terminating = nil
	}
}

// dropDisabledConditionsOnUpdate will drop the terminating condition field if the EndpointSliceTerminatingCondition
// feature gate is disabled unless an existing EndpointSlice object has the field already set. This ensures
// the field is not dropped on rollback.
func dropDisabledConditionsOnUpdate(oldEPS, newEPS *discovery.EndpointSlice) {
	if utilfeature.DefaultFeatureGate.Enabled(features.EndpointSliceTerminatingCondition) {
		return
	}

	// Only drop the serving/terminating condition if the existing EndpointSlice doesn't have it set.
	dropConditions := true
	for _, ep := range oldEPS.Endpoints {
		if ep.Conditions.Serving != nil || ep.Conditions.Terminating != nil {
			dropConditions = false
			break
		}
	}

	if dropConditions {
		for i := range newEPS.Endpoints {
			newEPS.Endpoints[i].Conditions.Serving = nil
			newEPS.Endpoints[i].Conditions.Terminating = nil
		}
	}
}
