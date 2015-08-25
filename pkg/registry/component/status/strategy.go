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

package status

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/registry/component"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// updateStrategy implements create and update validation for Components
type updateStrategy struct {
	runtime.ObjectTyper
}

// UpdateStrategy is the default update logic for Component status objects.
var UpdateStrategy = updateStrategy{api.Scheme}

// NamespaceScoped returns false because components are global.
func (updateStrategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate returns false because creation handles naming.
// If the component has a name (required for update) but the storage doesn't know about it, something is seriously wrong.
func (updateStrategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate returns true because the user is not required to supply a resource version when performing an update.
func (updateStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// PrepareForUpdate transfers implied component fields and populates timestamps.
func (updateStrategy) PrepareForUpdate(newO, oldO runtime.Object) {
	component.CreateUpdateStrategy.PrepareForUpdate(newO, oldO)

	component := newO.(*api.Component)
	old := oldO.(*api.Component)

	// transfer existing spec
	component.Spec = old.Spec
}

// Validate validates an update to an existing component status.
func (updateStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateComponentStatusUpdate(old.(*api.Component), obj.(*api.Component))
}
