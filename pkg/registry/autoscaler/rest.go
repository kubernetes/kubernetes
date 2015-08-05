/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package autoscaler

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

// autoScalerStrategy implements behavior for AutoScalers
type autoScalerStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// AutoScalers is the default logic that applies when creating and updating AutoScalers
// objects.
var AutoScalers = autoScalerStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for AutoScalers.
func (autoScalerStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (autoScalerStrategy) PrepareForCreate(obj runtime.Object) {
	autoScaler := obj.(*api.AutoScaler)
	autoScaler.Status = api.AutoScalerStatus{}
}

// PrepareForUpdate sets the Status field which is not allowed to be set by end users on update.
func (autoScalerStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newAs := obj.(*api.AutoScaler)
	oldAs := old.(*api.AutoScaler)
	newAs.Status = oldAs.Status
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (autoScalerStrategy) ResetBeforeCreate(obj runtime.Object) {
	autoScaler := obj.(*api.AutoScaler)
	autoScaler.Status = api.AutoScalerStatus{}
}

// Validate validates a new AutoScalers.
func (autoScalerStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	autoScaler := obj.(*api.AutoScaler)
	return validation.ValidateAutoScaler(autoScaler)
}

// AllowCreateOnUpdate dictates if you can create a new autoscaler with a PUT
func (autoScalerStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate validates AutoScalers during an update
func (autoScalerStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateAutoScalerUpdate(obj.(*api.AutoScaler), old.(*api.AutoScaler))
}

func (autoScalerStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type autoScalerStatusStrategy struct {
	autoScalerStrategy
}

var AutoScalersStatusStrategy = autoScalerStatusStrategy{AutoScalers}

// NamespaceScoped is true for AutoScalersStatusStrategy.
func (autoScalerStatusStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForUpdate sets the Spec field which is not allowed to be set by end users on update.
func (autoScalerStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newAs := obj.(*api.AutoScaler)
	oldAs := old.(*api.AutoScaler)
	newAs.Spec = oldAs.Spec
}

// Validate validates a new AutoScalerStatusStrategy.
func (autoScalerStatusStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	autoScaler := obj.(*api.AutoScaler)
	return validation.ValidateAutoScalerStatus(autoScaler)
}

// ValidateUpdate validates AutoScalersStatusStrategy during an update
func (autoScalerStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateAutoScalerStatusUpdate(obj.(*api.AutoScaler), old.(*api.AutoScaler))
}

// MatchAutoScaler returns a generic matcher for a given label and field selector.
func MatchAutoScaler(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		autoScaler, ok := obj.(*api.AutoScaler)
		if !ok {
			return false, fmt.Errorf("not an autoscaler")
		}

		fields := AutoScalerToSelectableFields(autoScaler)
		return label.Matches(labels.Set(autoScaler.ObjectMeta.Labels)) && field.Matches(fields), nil
	})
}

// AutoScalerToSelectableFields returns a label set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func AutoScalerToSelectableFields(autoScaler *api.AutoScaler) labels.Set {
	return labels.Set{
		"name": autoScaler.Name,
	}
}
