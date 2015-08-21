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

package horizontalpodautoscaler

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	errs "k8s.io/kubernetes/pkg/util/fielderrors"
)

// autoscalerStrategy implements behavior for HorizontalPodAutoscalers
type autoscalerStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating HorizontalPodAutoscaler
// objects via the REST API.
var Strategy = autoscalerStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for autoscaler.
func (autoscalerStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (autoscalerStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*expapi.HorizontalPodAutoscaler)
}

// Validate validates a new autoscaler.
func (autoscalerStrategy) Validate(ctx api.Context, obj runtime.Object) errs.ValidationErrorList {
	autoscaler := obj.(*expapi.HorizontalPodAutoscaler)
	return validation.ValidateHorizontalPodAutoscaler(autoscaler)
}

// AllowCreateOnUpdate is false for autoscalers.
func (autoscalerStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (autoscalerStrategy) PrepareForUpdate(obj, old runtime.Object) {
	_ = obj.(*expapi.HorizontalPodAutoscaler)
}

// ValidateUpdate is the default update validation for an end user.
func (autoscalerStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) errs.ValidationErrorList {
	return validation.ValidateHorizontalPodAutoscalerUpdate(obj.(*expapi.HorizontalPodAutoscaler), old.(*expapi.HorizontalPodAutoscaler))
}

func (autoscalerStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// MatchAutoscaler returns a generic matcher for a given label and field selector.
func MatchAutoscaler(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		autoscaler, ok := obj.(*expapi.HorizontalPodAutoscaler)
		if !ok {
			return false, fmt.Errorf("not a horizontal pod autoscaler")
		}
		return label.Matches(labels.Set(autoscaler.Labels)), nil
	})
}
