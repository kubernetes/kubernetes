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

package nodemetrics

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/apis/experimental/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	errs "k8s.io/kubernetes/pkg/util/fielderrors"
)

type nodemetricsStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// DerivedNodeMetrics objects via the REST API.
var Strategy = nodemetricsStrategy{api.Scheme, api.SimpleNameGenerator}

func (nodemetricsStrategy) NamespaceScoped() bool {
	return false
}

func (nodemetricsStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*experimental.DerivedNodeMetrics)
}

func (nodemetricsStrategy) Validate(ctx api.Context, obj runtime.Object) errs.ValidationErrorList {
	metrics := obj.(*experimental.DerivedNodeMetrics)
	return validation.ValidateDerivedNodeMetrics(metrics)
}

func (nodemetricsStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (nodemetricsStrategy) PrepareForUpdate(obj, old runtime.Object) {
	_ = obj.(*experimental.DerivedNodeMetrics)
}

func (nodemetricsStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) errs.ValidationErrorList {
	objMetrics := obj.(*experimental.DerivedNodeMetrics)
	oldMetrics := old.(*experimental.DerivedNodeMetrics)
	return validation.ValidateDerivedNodeMetricsUpdate(objMetrics, oldMetrics)
}

func (nodemetricsStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func MatchDerivedNodeMetrics(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		metrics, ok := obj.(*experimental.DerivedNodeMetrics)
		if !ok {
			return false, fmt.Errorf("not derived node metrics")
		}
		return label.Matches(labels.Set(metrics.Labels)), nil
	})
}
