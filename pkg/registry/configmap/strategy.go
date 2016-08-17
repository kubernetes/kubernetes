/*
Copyright 2015 The Kubernetes Authors.

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

package configmap

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// strategy implements behavior for ConfigMap objects
type strategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ConfigMap
// objects via the REST API.
var Strategy = strategy{api.Scheme, api.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

func (strategy) NamespaceScoped() bool {
	return true
}

func (strategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	_ = obj.(*api.ConfigMap)
}

func (strategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	cfg := obj.(*api.ConfigMap)

	return validation.ValidateConfigMap(cfg)
}

// Canonicalize normalizes the object after validation.
func (strategy) Canonicalize(obj runtime.Object) {
}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (strategy) PrepareForUpdate(ctx api.Context, newObj, oldObj runtime.Object) {
	_ = oldObj.(*api.ConfigMap)
	_ = newObj.(*api.ConfigMap)
}

func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

func (strategy) ValidateUpdate(ctx api.Context, newObj, oldObj runtime.Object) field.ErrorList {
	oldCfg, newCfg := oldObj.(*api.ConfigMap), newObj.(*api.ConfigMap)

	return validation.ValidateConfigMapUpdate(newCfg, oldCfg)
}

// ConfigMapToSelectableFields returns a field set that represents the object for matching purposes.
func ConfigMapToSelectableFields(cfg *api.ConfigMap) fields.Set {
	return generic.ObjectMetaFieldsSet(cfg.ObjectMeta, true)
}

// MatchConfigMap returns a generic matcher for a given label and field selector.
func MatchConfigMap(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			cfg, ok := obj.(*api.ConfigMap)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not of type ConfigMap")
			}

			return labels.Set(cfg.ObjectMeta.Labels), ConfigMapToSelectableFields(cfg), nil
		},
	}
}
