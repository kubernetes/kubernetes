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

package component

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
)

// ComponentToSelectableFields returns a field set that represents the object.
// These fields can be used for filtering.
// TODO(karlkfi): add time fields?
func ComponentToSelectableFields(component *api.Component) fields.Set {
	return fields.Set{
		"metadata.name": component.ObjectMeta.Name,
		"spec.type":     string(component.Spec.Type),
		"spec.address":  component.Spec.Address,
		"status.phase":  string(component.Status.Phase),
	}
}

// MatchNode returns a generic matcher for a given label and field selector.
func MatchComponent(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			component, ok := obj.(*api.Component)
			if !ok {
				return nil, nil, fmt.Errorf("not a component")
			}
			return labels.Set(component.ObjectMeta.Labels), ComponentToSelectableFields(component), nil
		},
	}
}
