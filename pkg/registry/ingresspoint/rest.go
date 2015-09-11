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

package ingresspoint

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// ingressPointStrategy implements verification logic for Replication IngressPoints.
type ingressPointStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication IngressPoint objects.
var Strategy = ingressPointStrategy{api.Scheme, api.SimpleNameGenerator}

func (ingressPointStrategy) NamespaceScoped() bool {
	return true
}

func (ingressPointStrategy) PrepareForCreate(obj runtime.Object) {
}

func (ingressPointStrategy) PrepareForUpdate(obj, old runtime.Object) {
}

func (ingressPointStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	return fielderrors.ValidationErrorList{}
}

func (ingressPointStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (ingressPointStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return fielderrors.ValidationErrorList{}
}

func (ingressPointStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// IngressPointToSelectableFields returns a label set that represents the object.
func IngressPointToSelectableFields(ingressPoint *api.IngressPoint) fields.Set {
	return fields.Set{
		"metadata.name": ingressPoint.Name,
	}
}

// MatchIngressPoint is the filter used by the generic etcd backend to ingressPoint
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchIngressPoint(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			ingressPoint, ok := obj.(*api.IngressPoint)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a replication controller.")
			}
			return labels.Set(ingressPoint.ObjectMeta.Labels), IngressPointToSelectableFields(ingressPoint), nil
		},
	}
}
