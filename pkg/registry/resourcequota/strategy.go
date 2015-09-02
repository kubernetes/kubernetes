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

package resourcequota

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

// resourcequotaStrategy implements behavior for ResourceQuota objects
type resourcequotaStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ResourceQuota
// objects via the REST API.
var Strategy = resourcequotaStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for resourcequotas.
func (resourcequotaStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (resourcequotaStrategy) PrepareForCreate(obj runtime.Object) {
	resourcequota := obj.(*api.ResourceQuota)
	resourcequota.Status = api.ResourceQuotaStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (resourcequotaStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newResourcequota := obj.(*api.ResourceQuota)
	oldResourcequota := old.(*api.ResourceQuota)
	newResourcequota.Status = oldResourcequota.Status
}

// Validate validates a new resourcequota.
func (resourcequotaStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	resourcequota := obj.(*api.ResourceQuota)
	return validation.ValidateResourceQuota(resourcequota)
}

// AllowCreateOnUpdate is false for resourcequotas.
func (resourcequotaStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (resourcequotaStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	errorList := validation.ValidateResourceQuota(obj.(*api.ResourceQuota))
	return append(errorList, validation.ValidateResourceQuotaUpdate(obj.(*api.ResourceQuota), old.(*api.ResourceQuota))...)
}

func (resourcequotaStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type resourcequotaStatusStrategy struct {
	resourcequotaStrategy
}

var StatusStrategy = resourcequotaStatusStrategy{Strategy}

func (resourcequotaStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newResourcequota := obj.(*api.ResourceQuota)
	oldResourcequota := old.(*api.ResourceQuota)
	newResourcequota.Spec = oldResourcequota.Spec
}

func (resourcequotaStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateResourceQuotaStatusUpdate(obj.(*api.ResourceQuota), old.(*api.ResourceQuota))
}

// MatchResourceQuota returns a generic matcher for a given label and field selector.
func MatchResourceQuota(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		resourcequotaObj, ok := obj.(*api.ResourceQuota)
		if !ok {
			return false, fmt.Errorf("not a resourcequota")
		}
		fields := ResourceQuotaToSelectableFields(resourcequotaObj)
		return label.Matches(labels.Set(resourcequotaObj.Labels)) && field.Matches(fields), nil
	})
}

// ResourceQuotaToSelectableFields returns a label set that represents the object
func ResourceQuotaToSelectableFields(resourcequota *api.ResourceQuota) labels.Set {
	return labels.Set{
		"metadata.name": resourcequota.Name,
	}
}
