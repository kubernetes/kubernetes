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

package externaladmissionhookconfiguration

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// externaladmissionhookConfigurationStrategy implements verification logic for ExternalAdmissionHookConfiguration.
type externaladmissionhookConfigurationStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ExternalAdmissionHookConfiguration objects.
var Strategy = externaladmissionhookConfigurationStrategy{api.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all ExternalAdmissionHookConfiguration' need to be within a namespace.
func (externaladmissionhookConfigurationStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an ExternalAdmissionHookConfiguration before creation.
func (externaladmissionhookConfigurationStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.ExternalAdmissionHookConfiguration)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (externaladmissionhookConfigurationStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ExternalAdmissionHookConfiguration)
	oldIC := old.(*admissionregistration.ExternalAdmissionHookConfiguration)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.ExternalAdmissionHooks, newIC.ExternalAdmissionHooks) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new ExternalAdmissionHookConfiguration.
func (externaladmissionhookConfigurationStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	ic := obj.(*admissionregistration.ExternalAdmissionHookConfiguration)
	return validation.ValidateExternalAdmissionHookConfiguration(ic)
}

// Canonicalize normalizes the object after validation.
func (externaladmissionhookConfigurationStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for ExternalAdmissionHookConfiguration; this means you may create one with a PUT request.
func (externaladmissionhookConfigurationStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (externaladmissionhookConfigurationStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateExternalAdmissionHookConfiguration(obj.(*admissionregistration.ExternalAdmissionHookConfiguration))
	updateErrorList := validation.ValidateExternalAdmissionHookConfigurationUpdate(obj.(*admissionregistration.ExternalAdmissionHookConfiguration), old.(*admissionregistration.ExternalAdmissionHookConfiguration))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for ExternalAdmissionHookConfiguration objects. Status update should
// only be allowed if version match.
func (externaladmissionhookConfigurationStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// MatchReplicaSet is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchExternalAdmissionHookConfiguration(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	ic, ok := obj.(*admissionregistration.ExternalAdmissionHookConfiguration)
	if !ok {
		return nil, nil, false, fmt.Errorf("Given object is not a ExternalAdmissionHookConfiguration.")
	}
	return labels.Set(ic.ObjectMeta.Labels), ExternalAdmissionHookConfigurationToSelectableFields(ic), ic.Initializers != nil, nil
}

// ExternalAdmissionHookConfigurationToSelectableFields returns a field set that represents the object.
func ExternalAdmissionHookConfigurationToSelectableFields(ic *admissionregistration.ExternalAdmissionHookConfiguration) fields.Set {
	return generic.ObjectMetaFieldsSet(&ic.ObjectMeta, false)
}
