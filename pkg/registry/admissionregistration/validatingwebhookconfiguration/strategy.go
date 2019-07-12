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

package validatingwebhookconfiguration

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// validatingWebhookConfigurationStrategy implements verification logic for validatingWebhookConfiguration.
type validatingWebhookConfigurationStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating validatingWebhookConfiguration objects.
var Strategy = validatingWebhookConfigurationStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all validatingWebhookConfiguration' need to be within a namespace.
func (validatingWebhookConfigurationStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an validatingWebhookConfiguration before creation.
func (validatingWebhookConfigurationStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.ValidatingWebhookConfiguration)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (validatingWebhookConfigurationStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.ValidatingWebhookConfiguration)
	oldIC := old.(*admissionregistration.ValidatingWebhookConfiguration)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.Webhooks, newIC.Webhooks) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new validatingWebhookConfiguration.
func (validatingWebhookConfigurationStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}

	return validation.ValidateValidatingWebhookConfiguration(obj.(*admissionregistration.ValidatingWebhookConfiguration), groupVersion)
}

// Canonicalize normalizes the object after validation.
func (validatingWebhookConfigurationStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for validatingWebhookConfiguration; this means you may create one with a PUT request.
func (validatingWebhookConfigurationStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (validatingWebhookConfigurationStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}

	return validation.ValidateValidatingWebhookConfigurationUpdate(obj.(*admissionregistration.ValidatingWebhookConfiguration), old.(*admissionregistration.ValidatingWebhookConfiguration), groupVersion)
}

// AllowUnconditionalUpdate is the default update policy for validatingWebhookConfiguration objects. Status update should
// only be allowed if version match.
func (validatingWebhookConfigurationStrategy) AllowUnconditionalUpdate() bool {
	return false
}
