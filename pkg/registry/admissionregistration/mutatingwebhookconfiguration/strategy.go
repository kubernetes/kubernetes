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

package mutatingwebhookconfiguration

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// mutatingWebhookConfigurationStrategy implements verification logic for mutatingWebhookConfiguration.
type mutatingWebhookConfigurationStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating mutatingWebhookConfiguration objects.
var Strategy = mutatingWebhookConfigurationStrategy{rest.DeclarativeValidation{Scheme: legacyscheme.Scheme}, names.SimpleNameGenerator}

// NamespaceScoped returns false because MutatingWebhookConfiguration is cluster-scoped resource.
func (mutatingWebhookConfigurationStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an mutatingWebhookConfiguration before creation.
func (mutatingWebhookConfigurationStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.MutatingWebhookConfiguration)
	ic.Generation = 1
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (mutatingWebhookConfigurationStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	ic := obj.(*admissionregistration.MutatingWebhookConfiguration)
	if !utilfeature.DefaultFeatureGate.Enabled(features.ManifestBasedAdmissionControlConfig) {
		return validation.WarningsForStaticSuffix(ic.Name)
	}
	return nil
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (mutatingWebhookConfigurationStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.MutatingWebhookConfiguration)
	oldIC := old.(*admissionregistration.MutatingWebhookConfiguration)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.Webhooks, newIC.Webhooks) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new mutatingWebhookConfiguration.
func (mutatingWebhookConfigurationStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ic := obj.(*admissionregistration.MutatingWebhookConfiguration)
	errs := validation.ValidateMutatingWebhookConfiguration(ic)
	if utilfeature.DefaultFeatureGate.Enabled(features.ManifestBasedAdmissionControlConfig) {
		errs = append(errs, validation.ValidateStaticSuffix(ic.Name, field.NewPath("metadata", "name"))...)
	}
	return errs
}

// Canonicalize normalizes the object after validation.
func (mutatingWebhookConfigurationStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for mutatingWebhookConfiguration; this means you may not create one with a PUT request.
func (mutatingWebhookConfigurationStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (mutatingWebhookConfigurationStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newIC := obj.(*admissionregistration.MutatingWebhookConfiguration)
	oldIC := old.(*admissionregistration.MutatingWebhookConfiguration)
	errs := validation.ValidateMutatingWebhookConfigurationUpdate(newIC, oldIC)
	return errs
}

// WarningsOnUpdate returns warnings for the given update.
func (mutatingWebhookConfigurationStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	newIC := obj.(*admissionregistration.MutatingWebhookConfiguration)
	return validation.WarningsForStaticSuffix(newIC.Name)
}

// AllowUnconditionalUpdate is the default update policy for mutatingWebhookConfiguration objects. Status update should
// only be allowed if version match.
func (mutatingWebhookConfigurationStrategy) AllowUnconditionalUpdate() bool {
	return false
}
