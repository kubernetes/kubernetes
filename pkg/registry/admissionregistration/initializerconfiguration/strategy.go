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

package initializerconfiguration

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

// initializerConfigurationStrategy implements verification logic for InitializerConfigurations.
type initializerConfigurationStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating InitializerConfiguration objects.
var Strategy = initializerConfigurationStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns true because all InitializerConfiguration' need to be within a namespace.
func (initializerConfigurationStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an InitializerConfiguration before creation.
func (initializerConfigurationStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*admissionregistration.InitializerConfiguration)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (initializerConfigurationStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*admissionregistration.InitializerConfiguration)
	oldIC := old.(*admissionregistration.InitializerConfiguration)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.Initializers, newIC.Initializers) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new InitializerConfiguration.
func (initializerConfigurationStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ic := obj.(*admissionregistration.InitializerConfiguration)
	return validation.ValidateInitializerConfiguration(ic)
}

// Canonicalize normalizes the object after validation.
func (initializerConfigurationStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for InitializerConfiguration; this means you may create one with a PUT request.
func (initializerConfigurationStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (initializerConfigurationStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateInitializerConfiguration(obj.(*admissionregistration.InitializerConfiguration))
	updateErrorList := validation.ValidateInitializerConfigurationUpdate(obj.(*admissionregistration.InitializerConfiguration), old.(*admissionregistration.InitializerConfiguration))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for InitializerConfiguration objects. Status update should
// only be allowed if version match.
func (initializerConfigurationStrategy) AllowUnconditionalUpdate() bool {
	return false
}
