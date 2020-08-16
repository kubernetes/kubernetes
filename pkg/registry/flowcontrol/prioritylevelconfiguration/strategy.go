/*
Copyright 2019 The Kubernetes Authors.

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

package prioritylevelconfiguration

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/validation"
)

// priorityLevelConfigurationStrategy implements verification logic for priority level configurations.
type priorityLevelConfigurationStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating priority level configuration objects.
var Strategy = priorityLevelConfigurationStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all PriorityClasses are global.
func (priorityLevelConfigurationStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of a priority-level-configuration before creation.
func (priorityLevelConfigurationStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pl := obj.(*flowcontrol.PriorityLevelConfiguration)
	pl.Status = flowcontrol.PriorityLevelConfigurationStatus{}
	pl.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (priorityLevelConfigurationStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPriorityLevelConfiguration := obj.(*flowcontrol.PriorityLevelConfiguration)
	oldPriorityLevelConfiguration := old.(*flowcontrol.PriorityLevelConfiguration)

	// Spec updates bump the generation so that we can distinguish between status updates.
	if !apiequality.Semantic.DeepEqual(newPriorityLevelConfiguration.Spec, oldPriorityLevelConfiguration.Spec) {
		newPriorityLevelConfiguration.Generation = oldPriorityLevelConfiguration.Generation + 1
	}
	newPriorityLevelConfiguration.Status = oldPriorityLevelConfiguration.Status
}

// Validate validates a new priority-level.
func (priorityLevelConfigurationStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidatePriorityLevelConfiguration(obj.(*flowcontrol.PriorityLevelConfiguration))
}

// Canonicalize normalizes the object after validation.
func (priorityLevelConfigurationStrategy) Canonicalize(obj runtime.Object) {
}

func (priorityLevelConfigurationStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for priority-level-configurations; this means a POST is needed to create one.
func (priorityLevelConfigurationStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (priorityLevelConfigurationStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePriorityLevelConfiguration(obj.(*flowcontrol.PriorityLevelConfiguration))
}

type priorityLevelConfigurationStatusStrategy struct {
	priorityLevelConfigurationStrategy
}

// StatusStrategy is the default logic that applies when updating priority level configuration objects' status.
var StatusStrategy = priorityLevelConfigurationStatusStrategy{Strategy}

func (priorityLevelConfigurationStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPriorityLevelConfiguration := obj.(*flowcontrol.PriorityLevelConfiguration)
	oldPriorityLevelConfiguration := old.(*flowcontrol.PriorityLevelConfiguration)

	// managedFields must be preserved since it's been modified to
	// track changed fields in the status update.
	managedFields := newPriorityLevelConfiguration.ManagedFields
	newPriorityLevelConfiguration.ObjectMeta = oldPriorityLevelConfiguration.ObjectMeta
	newPriorityLevelConfiguration.ManagedFields = managedFields
	newPriorityLevelConfiguration.Spec = oldPriorityLevelConfiguration.Spec
}

func (priorityLevelConfigurationStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePriorityLevelConfigurationStatusUpdate(old.(*flowcontrol.PriorityLevelConfiguration), obj.(*flowcontrol.PriorityLevelConfiguration))
}
