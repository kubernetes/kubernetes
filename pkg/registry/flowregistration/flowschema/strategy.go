/*
Copyright 2018 The Kubernetes Authors.

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

package flowschema

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	flow "k8s.io/kubernetes/pkg/apis/flowregistration"
	"k8s.io/kubernetes/pkg/apis/flowregistration/validation"
)

// flowSchemaStrategy implements verification logic for FlowSchema.
type flowSchemaStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating FlowSchema objects.
var Strategy = flowSchemaStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all FlowSchema's need to be cluster scoped
func (flowSchemaStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an FlowSchema before creation.
func (flowSchemaStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*flow.FlowSchema)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (flowSchemaStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*flow.FlowSchema)
	oldIC := old.(*flow.FlowSchema)

	// Any changes to the policy or backend increment the generation number
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new flowSchema.
func (flowSchemaStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ic := obj.(*flow.FlowSchema)
	return validation.ValidateFlowSchema(ic)
}

// Canonicalize normalizes the object after validation.
func (flowSchemaStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for flowSchema; this means you may create one with a PUT request.
func (flowSchemaStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (flowSchemaStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateFlowSchema(obj.(*flow.FlowSchema))
	updateErrorList := validation.ValidateFlowSchemaUpdate(obj.(*flow.FlowSchema), old.(*flow.FlowSchema))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for flowSchema objects. Status update should
// only be allowed if version match.
func (flowSchemaStrategy) AllowUnconditionalUpdate() bool {
	return false
}
