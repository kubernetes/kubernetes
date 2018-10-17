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

package auditsink

import (
	"context"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	audit "k8s.io/kubernetes/pkg/apis/auditregistration"
	"k8s.io/kubernetes/pkg/apis/auditregistration/validation"
)

// auditSinkStrategy implements verification logic for AuditSink.
type auditSinkStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating AuditSink objects.
var Strategy = auditSinkStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all AuditSink's need to be cluster scoped
func (auditSinkStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an AuditSink before creation.
func (auditSinkStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	ic := obj.(*audit.AuditSink)
	ic.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (auditSinkStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIC := obj.(*audit.AuditSink)
	oldIC := old.(*audit.AuditSink)

	// Any changes to the policy or backend increment the generation number
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldIC.Spec, newIC.Spec) {
		newIC.Generation = oldIC.Generation + 1
	}
}

// Validate validates a new auditSink.
func (auditSinkStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ic := obj.(*audit.AuditSink)
	return validation.ValidateAuditSink(ic)
}

// Canonicalize normalizes the object after validation.
func (auditSinkStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is true for auditSink; this means you may create one with a PUT request.
func (auditSinkStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (auditSinkStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateAuditSink(obj.(*audit.AuditSink))
	updateErrorList := validation.ValidateAuditSinkUpdate(obj.(*audit.AuditSink), old.(*audit.AuditSink))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for auditSink objects. Status update should
// only be allowed if version match.
func (auditSinkStrategy) AllowUnconditionalUpdate() bool {
	return false
}
