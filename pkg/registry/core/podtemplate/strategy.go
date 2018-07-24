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

package podtemplate

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
)

// podTemplateStrategy implements behavior for PodTemplates
type podTemplateStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodTemplate
// objects via the REST API.
var Strategy = podTemplateStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for pod templates.
func (podTemplateStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (podTemplateStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	template := obj.(*api.PodTemplate)

	pod.DropDisabledAlphaFields(&template.Template.Spec)
}

// Validate validates a new pod template.
func (podTemplateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pod := obj.(*api.PodTemplate)
	return validation.ValidatePodTemplate(pod)
}

// Canonicalize normalizes the object after validation.
func (podTemplateStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for pod templates.
func (podTemplateStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podTemplateStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newTemplate := obj.(*api.PodTemplate)
	oldTemplate := old.(*api.PodTemplate)

	pod.DropDisabledAlphaFields(&newTemplate.Template.Spec)
	pod.DropDisabledAlphaFields(&oldTemplate.Template.Spec)
}

// ValidateUpdate is the default update validation for an end user.
func (podTemplateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodTemplateUpdate(obj.(*api.PodTemplate), old.(*api.PodTemplate))
}

func (podTemplateStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func (podTemplateStrategy) Export(ctx context.Context, obj runtime.Object, exact bool) error {
	// Do nothing
	return nil
}
