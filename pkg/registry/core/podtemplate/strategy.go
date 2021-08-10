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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
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
	template.Generation = 1
	pod.DropDisabledTemplateFields(&template.Template, nil)
}

// Validate validates a new pod template.
func (podTemplateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	template := obj.(*api.PodTemplate)
	opts := pod.GetValidationOptionsFromPodTemplate(&template.Template, nil)
	return corevalidation.ValidatePodTemplate(template, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (podTemplateStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newPodTemplate := obj.(*api.PodTemplate)
	return pod.GetWarningsForPodTemplate(ctx, field.NewPath("template"), &newPodTemplate.Template, nil)
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

	pod.DropDisabledTemplateFields(&newTemplate.Template, &oldTemplate.Template)

	// Any changes to the template increment the generation number.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(newTemplate.Template, oldTemplate.Template) {
		newTemplate.Generation = oldTemplate.Generation + 1
	}

}

// ValidateUpdate is the default update validation for an end user.
func (podTemplateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	template := obj.(*api.PodTemplate)
	oldTemplate := old.(*api.PodTemplate)

	// Allow downward api usage of hugepages on pod update if feature is enabled or if the old pod already had used them.
	opts := pod.GetValidationOptionsFromPodTemplate(&template.Template, &oldTemplate.Template)
	return corevalidation.ValidatePodTemplateUpdate(template, oldTemplate, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (podTemplateStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	newTemplate := obj.(*api.PodTemplate)
	oldTemplate := old.(*api.PodTemplate)
	if newTemplate.Generation != oldTemplate.Generation {
		warnings = pod.GetWarningsForPodTemplate(ctx, field.NewPath("template"), &newTemplate.Template, &oldTemplate.Template)
	}
	return warnings
}

func (podTemplateStrategy) AllowUnconditionalUpdate() bool {
	return true
}
