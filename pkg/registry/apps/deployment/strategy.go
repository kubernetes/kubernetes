/*
Copyright 2015 The Kubernetes Authors.

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

package deployment

import (
	"context"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/apps/validation"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

// deploymentStrategy implements behavior for Deployments.
type deploymentStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Deployment
// objects via the REST API.
var Strategy = deploymentStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents for extensions/v1beta1, apps/v1beta1, and apps/v1beta2 for backwards compatibility,
// and DeleteDependents for all other versions.
func (deploymentStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	switch groupVersion {
	case extensionsv1beta1.SchemeGroupVersion, appsv1beta1.SchemeGroupVersion, appsv1beta2.SchemeGroupVersion:
		// for back compatibility
		return rest.OrphanDependents
	default:
		return rest.DeleteDependents
	}
}

// NamespaceScoped is true for deployment.
func (deploymentStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (deploymentStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	deployment := obj.(*apps.Deployment)
	deployment.Status = apps.DeploymentStatus{}
	deployment.Generation = 1

	pod.DropDisabledTemplateFields(&deployment.Spec.Template, nil)
}

// Validate validates a new deployment.
func (deploymentStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	deployment := obj.(*apps.Deployment)
	allErrs := validation.ValidateDeployment(deployment)
	allErrs = append(allErrs, corevalidation.ValidateConditionalPodTemplate(&deployment.Spec.Template, nil, field.NewPath("spec.template"))...)
	return allErrs
}

// Canonicalize normalizes the object after validation.
func (deploymentStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for deployments.
func (deploymentStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (deploymentStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDeployment := obj.(*apps.Deployment)
	oldDeployment := old.(*apps.Deployment)
	newDeployment.Status = oldDeployment.Status

	pod.DropDisabledTemplateFields(&newDeployment.Spec.Template, &oldDeployment.Spec.Template)

	// Spec updates bump the generation so that we can distinguish between
	// scaling events and template changes, annotation updates bump the generation
	// because annotations are copied from deployments to their replica sets.
	if !apiequality.Semantic.DeepEqual(newDeployment.Spec, oldDeployment.Spec) ||
		!apiequality.Semantic.DeepEqual(newDeployment.Annotations, oldDeployment.Annotations) {
		newDeployment.Generation = oldDeployment.Generation + 1
	}
}

// ValidateUpdate is the default update validation for an end user.
func (deploymentStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newDeployment := obj.(*apps.Deployment)
	oldDeployment := old.(*apps.Deployment)
	allErrs := validation.ValidateDeploymentUpdate(newDeployment, oldDeployment)
	allErrs = append(allErrs, corevalidation.ValidateConditionalPodTemplate(&newDeployment.Spec.Template, &oldDeployment.Spec.Template, field.NewPath("spec.template"))...)

	// Update is not allowed to set Spec.Selector for all groups/versions except extensions/v1beta1.
	// If RequestInfo is nil, it is better to revert to old behavior (i.e. allow update to set Spec.Selector)
	// to prevent unintentionally breaking users who may rely on the old behavior.
	// TODO(#50791): after apps/v1beta1 and extensions/v1beta1 are removed,
	// move selector immutability check inside ValidateDeploymentUpdate().
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		switch groupVersion {
		case appsv1beta1.SchemeGroupVersion, extensionsv1beta1.SchemeGroupVersion:
			// no-op for compatibility
		default:
			// disallow mutation of selector
			allErrs = append(allErrs, apivalidation.ValidateImmutableField(newDeployment.Spec.Selector, oldDeployment.Spec.Selector, field.NewPath("spec").Child("selector"))...)
		}
	}

	return allErrs
}

func (deploymentStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type deploymentStatusStrategy struct {
	deploymentStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = deploymentStatusStrategy{Strategy}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (deploymentStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDeployment := obj.(*apps.Deployment)
	oldDeployment := old.(*apps.Deployment)
	newDeployment.Spec = oldDeployment.Spec
	newDeployment.Labels = oldDeployment.Labels
}

// ValidateUpdate is the default update validation for an end user updating status
func (deploymentStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDeploymentStatusUpdate(obj.(*apps.Deployment), old.(*apps.Deployment))
}
