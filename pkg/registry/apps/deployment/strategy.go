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
	"fmt"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsvalidation "k8s.io/kubernetes/pkg/apis/apps/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// deploymentStrategy implements behavior for Deployments.
type deploymentStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Deployment
// objects via the REST API.
var Strategy = deploymentStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Make sure we correctly implement the interface.
var _ = rest.GarbageCollectionDeleteStrategy(Strategy)

// DefaultGarbageCollectionPolicy returns DeleteDependents for all currently served versions.
func (deploymentStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	return rest.DeleteDependents
}

// NamespaceScoped is true for deployment.
func (deploymentStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (deploymentStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apps/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
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
	opts := pod.GetValidationOptionsFromPodTemplate(&deployment.Spec.Template, nil)
	return appsvalidation.ValidateDeployment(deployment, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (deploymentStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newDeployment := obj.(*apps.Deployment)
	var warnings []string
	if msgs := utilvalidation.IsDNS1123Label(newDeployment.Name); len(msgs) != 0 {
		warnings = append(warnings, fmt.Sprintf("metadata.name: this is used in Pod names and hostnames, which can result in surprising behavior; a DNS label is recommended: %v", msgs))
	}
	warnings = append(warnings, pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), &newDeployment.Spec.Template, nil)...)
	return warnings
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

	opts := pod.GetValidationOptionsFromPodTemplate(&newDeployment.Spec.Template, &oldDeployment.Spec.Template)
	allErrs := appsvalidation.ValidateDeploymentUpdate(newDeployment, oldDeployment, opts)

	return allErrs
}

// WarningsOnUpdate returns warnings for the given update.
func (deploymentStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	newDeployment := obj.(*apps.Deployment)
	oldDeployment := old.(*apps.Deployment)
	if newDeployment.Generation != oldDeployment.Generation {
		warnings = pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), &newDeployment.Spec.Template, &oldDeployment.Spec.Template)
	}
	return warnings
}

func (deploymentStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type deploymentStatusStrategy struct {
	deploymentStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = deploymentStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (deploymentStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"apps/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata", "labels"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (deploymentStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDeployment := obj.(*apps.Deployment)
	oldDeployment := old.(*apps.Deployment)
	newDeployment.Spec = oldDeployment.Spec
	newDeployment.Labels = oldDeployment.Labels
	dropDisabledStatusFields(&newDeployment.Status, &oldDeployment.Status)
}

// ValidateUpdate is the default update validation for an end user updating status
func (deploymentStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return appsvalidation.ValidateDeploymentStatusUpdate(obj.(*apps.Deployment), old.(*apps.Deployment))
}

// WarningsOnUpdate returns warnings for the given update.
func (deploymentStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// dropDisabledStatusFields removes disabled fields from the deployment status.
func dropDisabledStatusFields(deploymentStatus, oldDeploymentStatus *apps.DeploymentStatus) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DeploymentReplicaSetTerminatingReplicas) &&
		(oldDeploymentStatus == nil || oldDeploymentStatus.TerminatingReplicas == nil) {
		deploymentStatus.TerminatingReplicas = nil
	}
}
