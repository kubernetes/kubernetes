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

package daemonset

import (
	"context"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/apps/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// daemonSetStrategy implements verification logic for daemon sets.
type daemonSetStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating DaemonSet objects.
var Strategy = daemonSetStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Make sure we correctly implement the interface.
var _ = rest.GarbageCollectionDeleteStrategy(Strategy)

// DefaultGarbageCollectionPolicy returns DeleteDependents for all currently served versions.
func (daemonSetStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	return rest.DeleteDependents
}

// NamespaceScoped returns true because all DaemonSets need to be within a namespace.
func (daemonSetStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (daemonSetStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"apps/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a daemon set before creation.
func (daemonSetStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	daemonSet := obj.(*apps.DaemonSet)
	daemonSet.Status = apps.DaemonSetStatus{}

	daemonSet.Generation = 1
	if daemonSet.Spec.TemplateGeneration < 1 {
		daemonSet.Spec.TemplateGeneration = 1
	}

	pod.DropDisabledTemplateFields(&daemonSet.Spec.Template, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (daemonSetStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDaemonSet := obj.(*apps.DaemonSet)
	oldDaemonSet := old.(*apps.DaemonSet)

	pod.DropDisabledTemplateFields(&newDaemonSet.Spec.Template, &oldDaemonSet.Spec.Template)

	// update is not allowed to set status
	newDaemonSet.Status = oldDaemonSet.Status

	// update is not allowed to set TemplateGeneration
	newDaemonSet.Spec.TemplateGeneration = oldDaemonSet.Spec.TemplateGeneration

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the manager has *seen*.
	//
	// TODO: Any changes to a part of the object that represents desired state (labels,
	// annotations etc) should also increment the generation.
	if !apiequality.Semantic.DeepEqual(oldDaemonSet.Spec.Template, newDaemonSet.Spec.Template) {
		newDaemonSet.Spec.TemplateGeneration = oldDaemonSet.Spec.TemplateGeneration + 1
		newDaemonSet.Generation = oldDaemonSet.Generation + 1
		return
	}
	if !apiequality.Semantic.DeepEqual(oldDaemonSet.Spec, newDaemonSet.Spec) {
		newDaemonSet.Generation = oldDaemonSet.Generation + 1
	}
}

// Validate validates a new daemon set.
func (daemonSetStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	daemonSet := obj.(*apps.DaemonSet)
	opts := pod.GetValidationOptionsFromPodTemplate(&daemonSet.Spec.Template, nil)
	return validation.ValidateDaemonSet(daemonSet, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (daemonSetStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newDaemonSet := obj.(*apps.DaemonSet)
	return pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), &newDaemonSet.Spec.Template, nil)
}

// Canonicalize normalizes the object after validation.
func (daemonSetStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for daemon set; this means a POST is
// needed to create one
func (daemonSetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (daemonSetStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newDaemonSet := obj.(*apps.DaemonSet)
	oldDaemonSet := old.(*apps.DaemonSet)

	opts := pod.GetValidationOptionsFromPodTemplate(&newDaemonSet.Spec.Template, &oldDaemonSet.Spec.Template)
	opts.AllowInvalidLabelValueInSelector = opts.AllowInvalidLabelValueInSelector || metav1validation.LabelSelectorHasInvalidLabelValue(oldDaemonSet.Spec.Selector)

	return validation.ValidateDaemonSetUpdate(newDaemonSet, oldDaemonSet, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (daemonSetStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	newDaemonSet := obj.(*apps.DaemonSet)
	oldDaemonSet := old.(*apps.DaemonSet)
	if newDaemonSet.Spec.TemplateGeneration != oldDaemonSet.Spec.TemplateGeneration {
		warnings = pod.GetWarningsForPodTemplate(ctx, field.NewPath("spec", "template"), &newDaemonSet.Spec.Template, &oldDaemonSet.Spec.Template)
	}
	return warnings
}

// AllowUnconditionalUpdate is the default update policy for daemon set objects.
func (daemonSetStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type daemonSetStatusStrategy struct {
	daemonSetStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = daemonSetStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (daemonSetStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"apps/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
}

func (daemonSetStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDaemonSet := obj.(*apps.DaemonSet)
	oldDaemonSet := old.(*apps.DaemonSet)
	newDaemonSet.Spec = oldDaemonSet.Spec
}

func (daemonSetStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDaemonSetStatusUpdate(obj.(*apps.DaemonSet), old.(*apps.DaemonSet))
}

// WarningsOnUpdate returns warnings for the given update.
func (daemonSetStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
