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
)

// daemonSetStrategy implements verification logic for daemon sets.
type daemonSetStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating DaemonSet objects.
var Strategy = daemonSetStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents for extensions/v1beta1 and apps/v1beta2 for backwards compatibility,
// and DeleteDependents for all other versions.
func (daemonSetStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	switch groupVersion {
	case extensionsv1beta1.SchemeGroupVersion, appsv1beta2.SchemeGroupVersion:
		// for back compatibility
		return rest.OrphanDependents
	default:
		return rest.DeleteDependents
	}
}

// NamespaceScoped returns true because all DaemonSets need to be within a namespace.
func (daemonSetStrategy) NamespaceScoped() bool {
	return true
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
	return validation.ValidateDaemonSet(daemonSet)
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
	allErrs := validation.ValidateDaemonSet(obj.(*apps.DaemonSet))
	allErrs = append(allErrs, validation.ValidateDaemonSetUpdate(newDaemonSet, oldDaemonSet)...)

	// Update is not allowed to set Spec.Selector for apps/v1 and apps/v1beta2 (allowed for extensions/v1beta1).
	// If RequestInfo is nil, it is better to revert to old behavior (i.e. allow update to set Spec.Selector)
	// to prevent unintentionally breaking users who may rely on the old behavior.
	// TODO(#50791): after extensions/v1beta1 is removed, move selector immutability check inside ValidateDaemonSetUpdate().
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		switch groupVersion {
		case extensionsv1beta1.SchemeGroupVersion:
			// no-op for compatibility
		default:
			// disallow mutation of selector
			allErrs = append(allErrs, apivalidation.ValidateImmutableField(newDaemonSet.Spec.Selector, oldDaemonSet.Spec.Selector, field.NewPath("spec").Child("selector"))...)
		}
	}

	return allErrs
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

func (daemonSetStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newDaemonSet := obj.(*apps.DaemonSet)
	oldDaemonSet := old.(*apps.DaemonSet)
	newDaemonSet.Spec = oldDaemonSet.Spec
}

func (daemonSetStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDaemonSetStatusUpdate(obj.(*apps.DaemonSet), old.(*apps.DaemonSet))
}
