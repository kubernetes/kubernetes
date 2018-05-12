/*
Copyright 2016 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"context"
	"fmt"
	"strconv"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/validation"
)

// rsStrategy implements verification logic for ReplicaSets.
type rsStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating ReplicaSet objects.
var Strategy = rsStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents by default. For apps/v1, returns DeleteDependents.
func (rsStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		switch groupVersion {
		case extensionsv1beta1.SchemeGroupVersion, appsv1beta2.SchemeGroupVersion:
			// for back compatibility
			return rest.OrphanDependents
		default:
			return rest.DeleteDependents
		}
	}
	return rest.OrphanDependents
}

// NamespaceScoped returns true because all ReplicaSets need to be within a namespace.
func (rsStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a ReplicaSet before creation.
func (rsStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	rs := obj.(*extensions.ReplicaSet)
	rs.Status = extensions.ReplicaSetStatus{}

	rs.Generation = 1

	pod.DropDisabledAlphaFields(&rs.Spec.Template.Spec)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (rsStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRS := obj.(*extensions.ReplicaSet)
	oldRS := old.(*extensions.ReplicaSet)
	// update is not allowed to set status
	newRS.Status = oldRS.Status

	pod.DropDisabledAlphaFields(&newRS.Spec.Template.Spec)
	pod.DropDisabledAlphaFields(&oldRS.Spec.Template.Spec)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the ReplicaSet has *seen*.
	if !apiequality.Semantic.DeepEqual(oldRS.Spec, newRS.Spec) {
		newRS.Generation = oldRS.Generation + 1
	}
}

// Validate validates a new ReplicaSet.
func (rsStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	rs := obj.(*extensions.ReplicaSet)
	return validation.ValidateReplicaSet(rs)
}

// Canonicalize normalizes the object after validation.
func (rsStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for ReplicaSets; this means a POST is
// needed to create one.
func (rsStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (rsStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newReplicaSet := obj.(*extensions.ReplicaSet)
	oldReplicaSet := old.(*extensions.ReplicaSet)
	allErrs := validation.ValidateReplicaSet(obj.(*extensions.ReplicaSet))
	allErrs = append(allErrs, validation.ValidateReplicaSetUpdate(newReplicaSet, oldReplicaSet)...)

	// Update is not allowed to set Spec.Selector for all groups/versions except extensions/v1beta1.
	// If RequestInfo is nil, it is better to revert to old behavior (i.e. allow update to set Spec.Selector)
	// to prevent unintentionally breaking users who may rely on the old behavior.
	// TODO(#50791): after extensions/v1beta1 is removed, move selector immutability check inside ValidateReplicaSetUpdate().
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		switch groupVersion {
		case extensionsv1beta1.SchemeGroupVersion:
			// no-op for compatibility
		default:
			// disallow mutation of selector
			allErrs = append(allErrs, apivalidation.ValidateImmutableField(newReplicaSet.Spec.Selector, oldReplicaSet.Spec.Selector, field.NewPath("spec").Child("selector"))...)
		}
	}

	return allErrs
}

func (rsStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// ReplicaSetToSelectableFields returns a field set that represents the object.
func ReplicaSetToSelectableFields(rs *extensions.ReplicaSet) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&rs.ObjectMeta, true)
	rsSpecificFieldsSet := fields.Set{
		"status.replicas": strconv.Itoa(int(rs.Status.Replicas)),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, rsSpecificFieldsSet)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	rs, ok := obj.(*extensions.ReplicaSet)
	if !ok {
		return nil, nil, false, fmt.Errorf("given object is not a ReplicaSet.")
	}
	return labels.Set(rs.ObjectMeta.Labels), ReplicaSetToSelectableFields(rs), rs.Initializers != nil, nil
}

// MatchReplicaSet is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchReplicaSet(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

type rsStatusStrategy struct {
	rsStrategy
}

var StatusStrategy = rsStatusStrategy{Strategy}

func (rsStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRS := obj.(*extensions.ReplicaSet)
	oldRS := old.(*extensions.ReplicaSet)
	// update is not allowed to set spec
	newRS.Spec = oldRS.Spec
}

func (rsStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateReplicaSetStatusUpdate(obj.(*extensions.ReplicaSet), old.(*extensions.ReplicaSet))
}
