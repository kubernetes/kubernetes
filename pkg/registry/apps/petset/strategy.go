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

package petset

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/apps/validation"
)

// statefulSetStrategy implements verification logic for Replication StatefulSets.
type statefulSetStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication StatefulSet objects.
var Strategy = statefulSetStrategy{api.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns Orphan because that was the default
// behavior before the server-side garbage collection was implemented.
func (statefulSetStrategy) DefaultGarbageCollectionPolicy() rest.GarbageCollectionPolicy {
	return rest.OrphanDependents
}

// NamespaceScoped returns true because all StatefulSet' need to be within a namespace.
func (statefulSetStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of an StatefulSet before creation.
func (statefulSetStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	statefulSet := obj.(*apps.StatefulSet)
	// create cannot set status
	statefulSet.Status = apps.StatefulSetStatus{}

	statefulSet.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (statefulSetStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newStatefulSet := obj.(*apps.StatefulSet)
	oldStatefulSet := old.(*apps.StatefulSet)
	// Update is not allowed to set status
	newStatefulSet.Status = oldStatefulSet.Status

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object.
	// See metav1.ObjectMeta description for more information on Generation.
	if !reflect.DeepEqual(oldStatefulSet.Spec, newStatefulSet.Spec) {
		newStatefulSet.Generation = oldStatefulSet.Generation + 1
	}

}

// Validate validates a new StatefulSet.
func (statefulSetStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	statefulSet := obj.(*apps.StatefulSet)
	return validation.ValidateStatefulSet(statefulSet)
}

// Canonicalize normalizes the object after validation.
func (statefulSetStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for StatefulSet; this means POST is needed to create one.
func (statefulSetStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (statefulSetStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateStatefulSet(obj.(*apps.StatefulSet))
	updateErrorList := validation.ValidateStatefulSetUpdate(obj.(*apps.StatefulSet), old.(*apps.StatefulSet))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for StatefulSet objects.
func (statefulSetStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// StatefulSetToSelectableFields returns a field set that represents the object.
func StatefulSetToSelectableFields(statefulSet *apps.StatefulSet) fields.Set {
	return generic.ObjectMetaFieldsSet(&statefulSet.ObjectMeta, true)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	statefulSet, ok := obj.(*apps.StatefulSet)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not an StatefulSet.")
	}
	return labels.Set(statefulSet.ObjectMeta.Labels), StatefulSetToSelectableFields(statefulSet), nil
}

// MatchStatefulSet is the filter used by the generic etcd backend to watch events
// from etcd to clients of the apiserver only interested in specific labels/fields.
func MatchStatefulSet(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

type statefulSetStatusStrategy struct {
	statefulSetStrategy
}

var StatusStrategy = statefulSetStatusStrategy{Strategy}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (statefulSetStatusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newStatefulSet := obj.(*apps.StatefulSet)
	oldStatefulSet := old.(*apps.StatefulSet)
	// status changes are not allowed to update spec
	newStatefulSet.Spec = oldStatefulSet.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (statefulSetStatusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	// TODO: Validate status updates.
	return validation.ValidateStatefulSetStatusUpdate(obj.(*apps.StatefulSet), old.(*apps.StatefulSet))
}
