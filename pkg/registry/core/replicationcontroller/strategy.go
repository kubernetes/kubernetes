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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replicationcontroller

import (
	"fmt"
	"strconv"
	"strings"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	apistorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/api/validation"
)

// rcStrategy implements verification logic for Replication Controllers.
type rcStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = rcStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns Orphan because that was the default
// behavior before the server-side garbage collection was implemented.
func (rcStrategy) DefaultGarbageCollectionPolicy() rest.GarbageCollectionPolicy {
	return rest.OrphanDependents
}

// NamespaceScoped returns true because all Replication Controllers need to be within a namespace.
func (rcStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a replication controller before creation.
func (rcStrategy) PrepareForCreate(ctx genericapirequest.Context, obj runtime.Object) {
	controller := obj.(*api.ReplicationController)
	controller.Status = api.ReplicationControllerStatus{}

	controller.Generation = 1

	if controller.Spec.Template != nil {
		pod.DropDisabledAlphaFields(&controller.Spec.Template.Spec)
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (rcStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newController := obj.(*api.ReplicationController)
	oldController := old.(*api.ReplicationController)
	// update is not allowed to set status
	newController.Status = oldController.Status

	if oldController.Spec.Template != nil {
		pod.DropDisabledAlphaFields(&oldController.Spec.Template.Spec)
	}
	if newController.Spec.Template != nil {
		pod.DropDisabledAlphaFields(&newController.Spec.Template.Spec)
	}

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the controller has *seen*.
	if !apiequality.Semantic.DeepEqual(oldController.Spec, newController.Spec) {
		newController.Generation = oldController.Generation + 1
	}
}

// Validate validates a new replication controller.
func (rcStrategy) Validate(ctx genericapirequest.Context, obj runtime.Object) field.ErrorList {
	controller := obj.(*api.ReplicationController)
	return validation.ValidateReplicationController(controller)
}

// Canonicalize normalizes the object after validation.
func (rcStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for replication controllers; this means a POST is
// needed to create one.
func (rcStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (rcStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	oldRc := old.(*api.ReplicationController)
	newRc := obj.(*api.ReplicationController)

	validationErrorList := validation.ValidateReplicationController(newRc)
	updateErrorList := validation.ValidateReplicationControllerUpdate(newRc, oldRc)
	errs := append(validationErrorList, updateErrorList...)

	for key, value := range helper.NonConvertibleFields(oldRc.Annotations) {
		parts := strings.Split(key, "/")
		if len(parts) != 2 {
			continue
		}
		brokenField := parts[1]

		switch {
		case strings.Contains(brokenField, "selector"):
			if !apiequality.Semantic.DeepEqual(oldRc.Spec.Selector, newRc.Spec.Selector) {
				errs = append(errs, field.Invalid(field.NewPath("spec").Child("selector"), newRc.Spec.Selector, "cannot update non-convertible selector"))
			}
		default:
			errs = append(errs, &field.Error{Type: field.ErrorTypeNotFound, BadValue: value, Field: brokenField, Detail: "unknown non-convertible field"})
		}
	}

	return errs
}

func (rcStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// ControllerToSelectableFields returns a field set that represents the object.
func ControllerToSelectableFields(controller *api.ReplicationController) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&controller.ObjectMeta, true)
	controllerSpecificFieldsSet := fields.Set{
		"status.replicas": strconv.Itoa(int(controller.Status.Replicas)),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, controllerSpecificFieldsSet)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	rc, ok := obj.(*api.ReplicationController)
	if !ok {
		return nil, nil, false, fmt.Errorf("given object is not a replication controller.")
	}
	return labels.Set(rc.ObjectMeta.Labels), ControllerToSelectableFields(rc), rc.Initializers != nil, nil
}

// MatchController is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchController(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

type rcStatusStrategy struct {
	rcStrategy
}

var StatusStrategy = rcStatusStrategy{Strategy}

func (rcStatusStrategy) PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object) {
	newRc := obj.(*api.ReplicationController)
	oldRc := old.(*api.ReplicationController)
	// update is not allowed to set spec
	newRc.Spec = oldRc.Spec
}

func (rcStatusStrategy) ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateReplicationControllerStatusUpdate(obj.(*api.ReplicationController), old.(*api.ReplicationController))
}
