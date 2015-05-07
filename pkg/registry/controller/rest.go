/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package controller

import (
	"fmt"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
)

// rcStrategy implements verification logic for Replication Controllers.
type rcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = rcStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all Replication Controllers need to be within a namespace.
func (rcStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a replication controller before creation.
func (rcStrategy) PrepareForCreate(obj runtime.Object) {
	controller := obj.(*api.ReplicationController)
	controller.Status = api.ReplicationControllerStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (rcStrategy) PrepareForUpdate(obj, old runtime.Object) {
	// TODO: once RC has a status sub-resource we can enable this.
	//newController := obj.(*api.ReplicationController)
	//oldController := old.(*api.ReplicationController)
	//newController.Status = oldController.Status
}

// Validate validates a new replication controller.
func (rcStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	controller := obj.(*api.ReplicationController)
	return validation.ValidateReplicationController(controller)
}

// AllowCreateOnUpdate is false for replication controllers; this means a POST is
// needed to create one.
func (rcStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (rcStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	validationErrorList := validation.ValidateReplicationController(obj.(*api.ReplicationController))
	updateErrorList := validation.ValidateReplicationControllerUpdate(old.(*api.ReplicationController), obj.(*api.ReplicationController))
	return append(validationErrorList, updateErrorList...)
}

// ControllerToSelectableFields returns a label set that represents the object.
func ControllerToSelectableFields(controller *api.ReplicationController) fields.Set {
	return fields.Set{
		"metadata.name":   controller.Name,
		"status.replicas": strconv.Itoa(controller.Status.Replicas),
	}
}

// MatchController is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchController(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			rc, ok := obj.(*api.ReplicationController)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a replication controller.")
			}
			return labels.Set(rc.ObjectMeta.Labels), ControllerToSelectableFields(rc), nil
		},
	}
}
