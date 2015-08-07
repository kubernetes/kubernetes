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

package component

import (
	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// createStrategy implements create and update validation for Components
type createUpdateStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// CreateUpdateStrategy is the default create/update logic for Component objects.
var CreateUpdateStrategy = createUpdateStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns false because components are global.
func (createUpdateStrategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate returns false because creation handles naming.
// If the component has a name (required for update) but the storage doesn't know about it, something is seriously wrong.
func (createUpdateStrategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate returns true because the user is not required to supply a resource version when performing an update.
func (createUpdateStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by components on creation.
func (createUpdateStrategy) PrepareForCreate(obj runtime.Object) {
	component := obj.(*api.Component)

	component.Name = ""
	component.GenerateName = fmt.Sprintf("%s-", component.Spec.Type)

	now := util.Now()
	component.Status.LastHeartbeatTime = now // TODO(karlkfi): how do we know this is a heartbeat?
	component.Status.LastUpdateTime = now
	component.Status.LastTransitionTime = now
	//TODO(karlkfi): do the timestamps need to match component.ObjectMeta.CreationTimestamp?
}

// PrepareForUpdate clears fields that are not allowed to be set by components on update.
// LastTimestamp is set to the current server time.
func (createUpdateStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newC := obj.(*api.Component)
	oldC := old.(*api.Component)

	// allow some metadata values to be omitted on update (name is unique id)
	newC.CreationTimestamp = oldC.CreationTimestamp
	newC.UID = oldC.UID

	now := util.Now()
	newC.Status.LastHeartbeatTime = now // TODO(karlkfi): how do we know this is a heartbeat?
	newC.Status.LastUpdateTime = now
	if newC.Status.Phase != oldC.Status.Phase {
		newC.Status.LastTransitionTime = now
	}
}

// Validate validates a new component.
func (createUpdateStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	service := obj.(*api.Component)
	return validation.ValidateComponentCreate(service)
}

// Validate validates an update to an existing component.
func (createUpdateStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateComponentUpdate(old.(*api.Component), obj.(*api.Component))
}
