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

// PrepareForCreate initializes the status and enables name generation (if name is empty).
func (createUpdateStrategy) PrepareForCreate(obj runtime.Object) {
	component := obj.(*api.Component)

	// Generate the name by default
	if component.Name == "" {
		component.GenerateName = fmt.Sprintf("%s-", component.Spec.Type)
	}

	// Status sent by creator will be ignored
	now := util.Now()
	component.Status = api.ComponentStatus{
		Phase:              api.ComponentPending,
		Conditions:         []api.ComponentCondition{},
		LastUpdateTime:     now,
		LastTransitionTime: now,
	}
	// TODO(karlkfi): do the timestamps need to match component.ObjectMeta.CreationTimestamp? Is CreationTimestamp already populated?
}

// PrepareForUpdate updates timestamps.
func (createUpdateStrategy) PrepareForUpdate(newO, oldO runtime.Object) {
	component := newO.(*api.Component)
	old := oldO.(*api.Component)

	// Allow some metadata values to be omitted on update
	component.CreationTimestamp = old.CreationTimestamp

	// Component update should not be used if only the status phase changed. Use Component/Status update instead.
	now := util.Now()
	// every update bumps LastUpdateTime
	component.Status.LastUpdateTime = now
	// only updates that change the phase bump LastTransitionTime
	if component.Status.Phase != old.Status.Phase {
		component.Status.LastTransitionTime = now
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
