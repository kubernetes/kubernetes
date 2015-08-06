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
type createStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// CreateStrategy is the default create/update logic for Component objects.
var CreateStrategy = createStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns false because components are global.
func (createStrategy) NamespaceScoped() bool {
	return false
}

// AllowCreateOnUpdate returns false because creation handles naming.
// If the component has a name (required for update) but the storage doesn't know about it, something is seriously wrong.
func (createStrategy) AllowCreateOnUpdate() bool {
	return false
}

// AllowUnconditionalUpdate returns true because the user is not required to supply a resource version when performing an update.
func (createStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by components on creation.
func (createStrategy) PrepareForCreate(obj runtime.Object) {
	component := obj.(*api.Component)

	component.Name = ""
	component.GenerateName = fmt.Sprintf("%s-", component.Type)
	//TODO(karlkfi): does LastTimestamp need to match component.ObjectMeta.CreationTimestamp?
	component.LastTimestamp = util.Now()
}

// PrepareForUpdate clears fields that are not allowed to be set by components on update.
// LastTimestamp is set to the current server time.
func (createStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newC := obj.(*api.Component)
	oldC := old.(*api.Component)

	newC.LastTimestamp = util.Now()
	newC.CreationTimestamp = oldC.CreationTimestamp
	newC.UID = oldC.UID //TODO(karlkfi): validate uid match or just use name??
}

// Validate validates a new component.
func (createStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	service := obj.(*api.Component)
	return validation.ValidateComponent(service)
}

// Validate validates an update to an existing component.
func (createStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateComponentUpdate(old.(*api.Component), obj.(*api.Component))
}
