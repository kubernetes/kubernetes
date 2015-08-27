/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package daemon

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// daemonStrategy implements verification logic for daemons.
type daemonStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Daemon objects.
var Strategy = daemonStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all Daemons need to be within a namespace.
func (daemonStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a daemon before creation.
func (daemonStrategy) PrepareForCreate(obj runtime.Object) {
	daemon := obj.(*expapi.Daemon)
	daemon.Status = expapi.DaemonStatus{}

	daemon.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (daemonStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newDaemon := obj.(*expapi.Daemon)
	oldDaemon := old.(*expapi.Daemon)

	// Any changes to the spec increment the generation number, any changes to the
	// status should reflect the generation number of the corresponding object. We push
	// the burden of managing the status onto the clients because we can't (in general)
	// know here what version of spec the writer of the status has seen. It may seem like
	// we can at first -- since obj contains spec -- but in the future we will probably make
	// status its own object, and even if we don't, writes may be the result of a
	// read-update-write loop, so the contents of spec may not actually be the spec that
	// the controller has *seen*.
	//
	// TODO: Any changes to a part of the object that represents desired state (labels,
	// annotations etc) should also increment the generation.
	if !reflect.DeepEqual(oldDaemon.Spec, newDaemon.Spec) {
		newDaemon.Generation = oldDaemon.Generation + 1
	}
}

// Validate validates a new daemon.
func (daemonStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	daemon := obj.(*expapi.Daemon)
	return validation.ValidateDaemon(daemon)
}

// AllowCreateOnUpdate is false for daemon; this means a POST is
// needed to create one
func (daemonStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (daemonStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	validationErrorList := validation.ValidateDaemon(obj.(*expapi.Daemon))
	updateErrorList := validation.ValidateDaemonUpdate(old.(*expapi.Daemon), obj.(*expapi.Daemon))
	return append(validationErrorList, updateErrorList...)
}

// AllowUnconditionalUpdate is the default update policy for daemon objects.
func (daemonStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// DaemonToSelectableFields returns a field set that represents the object.
func DaemonToSelectableFields(daemon *expapi.Daemon) fields.Set {
	return fields.Set{
		"metadata.name": daemon.Name,
	}
}

// MatchDaemon is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchDaemon(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			daemon, ok := obj.(*expapi.Daemon)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not a daemon.")
			}
			return labels.Set(daemon.ObjectMeta.Labels), DaemonToSelectableFields(daemon), nil
		},
	}
}
