/*
Copyright 2014 Google Inc. All rights reserved.

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

package rest

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// rcStrategy implements behavior for Replication Controllers.
// TODO: move to a replicationcontroller specific package.
type rcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// ReplicationControllers is the default logic that applies when creating and updating Replication Controller
// objects.
var ReplicationControllers RESTCreateStrategy = rcStrategy{api.Scheme, api.SimpleNameGenerator}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (rcStrategy) ResetBeforeCreate(obj runtime.Object) {
	controller := obj.(*api.ReplicationController)
	controller.Status = api.ReplicationControllerStatus{}
}

// Validate validates a new replication controller.
func (rcStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	controller := obj.(*api.ReplicationController)
	return validation.ValidateReplicationController(controller)
}
