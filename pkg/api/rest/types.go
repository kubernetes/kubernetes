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

// NamespaceScoped is true for replication controllers.
func (rcStrategy) NamespaceScoped() bool {
	return true
}

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

// podStrategy implements behavior for Pods
// TODO: move to a pod specific package.
type podStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Pods is the default logic that applies when creating and updating Pod
// objects.
var Pods RESTCreateStrategy = podStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for pods.
func (podStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (podStrategy) ResetBeforeCreate(obj runtime.Object) {
	pod := obj.(*api.Pod)
	pod.Status = api.PodStatus{}
}

// Validate validates a new pod.
func (podStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	pod := obj.(*api.Pod)
	return validation.ValidatePod(pod)
}

// svcStrategy implements behavior for Services
// TODO: move to a service specific package.
type svcStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Services is the default logic that applies when creating and updating Service
// objects.
var Services RESTCreateStrategy = svcStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for services.
func (svcStrategy) NamespaceScoped() bool {
	return true
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (svcStrategy) ResetBeforeCreate(obj runtime.Object) {
	service := obj.(*api.Service)
	service.Status = api.ServiceStatus{}
}

// Validate validates a new service.
func (svcStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	service := obj.(*api.Service)
	return validation.ValidateService(service)
}

// nodeStrategy implements behavior for nodes
// TODO: move to a node specific package.
type nodeStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Nodes is the default logic that applies when creating and updating Node
// objects.
var Nodes RESTCreateStrategy = nodeStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for nodes.
func (nodeStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (nodeStrategy) ResetBeforeCreate(obj runtime.Object) {
	_ = obj.(*api.Node)
	// Nodes allow *all* fields, including status, to be set.
}

// Validate validates a new node.
func (nodeStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	node := obj.(*api.Node)
	return validation.ValidateMinion(node)
}

// namespaceStrategy implements behavior for nodes
type namespaceStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Namespaces is the default logic that applies when creating and updating Namespace
// objects.
var Namespaces RESTCreateStrategy = namespaceStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is false for namespaces.
func (namespaceStrategy) NamespaceScoped() bool {
	return false
}

// ResetBeforeCreate clears fields that are not allowed to be set by end users on creation.
func (namespaceStrategy) ResetBeforeCreate(obj runtime.Object) {
	_ = obj.(*api.Namespace)
	// Namespace allow *all* fields, including status, to be set.
}

// Validate validates a new namespace.
func (namespaceStrategy) Validate(obj runtime.Object) errors.ValidationErrorList {
	namespace := obj.(*api.Namespace)
	return validation.ValidateNamespace(namespace)
}
