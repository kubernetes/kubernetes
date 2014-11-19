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

package per_node_controller

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store PerNodeControllers.
type Registry interface {
	ListPerNodeControllers(ctx api.Context) (*api.PerNodeControllerList, error)
	WatchPerNodeControllers(ctx api.Context, resourceVersion string) (watch.Interface, error)
	GetPerNodeController(ctx api.Context, controllerID string) (*api.PerNodeController, error)
	CreatePerNodeController(ctx api.Context, controller *api.PerNodeController) error
	UpdatePerNodeController(ctx api.Context, controller *api.PerNodeController) error
	DeletePerNodeController(ctx api.Context, controllerID string) error
}
