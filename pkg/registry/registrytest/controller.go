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

package registrytest

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// TODO: Why do we have this AND MemoryRegistry?
type ControllerRegistry struct {
	Err         error
	Controllers *api.ReplicationControllerList
}

func (r *ControllerRegistry) ListControllers() (*api.ReplicationControllerList, error) {
	return r.Controllers, r.Err
}

func (r *ControllerRegistry) GetController(ID string) (*api.ReplicationController, error) {
	return &api.ReplicationController{}, r.Err
}

func (r *ControllerRegistry) CreateController(controller api.ReplicationController) error {
	return r.Err
}

func (r *ControllerRegistry) UpdateController(controller api.ReplicationController) error {
	return r.Err
}

func (r *ControllerRegistry) DeleteController(ID string) error {
	return r.Err
}

func (r *ControllerRegistry) WatchControllers(resourceVersion uint64) (watch.Interface, error) {
	return nil, r.Err
}
