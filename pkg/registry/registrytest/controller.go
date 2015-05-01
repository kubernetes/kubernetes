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

package registrytest

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// TODO: Why do we have this AND MemoryRegistry?
type ControllerRegistry struct {
	Err         error
	Controllers *api.ReplicationControllerList
	sync.Mutex
}

func (r *ControllerRegistry) SetError(err error) {
	r.Lock()
	defer r.Unlock()
	r.Err = err
}

func (r *ControllerRegistry) ListControllers(ctx api.Context) (*api.ReplicationControllerList, error) {
	r.Lock()
	defer r.Unlock()
	return r.Controllers, r.Err
}

func (r *ControllerRegistry) GetController(ctx api.Context, ID string) (*api.ReplicationController, error) {
	r.Lock()
	defer r.Unlock()
	if r.Controllers != nil {
		for _, rc := range r.Controllers.Items {
			if ID == rc.Name {
				return &r.Controllers.Items[0], r.Err
			}
		}
	}
	return &api.ReplicationController{}, r.Err
}

func (r *ControllerRegistry) CreateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	r.Lock()
	defer r.Unlock()
	if r.Controllers != nil {
		r.Controllers.Items = append(r.Controllers.Items, *controller)
	}
	return controller, r.Err
}

func (r *ControllerRegistry) UpdateController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	r.Lock()
	defer r.Unlock()
	return controller, r.Err
}

func (r *ControllerRegistry) DeleteController(ctx api.Context, ID string) error {
	r.Lock()
	defer r.Unlock()
	return r.Err
}

func (r *ControllerRegistry) WatchControllers(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	r.Lock()
	defer r.Unlock()
	return nil, r.Err
}
