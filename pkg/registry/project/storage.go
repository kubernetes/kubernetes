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

package project

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"code.google.com/p/go-uuid/uuid"
)

// RegistryStorage stores data for the replication controller service.
// It implements apiserver.RESTStorage.
type RegistryStorage struct {
	registry Registry
}

// NewRegistryStorage returns a new apiserver.RESTStorage for Project resources
func NewRegistryStorage(registry Registry) apiserver.RESTStorage {
	return &RegistryStorage{
		registry: registry,
	}
}

// Create registers the given Project
func (rs *RegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	project, ok := obj.(*api.Project)
	if !ok {
		return nil, fmt.Errorf("not a project: %#v", obj)
	}
	if len(project.ID) == 0 {
		project.ID = uuid.NewUUID().String()
	}
	if errs := api.ValidateProject(project); len(errs) > 0 {
		return nil, fmt.Errorf("Validation errors: %v", errs)
	}

	project.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := rs.registry.CreateProject(*project)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetProject(project.ID)
	}), nil
}

// Delete asynchronously deletes the Project specified by its id.
func (rs *RegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteProject(id)
	}), nil
}

// Get obtains the Project specified by its id.
func (rs *RegistryStorage) Get(id string) (interface{}, error) {
	project, err := rs.registry.GetProject(id)
	if err != nil {
		return nil, err
	}
	return project, err
}

// List obtains a list of Projects that match selector.
func (rs *RegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := api.ProjectList{}
	projects, err := rs.registry.ListProjects()
	if err == nil {
		for _, project := range projects {
			if selector.Matches(labels.Set(project.Labels)) {
				result.Items = append(result.Items, project)
			}
		}
	}
	return result, err
}

// New creates a new Project for use with Create and Update.
func (rs RegistryStorage) New() interface{} {
	return &api.Project{}
}

// Update replaces a given Project instance with an existing storage registry
func (rs *RegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	return nil, fmt.Errorf("Not supported")
}

// Watch returns Project events via a watch.Interface.
// It implements apiserver.ResourceWatcher.
func (rs *RegistryStorage) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("no field selector implemented for projects")
	}
	incoming, err := rs.registry.WatchProjects(resourceVersion)
	if err != nil {
		return nil, err
	}
	return watch.Filter(incoming, func(e watch.Event) (watch.Event, bool) {
		project := e.Object.(*api.Project)
		return e, label.Matches(labels.Set(project.Labels))
	}), nil
}
