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
	"errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"reflect"
)

// ProjectRESTStorage implements the apiserver.RESTStorage and apiserver.ResourceWatcher interface and enforces project scoping of delegated resource types.
type ProjectRESTStorage struct {
	project  apiserver.RESTStorage
	resource apiserver.RESTStorage
}

// NewProjectRESTStorage creates a new instance of ProjectRESTStorage
func NewProjectRESTStorage(project apiserver.RESTStorage, resource apiserver.RESTStorage) *ProjectRESTStorage {
	return &ProjectRESTStorage{
		project:  project,
		resource: resource,
	}
}

// fetchLabelMap uses reflection on the project-scoped resource to retrieve the Labels field on the struct
func (rs *ProjectRESTStorage) fetchLabelMap(obj interface{}) (map[string]string, error) {
	valuePtr := reflect.ValueOf(obj)
	value := valuePtr.Elem()
	labelMapValue := value.FieldByName("Labels")
	if labelMapValue.Kind() != reflect.Map {
		return nil, errors.New("Invalid object struct must have field with name: Labels to support project scoping")
	}
	result, ok := labelMapValue.Interface().(map[string]string)
	if !ok {
		return nil, errors.New("Invalid map to cast")
	}
	return result, nil
}

// fetchProjectId looks for a label on the object called "project" and returns its value if found.
func (rs *ProjectRESTStorage) fetchProjectId(obj interface{}) (string, error) {
	labelMap, err := rs.fetchLabelMap(obj)
	if err != nil {
		return "", err
	}
	if len(labelMap["project"]) == 0 {
		return "", errors.New("Missing project label on resource")
	}
	return labelMap["project"], nil
}

// Create enforces that a project-scoped resource has a project label that refers to a project that actually exists prior to persisting the project-scoped resource.
func (rs *ProjectRESTStorage) Create(obj interface{}) (<-chan interface{}, error) {
	// validate input object has a project=<projectId> label
	projectId, err := rs.fetchProjectId(obj)
	if err != nil {
		return nil, err
	}
	// validate input project exists
	if _, err = rs.project.Get(projectId); err != nil {
		return nil, err
	}
	// create the resource
	return rs.resource.Create(obj)
}

// Delete deletes the project-scoped resource
func (rs *ProjectRESTStorage) Delete(id string) (<-chan interface{}, error) {
	return rs.resource.Delete(id)
}

// Get obtains the project-scoped resource
func (rs *ProjectRESTStorage) Get(id string) (interface{}, error) {
	return rs.resource.Get(id)
}

// List obtains the project-scoped resource
// TODO - discuss if this is a choke-point for enforcing project-scoped resources are not enumerable without a project-context
func (rs *ProjectRESTStorage) List(selector labels.Selector) (interface{}, error) {
	return rs.resource.List(selector)
}

// New creates a new resource for persistence associated with a project
func (rs *ProjectRESTStorage) New() interface{} {
	return rs.resource.New()
}

// Update replaces a given resource with a new copy, but it enforces that the project label is immutable on the resource
func (rs *ProjectRESTStorage) Update(obj interface{}) (<-chan interface{}, error) {
	// TODO enforce that the project label is immutable and present on the resource
	return rs.resource.Update(obj)
}

// Watch delegates to the resource Watch method if its supported
func (rs *ProjectRESTStorage) Watch(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	watcher, ok := rs.resource.(apiserver.ResourceWatcher)
	if !ok {
		return nil, errors.New("Watch is not supported by resource")
	}
	return watcher.Watch(label, field, resourceVersion)
}
