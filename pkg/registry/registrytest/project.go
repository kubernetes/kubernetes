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

type ProjectRegistry struct {
	Err      error
	Projects []api.Project
}

func (r *ProjectRegistry) ListProjects() ([]api.Project, error) {
	return r.Projects, r.Err
}

func (r *ProjectRegistry) GetProject(ID string) (*api.Project, error) {
	return &api.Project{}, r.Err
}

func (r *ProjectRegistry) CreateProject(controller api.Project) error {
	return r.Err
}

func (r *ProjectRegistry) UpdateProject(controller api.Project) error {
	return r.Err
}

func (r *ProjectRegistry) DeleteProject(ID string) error {
	return r.Err
}

func (r *ProjectRegistry) WatchProjects(resourceVersion uint64) (watch.Interface, error) {
	return nil, r.Err
}
