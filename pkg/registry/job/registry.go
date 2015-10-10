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

package job

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store Jobs.
type Registry interface {
	// ListJobs obtains a list of Jobs having labels and fields which match selector.
	ListJobs(ctx api.Context, label labels.Selector, field fields.Selector) (*extensions.JobList, error)
	// WatchJobs watch for new/changed/deleted Jobs.
	WatchJobs(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	// GetJobs gets a specific Job.
	GetJob(ctx api.Context, name string) (*extensions.Job, error)
	// CreateJob creates a Job based on a specification.
	CreateJob(ctx api.Context, job *extensions.Job) (*extensions.Job, error)
	// UpdateJob updates an existing Job.
	UpdateJob(ctx api.Context, job *extensions.Job) (*extensions.Job, error)
	// DeleteJob deletes an existing Job.
	DeleteJob(ctx api.Context, name string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListJobs(ctx api.Context, label labels.Selector, field fields.Selector) (*extensions.JobList, error) {
	if !field.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, label, field)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.JobList), err
}

func (s *storage) WatchJobs(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return s.Watch(ctx, label, field, resourceVersion)
}

func (s *storage) GetJob(ctx api.Context, name string) (*extensions.Job, error) {
	obj, err := s.Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Job), nil
}

func (s *storage) CreateJob(ctx api.Context, job *extensions.Job) (*extensions.Job, error) {
	obj, err := s.Create(ctx, job)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Job), nil
}

func (s *storage) UpdateJob(ctx api.Context, job *extensions.Job) (*extensions.Job, error) {
	obj, _, err := s.Update(ctx, job)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Job), nil
}

func (s *storage) DeleteJob(ctx api.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
