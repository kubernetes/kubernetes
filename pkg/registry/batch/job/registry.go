/*
Copyright 2018 The Kubernetes Authors.

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

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/batch"
)

// Registry is an interface for things that know how to store Jobs.
type Registry interface {
	ListJobs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*batch.JobList, error)
	GetJob(ctx genericapirequest.Context, jobID string, options *metav1.GetOptions) (*batch.Job, error)
	CreateJob(ctx genericapirequest.Context, job *batch.Job, createValidation rest.ValidateObjectFunc) (*batch.Job, error)
	UpdateJob(ctx genericapirequest.Context, job *batch.Job, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*batch.Job, error)
	DeleteJob(ctx genericapirequest.Context, jobID string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListJobs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*batch.JobList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*batch.JobList), err
}

func (s *storage) GetJob(ctx genericapirequest.Context, jobID string, options *metav1.GetOptions) (*batch.Job, error) {
	obj, err := s.Get(ctx, jobID, options)
	if err != nil {
		return nil, err
	}
	return obj.(*batch.Job), nil
}

func (s *storage) CreateJob(ctx genericapirequest.Context, job *batch.Job, createValidation rest.ValidateObjectFunc) (*batch.Job, error) {
	obj, err := s.Create(ctx, job, createValidation, false)
	if err != nil {
		return nil, err
	}
	return obj.(*batch.Job), nil
}

func (s *storage) UpdateJob(ctx genericapirequest.Context, job *batch.Job, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (*batch.Job, error) {
	obj, _, err := s.Update(ctx, job.Name, rest.DefaultUpdatedObjectInfo(job), createValidation, updateValidation)
	if err != nil {
		return nil, err
	}
	return obj.(*batch.Job), nil
}

func (s *storage) DeleteJob(ctx genericapirequest.Context, jobID string) error {
	_, _, err := s.Delete(ctx, jobID, nil)
	return err
}
