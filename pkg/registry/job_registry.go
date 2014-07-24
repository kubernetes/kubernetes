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

package registry

import (
	"fmt"

	"code.google.com/p/go-uuid/uuid"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// JobRegistryStorage is an implementation of RESTStorage for the api server.
type JobRegistryStorage struct {
	registry JobRegistry
}

func NewJobRegistryStorage(registry JobRegistry) apiserver.RESTStorage {
	return &JobRegistryStorage{
		registry: registry,
	}
}

// List obtains a list of Jobs that match selector.
func (storage *JobRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := api.JobList{}
	jobs, err := storage.registry.ListJobs()
	if err == nil {
		for _, job := range jobs.Items {
			result.Items = append(result.Items, job)
		}
	}
	return result, err
}

// Get obtains the job specified by its id.
func (storage *JobRegistryStorage) Get(id string) (interface{}, error) {
	job, err := storage.registry.GetJob(id)
	if err != nil {
		return nil, err
	}
	return job, err
}

// Delete asynchronously deletes the Job specified by its id.
func (storage *JobRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, storage.registry.DeleteJob(id)
	}), nil
}

// Extract deserializes user provided data into an api.Job.
func (storage *JobRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := api.Job{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create registers a given new Job instance to storage.registry.
func (storage *JobRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	job, ok := obj.(api.Job)
	if !ok {
		return nil, fmt.Errorf("not a job: %#v", obj)
	}
	if len(job.ID) == 0 {
		job.ID = uuid.NewUUID().String()
	}
	if len(job.State) == 0 {
		job.State = api.JobNew
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.CreateJob(job)
		if err != nil {
			return nil, err
		}
		return job, nil
	}), nil
}

// Update replaces a given Job instance with an existing instance in storage.registry.
func (storage *JobRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	job, ok := obj.(api.Job)
	if !ok {
		return nil, fmt.Errorf("not a job: %#v", obj)
	}
	if len(job.ID) == 0 {
		return nil, fmt.Errorf("ID should not be empty: %#v", job)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		err := storage.registry.UpdateJob(job)
		if err != nil {
			return nil, err
		}
		return job, nil
	}), nil
}
