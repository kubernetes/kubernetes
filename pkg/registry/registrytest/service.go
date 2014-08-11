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
)

type ServiceRegistry struct {
	List      api.ServiceList
	Err       error
	Endpoints api.Endpoints
}

func (r *ServiceRegistry) ListServices() (api.ServiceList, error) {
	return r.List, r.Err
}

func (r *ServiceRegistry) CreateService(svc api.Service) error {
	return r.Err
}

func (r *ServiceRegistry) GetService(name string) (*api.Service, error) {
	return nil, r.Err
}

func (r *ServiceRegistry) DeleteService(name string) error {
	return r.Err
}

func (r *ServiceRegistry) UpdateService(svc api.Service) error {
	return r.Err
}

func (r *ServiceRegistry) UpdateEndpoints(e api.Endpoints) error {
	r.Endpoints = e
	return r.Err
}
