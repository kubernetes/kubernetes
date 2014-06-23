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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type MockServiceRegistry struct {
	list      api.ServiceList
	err       error
	endpoints api.Endpoints
}

func (m *MockServiceRegistry) ListServices() (api.ServiceList, error) {
	return m.list, m.err
}

func (m *MockServiceRegistry) CreateService(svc api.Service) error {
	return m.err
}

func (m *MockServiceRegistry) GetService(name string) (*api.Service, error) {
	return nil, m.err
}

func (m *MockServiceRegistry) DeleteService(name string) error {
	return m.err
}

func (m *MockServiceRegistry) UpdateService(svc api.Service) error {
	return m.err
}

func (m *MockServiceRegistry) UpdateEndpoints(e api.Endpoints) error {
	m.endpoints = e
	return m.err
}
