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
	"encoding/json"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type ServiceRegistryStorage struct {
	registry ServiceRegistry
}

func MakeServiceRegistryStorage(registry ServiceRegistry) apiserver.RESTStorage {
	return &ServiceRegistryStorage{registry: registry}
}

// GetServiceEnvironmentVariables populates a list of environment variables that are use
// in the container environment to get access to services.
func GetServiceEnvironmentVariables(registry ServiceRegistry, machine string) ([]api.EnvVar, error) {
	var result []api.EnvVar
	services, err := registry.ListServices()
	if err != nil {
		return result, err
	}
	for _, service := range services.Items {
		name := strings.ToUpper(service.ID) + "_SERVICE_PORT"
		value := strconv.Itoa(service.Port)
		result = append(result, api.EnvVar{Name: name, Value: value})
	}
	result = append(result, api.EnvVar{Name: "SERVICE_HOST", Value: machine})
	return result, nil
}

func (sr *ServiceRegistryStorage) List(query labels.Query) (interface{}, error) {
	list, err := sr.registry.ListServices()
	if err != nil {
		return nil, err
	}
	list.Kind = "cluster#serviceList"
	var filtered []api.Service
	for _, service := range list.Items {
		if query.Matches(labels.Set(service.Labels)) {
			filtered = append(filtered, service)
		}
	}
	list.Items = filtered
	return list, err
}

func (sr *ServiceRegistryStorage) Get(id string) (interface{}, error) {
	service, err := sr.registry.GetService(id)
	if err != nil {
		return nil, err
	}
	service.Kind = "cluster#service"
	return service, err
}

func (sr *ServiceRegistryStorage) Delete(id string) error {
	return sr.registry.DeleteService(id)
}

func (sr *ServiceRegistryStorage) Extract(body string) (interface{}, error) {
	var svc api.Service
	err := json.Unmarshal([]byte(body), &svc)
	svc.Kind = "cluster#service"
	return svc, err
}

func (sr *ServiceRegistryStorage) Create(obj interface{}) error {
	return sr.registry.CreateService(obj.(api.Service))
}

func (sr *ServiceRegistryStorage) Update(obj interface{}) error {
	return sr.registry.UpdateService(obj.(api.Service))
}
