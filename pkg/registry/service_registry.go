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
	"net/url"
	"strconv"
	"strings"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
)

type ServiceRegistry interface {
	ListServices() (ServiceList, error)
	CreateService(svc Service) error
	GetService(name string) (*Service, error)
	DeleteService(name string) error
	UpdateService(svc Service) error
	UpdateEndpoints(e Endpoints) error
}

type ServiceRegistryStorage struct {
	registry ServiceRegistry
}

func MakeServiceRegistryStorage(registry ServiceRegistry) apiserver.RESTStorage {
	return &ServiceRegistryStorage{registry: registry}
}

// GetServiceEnvironmentVariables populates a list of environment variables that are use
// in the container environment to get access to services.
func GetServiceEnvironmentVariables(registry ServiceRegistry, machine string) ([]EnvVar, error) {
	var result []EnvVar
	services, err := registry.ListServices()
	if err != nil {
		return result, err
	}
	for _, service := range services.Items {
		name := strings.ToUpper(service.ID) + "_SERVICE_PORT"
		value := strconv.Itoa(service.Port)
		result = append(result, EnvVar{Name: name, Value: value})
	}
	result = append(result, EnvVar{Name: "SERVICE_HOST", Value: machine})
	return result, nil
}

func (sr *ServiceRegistryStorage) List(*url.URL) (interface{}, error) {
	list, err := sr.registry.ListServices()
	list.Kind = "cluster#serviceList"
	return list, err
}

func (sr *ServiceRegistryStorage) Get(id string) (interface{}, error) {
	service, err := sr.registry.GetService(id)
	service.Kind = "cluster#service"
	return service, err
}

func (sr *ServiceRegistryStorage) Delete(id string) error {
	return sr.registry.DeleteService(id)
}

func (sr *ServiceRegistryStorage) Extract(body string) (interface{}, error) {
	var svc Service
	err := json.Unmarshal([]byte(body), &svc)
	svc.Kind = "cluster#service"
	return svc, err
}

func (sr *ServiceRegistryStorage) Create(obj interface{}) error {
	return sr.registry.CreateService(obj.(Service))
}

func (sr *ServiceRegistryStorage) Update(obj interface{}) error {
	return sr.registry.UpdateService(obj.(Service))
}
