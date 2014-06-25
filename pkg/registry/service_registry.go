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
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type ServiceRegistryStorage struct {
	registry ServiceRegistry
	cloud    cloudprovider.Interface
	machines MinionRegistry
}

func MakeServiceRegistryStorage(registry ServiceRegistry, cloud cloudprovider.Interface, machines MinionRegistry) apiserver.RESTStorage {
	return &ServiceRegistryStorage{
		registry: registry,
		cloud:    cloud,
		machines: machines,
	}
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

func (sr *ServiceRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	list, err := sr.registry.ListServices()
	if err != nil {
		return nil, err
	}
	var filtered []api.Service
	for _, service := range list.Items {
		if selector.Matches(labels.Set(service.Labels)) {
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
	return service, err
}

func (sr *ServiceRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	service, err := sr.registry.GetService(id)
	if err != nil {
		return nil, err
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		if service.CreateExternalLoadBalancer {
			var balancer cloudprovider.TCPLoadBalancer
			var ok bool
			if sr.cloud != nil {
				balancer, ok = sr.cloud.TCPLoadBalancer()
			}
			if ok && balancer != nil {
				err = balancer.DeleteTCPLoadBalancer(id, "us-central1")
				if err != nil {
					return nil, err
				}
			}
		}
		return api.Status{Status: api.StatusSuccess}, sr.registry.DeleteService(id)
	}), nil
}

func (sr *ServiceRegistryStorage) Extract(body []byte) (interface{}, error) {
	var svc api.Service
	err := api.DecodeInto(body, &svc)
	return svc, err
}

func (sr *ServiceRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	srv := obj.(api.Service)
	if srv.ID == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", srv)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		// TODO: Consider moving this to a rectification loop, so that we make/remove external load balancers
		// correctly no matter what http operations happen.
		if srv.CreateExternalLoadBalancer {
			var balancer cloudprovider.TCPLoadBalancer
			var ok bool
			if sr.cloud != nil {
				balancer, ok = sr.cloud.TCPLoadBalancer()
			}
			if ok && balancer != nil {
				hosts, err := sr.machines.List()
				if err != nil {
					return nil, err
				}
				err = balancer.CreateTCPLoadBalancer(srv.ID, "us-central1", srv.Port, hosts)
				if err != nil {
					return nil, err
				}
			} else {
				return nil, fmt.Errorf("requested an external service, but no cloud provider supplied.")
			}
		}
		// TODO actually wait for the object to be fully created here.
		err := sr.registry.CreateService(srv)
		if err != nil {
			return nil, err
		}
		return sr.registry.GetService(srv.ID)
	}), nil
}

func (sr *ServiceRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	srv := obj.(api.Service)
	if srv.ID == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", srv)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		// TODO: check to see if external load balancer status changed
		err := sr.registry.UpdateService(srv)
		if err != nil {
			return nil, err
		}
		return sr.registry.GetService(srv.ID)
	}), nil
}
