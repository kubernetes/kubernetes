/*
Copyright 2014 The Kubernetes Authors.

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
	"sync"

	"k8s.io/kubernetes/pkg/api"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/watch"
)

func NewServiceRegistry() *ServiceRegistry {
	return &ServiceRegistry{}
}

type ServiceRegistry struct {
	mu      sync.Mutex
	List    api.ServiceList
	Service *api.Service
	Updates []api.Service
	Err     error

	DeletedID string
	GottenID  string
	UpdatedID string
}

func (r *ServiceRegistry) SetError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Err = err
}

func (r *ServiceRegistry) ListServices(ctx api.Context, options *api.ListOptions) (*api.ServiceList, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	ns, _ := api.NamespaceFrom(ctx)

	// Copy metadata from internal list into result
	res := new(api.ServiceList)
	res.TypeMeta = r.List.TypeMeta
	res.ListMeta = r.List.ListMeta

	if ns != api.NamespaceAll {
		for _, service := range r.List.Items {
			if ns == service.Namespace {
				res.Items = append(res.Items, service)
			}
		}
	} else {
		res.Items = append([]api.Service{}, r.List.Items...)
	}

	return res, r.Err
}

func (r *ServiceRegistry) CreateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.Service = new(api.Service)
	clone, err := api.Scheme.DeepCopy(svc)
	if err != nil {
		return nil, err
	}
	r.Service = clone.(*api.Service)

	r.List.Items = append(r.List.Items, *svc)
	return svc, r.Err
}

func (r *ServiceRegistry) GetService(ctx api.Context, id string, options *metav1.GetOptions) (*api.Service, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.GottenID = id
	return r.Service, r.Err
}

func (r *ServiceRegistry) DeleteService(ctx api.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.DeletedID = id
	r.Service = nil
	return r.Err
}

func (r *ServiceRegistry) UpdateService(ctx api.Context, svc *api.Service) (*api.Service, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.UpdatedID = svc.Name
	*r.Service = *svc
	r.Updates = append(r.Updates, *svc)
	return svc, r.Err
}

func (r *ServiceRegistry) WatchServices(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	return nil, r.Err
}

func (r *ServiceRegistry) ExportService(ctx api.Context, name string, options metav1.ExportOptions) (*api.Service, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.Service, r.Err
}
