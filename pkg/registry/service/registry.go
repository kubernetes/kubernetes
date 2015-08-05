/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package service

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store services.
type Registry interface {
	ListServices(ctx api.Context) (*api.ServiceList, error)
	CreateService(ctx api.Context, svc *api.Service) (*api.Service, error)
	GetService(ctx api.Context, name string) (*api.Service, error)
	DeleteService(ctx api.Context, name string) error
	UpdateService(ctx api.Context, svc *api.Service) (*api.Service, error)
	WatchServices(ctx api.Context, labels labels.Selector, fields fields.Selector, resourceVersion string) (watch.Interface, error)
}

// TODO: Move to a general location (as other components may need allocation in future; it's not service specific)
// RangeRegistry is a registry that can retrieve or persist a RangeAllocation object.
type RangeRegistry interface {
	// Get returns the latest allocation, an empty object if no allocation has been made,
	// or an error if the allocation could not be retrieved.
	Get() (*api.RangeAllocation, error)
	// CreateOrUpdate should create or update the provide allocation, unless a conflict
	// has occured since the item was last created.
	CreateOrUpdate(*api.RangeAllocation) error
}
