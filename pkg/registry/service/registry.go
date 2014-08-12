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

package service

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Registry is an interface for things that know how to store services.
type Registry interface {
	ListServices() (api.ServiceList, error)
	CreateService(svc api.Service) error
	GetService(name string) (*api.Service, error)
	DeleteService(name string) error
	UpdateService(svc api.Service) error
	UpdateEndpoints(e api.Endpoints) error
}
