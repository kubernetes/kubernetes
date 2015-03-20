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

package endpoint

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// Registry is an interface for things that know how to store endpoints.
type Registry interface {
	ListEndpoints(ctx api.Context) (*api.EndpointsList, error)
	GetEndpoints(ctx api.Context, name string) (*api.Endpoints, error)
	WatchEndpoints(ctx api.Context, labels labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
	UpdateEndpoints(ctx api.Context, e *api.Endpoints) error
}
