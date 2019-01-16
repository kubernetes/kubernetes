/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"github.com/go-openapi/spec"
)

// AggregationManager is the interface between OpenAPI Aggregator service and a controller
// that manages CRD openapi spec aggregation
type AggregationManager interface {
	// AddUpdateLocalAPIService allows adding/updating local API service with nil handler and
	// nil Spec.Service. This function can be used for local dynamic OpenAPI spec aggregation
	// management (e.g. CRD)
	AddUpdateLocalAPIServiceSpec(name string, spec *spec.Swagger, etag string) error
	RemoveAPIServiceSpec(apiServiceName string) error
}
