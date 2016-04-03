// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1

import kube_api "k8s.io/kubernetes/pkg/api"

// ExternalSink is the interface that needs to be implemented by all external storage backends.
type ExternalSink interface {
	// Registers a metric with the backend.
	Register([]MetricDescriptor) error
	// Unregisters a metric with the backend.
	Unregister([]MetricDescriptor) error
	// Stores input data into the backend.
	// Support input types are as follows:
	// 1. []Timeseries
	// TODO: Add map[string]string to support storing raw data like node or pod status, spec, etc.
	StoreTimeseries([]Timeseries) error
	// Stores events into the backend.
	StoreEvents([]kube_api.Event) error
	// Returns debug information specific to the sink.
	DebugInfo() string
	// Returns an user friendly string that describes the External Sink.
	Name() string
}
