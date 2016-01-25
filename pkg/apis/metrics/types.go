/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package metrics

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Placeholder top-level node resource metrics.
type RawNode struct {
	unversioned.TypeMeta `json:",inline"`
}

// Placeholder top-level pod resource metrics.
type RawPod struct {
	unversioned.TypeMeta `json:",inline"`
}

// Node-level metrics.
type Node struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`
	Metrics              Metrics `json:"metrics,omitempty"`
}

// Pod-level metrics.
type Pod struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`
	Metrics              Metrics `json:"metrics,omitempty"`
}

// The latest available metrics for the appriopriate resource.
type Metrics struct {
	// StartTime and EndTime specifies the time window from which the response was returned.
	StartTime unversioned.Time `json:"start"`
	EndTime   unversioned.Time `json:"end"`
	// List of available resources' usage.
	Usage api.ResourceList `json:"usage",omitempty`
}
