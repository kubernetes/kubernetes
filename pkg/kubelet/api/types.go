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

package api

import (
	cadvisorv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// ContainerMetrics is a wrapper of cAdvisor per-container metrics.
type ContainerMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	// should contain container name -- FIXME
	api.ObjectMeta `json:"metadata,omitempty"`

	// TODO
	Spec *cadvisorv2.ContainerSpec

	// TODO
	Stats []*cadvisorv2.ContainerStats
}

// TODO
type PodMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	// should contain pod name -- FIXME
	api.ObjectMeta `json:"metadata,omitempty"`

	// TODO
	Containers []ContainerMetrics
}

// TODO
type RawMetrics struct {
	unversioned.TypeMeta `json:",inline"`
	// should contain node name -- FIXME
	api.ObjectMeta `json:"metadata,omitempty"`

	// TODO
	Machine ContainerMetrics // Overall machine usage

	// TODO
	SystemContainers []ContainerMetrics // System containers like /kubelet, /docker-daemon, etc.

	// TODO
	Pods []PodMetrics
}
