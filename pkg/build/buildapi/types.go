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

package buildapi

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/buildconfig/buildconfigapi"
)

// Build encapsulates the inputs needed to produce a new deployable image, as well as
// the status of the operation and a reference to the Pod which runs the build.
type Build struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Config       buildconfigapi.BuildConfig `json:"config,omitempty" yaml:"config,omitempty"`
	Status       BuildStatus                `json:"status,omitempty" yaml:"status,omitempty"`
	PodID        string                     `json:"podID,omitempty" yaml:"podID,omitempty"`
}

// BuildStatus represents the status of a Build at a point in time.
type BuildStatus string

const (
	BuildNew      BuildStatus = "new"
	BuildPending  BuildStatus = "pending"
	BuildRunning  BuildStatus = "running"
	BuildComplete BuildStatus = "complete"
	BuildFailed   BuildStatus = "failed"
)

// BuildList is a collection of Builds.
type BuildList struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Items        []Build `json:"items,omitempty" yaml:"items,omitempty"`
}

func init() {
	api.AddKnownTypes("",
		Build{},
		BuildList{},
	)

	api.AddKnownTypes("v1beta1",
		Build{},
		BuildList{},
	)
}
