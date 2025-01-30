/*
Copyright 2022 The Kubernetes Authors.

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

package app

// These definitions are sufficient to generate simple CDI files which set env
// variables in a container, which is all that the test driver does.  A real
// driver will either use pre-generated CDI files or can use the
// github.com/container-orchestrated-devices/container-device-interface/pkg/cdi
// helper package to generate files.
//
// This is not done in Kubernetes to minimize dependencies.

// spec is the base configuration for CDI.
type spec struct {
	Version string `json:"cdiVersion"`
	Kind    string `json:"kind"`

	Devices []device `json:"devices"`
}

// device is a "Device" a container runtime can add to a container.
type device struct {
	Name           string         `json:"name"`
	ContainerEdits containerEdits `json:"containerEdits"`
}

// containerEdits are edits a container runtime must make to the OCI spec to expose the device.
type containerEdits struct {
	Env []string `json:"env,omitempty"`
}
